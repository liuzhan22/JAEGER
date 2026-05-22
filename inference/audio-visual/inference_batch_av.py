# ./inference/av/inference_batch_av.py

import argparse
import yaml
import torch
import librosa
import os
import sys
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.SALMONN3D import SALMONN3D
from transformers import Qwen2_5OmniProcessor
from safetensors.torch import load_file

# ==================== Helper Functions ====================

def _generate_ordinal_prompt(template, num_speakers):
    """
    根据speaker数量动态生成序数词prompt
    """
    if num_speakers <= 0:
        return template
    
    # 生成序数词序列
    ordinals = []
    for i in range(1, num_speakers + 1):
        if i == 1:
            ordinals.append("1st")
        elif i == 2:
            ordinals.append("2nd")
        elif i == 3:
            ordinals.append("3rd")
        else:
            ordinals.append(f"{i}th")
    
    ordinal_list = ", ".join(ordinals)
    
    # 替换prompt中的{ordinals}占位符
    try:
        return template.format(ordinals=ordinal_list)
    except KeyError:
        # 如果没有{ordinals}占位符，保持原样
        return template

def _depth_to_point_cloud(depth_img, target_resolution=(672, 378), hfov=90.0, vfov=58.72):
    if depth_img.size != target_resolution:
        depth_img = depth_img.resize(target_resolution, Image.NEAREST)
        
    w, h = depth_img.size
    depth = np.array(depth_img).astype(np.float32)
    
    depth = depth / 255 * 10.0

    fx = (w / 2) / np.tan(np.deg2rad(hfov) / 2)
    fy = (h / 2) / np.tan(np.deg2rad(vfov) / 2)
    cx = w / 2
    cy = h / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z_cam = -depth
    x_cam = (u - cx) * depth / fx
    y_cam = -(v - cy) * depth / fy
    
    point_cloud = np.stack([x_cam, y_cam, z_cam], axis=-1)  
    point_cloud = point_cloud.reshape(-1, 3) 
    return point_cloud

# ==================== Main Workflow ====================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="configs/decode_config.yaml")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    return parser.parse_args()

def main():
    args = get_args()

    device = torch.device("cuda:0")
    print(f"[Rank {args.rank}/{args.world_size}] Initialized on device {device} (PID: {os.getpid()})")

    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    dataset_cfg = config["datasets"]
    
    ckpt_path = model_cfg.get("ckpt", "")
    test_ann_path = dataset_cfg.get("test_ann_path")
    test_prompt_path = model_cfg.get("test_prompt_path")
    
    output_dir = args.output_dir
    if args.rank == 0 and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"[Rank {args.rank}] Loading SALMONN3D model base ...")
    model = SALMONN3D.from_config(model_cfg).to(device)
    processor = model.processor

    if ckpt_path and os.path.exists(ckpt_path):
        if args.rank == 0:
            print(f"[Rank {args.rank}] Loading checkpoint from {ckpt_path} ...")
        state_dict = load_file(ckpt_path) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        msg = model.load_state_dict(state_dict, strict=False)
        if args.rank == 0:
            print(f"Load Status: {msg}")
    else:
        print(f"[Rank {args.rank}] Warning: No checkpoint provided! Using initialized weights.")

    model.eval()

    with open(test_prompt_path, "r", encoding="utf-8") as f:
        prompt_cfg = json.load(f)
    
    with open(test_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)["annotation"]

    if args.max_samples > 0:
        random.seed(42)
        data = random.sample(data, args.max_samples) if len(data) > args.max_samples else data

    # ==================== DATA GROUPING & INFO EXTRACTION ====================
    by_path = defaultdict(dict)
    for ann in data:
        path = ann["path"]
        task = ann["task"]
        
        info = {
            "gt_text": ann.get("text", ""),
            "obj_category": ann.get("obj_category", []),
            "gender": ann.get("gender", None), 
            "category": ann.get("category", ""), # Added for VGG task
            "num_speakers": ann.get("num_speakers", 0)
        }
        
        by_path[path][task] = info
        
        if "hfov" in ann:
            by_path[path]["_meta"] = {"hfov": ann["hfov"], "vfov": ann.get("vfov", 58.72)}

    all_paths = sorted(list(by_path.keys()))
    my_paths = all_paths[args.rank::args.world_size]
    my_by_path = {p: by_path[p] for p in my_paths}
    print(f"[Rank {args.rank}] Processing {len(my_by_path)} samples.")

    results = []
    
    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
    }

    TARGET_RES = (672, 378)

    for path, tasks in tqdm(my_by_path.items(), position=args.rank, desc=f"Rank {args.rank}"):
        
        rgb_path = os.path.join(path, "rgb.png")
        depth_path = os.path.join(path, "depth.png")
        
        # Audio fallback logic prioritizing foa_vgg.wav
        foa_path = os.path.join(path, "foa_vgg.wav")
        if not os.path.exists(foa_path):
            foa_ls_path = os.path.join(path, "foa_ls.wav")
            if os.path.exists(foa_ls_path):
                foa_path = foa_ls_path
            elif os.path.exists(path + ".wav"):
                foa_path = path + ".wav"

        if not (os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(foa_path)):
            continue

        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
            if rgb_image.size != TARGET_RES:
                rgb_image = rgb_image.resize(TARGET_RES, Image.BILINEAR)

            meta = tasks.get("_meta", {})
            depth_img = Image.open(depth_path)
            point_cloud_np = _depth_to_point_cloud(depth_img, TARGET_RES, meta.get("hfov", 90.0), meta.get("vfov", 58.72))

            y, sr = librosa.load(foa_path, sr=16000, mono=False)
            if y.ndim == 1:
                continue 
            if y.shape[0] != 4 and y.shape[1] == 4:
                y = y.T
            if y.shape[0] != 4:
                continue 

            ch0 = y[0]  
            foa_np = y  

        except Exception:
            continue

        for task_type, task_info in tasks.items():
            if task_type.startswith("_"): continue 
            if task_type not in prompt_cfg: continue
            
            template = prompt_cfg[task_type]
            user_text = ""

            # ==================== PROMPT LOGIC ====================
            if task_type == "dual_source":
                gender = task_info.get("gender", "unknown")
                try:
                    user_text = template.format(gender=gender)
                except KeyError:
                    user_text = template
            
            elif task_type == "dual_source_vgg":
                category = task_info.get("category", "unknown")
                try:
                    user_text = template.format(category=category)
                except KeyError:
                    user_text = template

            elif task_type == "single_source":
                user_text = template
            
            elif task_type == "dual_source_more_cand":
                gender = task_info.get("gender", "unknown")
                num_speakers = task_info.get("num_speakers", 0)
                try:
                    formatted_template = template.format(gender=gender)
                    user_text = _generate_ordinal_prompt(formatted_template, num_speakers)
                except KeyError:
                    user_text = _generate_ordinal_prompt(template, num_speakers)

            elif task_type == "single_source_more_cand":
                num_speakers = task_info.get("num_speakers", 0)
                user_text = _generate_ordinal_prompt(template, num_speakers)
            
            else:
                user_text = template

            conversation = [
                system_prompt,
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb_path}, 
                        {"type": "audio", "audio": None}, 
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            
            text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            inputs = processor(
                text=text_input, 
                images=[rgb_image], 
                audio=[ch0],
                sampling_rate=16000,
                return_tensors="pt"
            ).to(device)

            pc_tensor = torch.from_numpy(point_cloud_np).unsqueeze(0).to(device).to(torch.bfloat16)
            inputs["point_clouds"] = pc_tensor
            
            raw_wav_tensor = torch.from_numpy(foa_np).unsqueeze(0).to(device).to(torch.bfloat16)
            inputs["raw_wav"] = raw_wav_tensor
            inputs["raw_wav_lens"] = torch.tensor([foa_np.shape[1]], dtype=torch.long).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
            
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            results.append({
                "path": path,
                "task": task_type,
                "gender": task_info.get("gender", None), 
                "category": task_info.get("category", ""), 
                "gt_text": task_info["gt_text"],
                "pred_text": pred_text
            })

    out_file = os.path.join(output_dir, f"av_results_rank_{args.rank}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
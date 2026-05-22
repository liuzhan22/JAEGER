# ./inference/visual-only/inference_batch_visual.py

import argparse
import yaml
import torch
import os
import sys
import json
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.SALMONN3D import SALMONN3D
from transformers import Qwen2_5OmniProcessor
from safetensors.torch import load_file

# ==================== Helper Functions ====================

def _depth_to_point_cloud(depth_img, target_resolution=(672, 378), hfov=90.0, vfov=58.72):
    """
    Convert a depth image to a point cloud.
    """
    if depth_img.size != target_resolution:
        depth_img = depth_img.resize(target_resolution, Image.NEAREST)
        
    w, h = depth_img.size
    depth = np.array(depth_img).astype(np.float32)
    
    # Convert depth map to meter (assuming 10.0 scale)
    depth = depth / 255 * 10.0

    # Camera intrinsics
    fx = (w / 2) / np.tan(np.deg2rad(hfov) / 2)
    fy = (h / 2) / np.tan(np.deg2rad(vfov) / 2)
    cx = w / 2
    cy = h / 2

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z_cam = -depth
    x_cam = (u - cx) * depth / fx
    y_cam = -(v - cy) * depth / fy
    
    point_cloud = np.stack([x_cam, y_cam, z_cam], axis=-1)  # [H, W, 3]
    point_cloud = point_cloud.reshape(-1, 3)  # [H*W, 3]
    return point_cloud

# ==================== Main Workflow ====================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", type=str, default="configs/decode_config.yaml", help="Path to decode config.")
    parser.add_argument("--rank", type=int, default=0, help="Current process rank")
    parser.add_argument("--world-size", type=int, default=1, help="Total number of processes")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output dir defined in config.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Limit number of samples for debugging")
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Device Setup
    device = torch.device("cuda:0")
    print(f"[Rank {args.rank}/{args.world_size}] Initialized on device {device} (PID: {os.getpid()})")

    # 2. Load Config
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    dataset_cfg = config["datasets"]
    
    ckpt_path = model_cfg.get("ckpt", "")
    test_ann_path = dataset_cfg.get("test_ann_path")
    test_prompt_path = model_cfg.get("test_prompt_path")
    
    output_dir = args.output_dir
    if args.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # 3. Load Model
    print(f"[Rank {args.rank}] Loading SALMONN3D model base ...")
    model = SALMONN3D.from_config(model_cfg).to(device)
    processor = model.processor

    # 4. Load Checkpoint
    if ckpt_path and os.path.exists(ckpt_path):
        if args.rank == 0:
            print(f"[Rank {args.rank}] Loading checkpoint from {ckpt_path} ...")
        state_dict = load_file(ckpt_path) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"[Rank {args.rank}] ⚠️ No checkpoint provided! Using initialized weights.")

    model.eval()

    # 5. Prepare Data
    with open(test_prompt_path, "r", encoding="utf-8") as f:
        prompt_cfg = json.load(f)
    
    with open(test_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)["annotation"]

    # Debug with fewer samples
    if args.max_samples > 0:
        random.seed(42)
        data = random.sample(data, args.max_samples) if len(data) > args.max_samples else data

    # Group by path to handle multi-task or efficient loading
    by_path = defaultdict(dict)
    for ann in data:
        # Save all necessary metadata here
        by_path[ann["path"]][ann["task"]] = {
            "gt_text": ann.get("text", ""),           # 原始 text (Bbox信息)
            "obj_category": ann.get("obj_category", []) # 原始 obj_category 列表
        }
        # Handle Metadata (FOV)
        if "hfov" in ann:
            by_path[ann["path"]]["_meta"] = {"hfov": ann["hfov"], "vfov": ann.get("vfov", 58.72)}

    # ==================== DATA SHARDING ====================
    all_paths = sorted(list(by_path.keys()))
    my_paths = all_paths[args.rank::args.world_size]
    my_by_path = {p: by_path[p] for p in my_paths}
    print(f"[Rank {args.rank}] Processing {len(my_by_path)} samples.")
    # =======================================================

    results = []

    # 6. Inference Loop
    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
    }

    TARGET_RES = (672, 378)

    for path, tasks in tqdm(my_by_path.items(), position=args.rank, desc=f"Rank {args.rank}"):
        # base_name = os.path.basename(path)
        
        # --- Visual Loading ---
        rgb_path = os.path.join(path, "rgb.png")
        depth_path = os.path.join(path, "depth.png")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        try:
            # Load RGB
            rgb_image = Image.open(rgb_path).convert("RGB")
            if rgb_image.size != TARGET_RES:
                rgb_image = rgb_image.resize(TARGET_RES, Image.BILINEAR)

            # Load Depth & Point Cloud
            meta = tasks.get("_meta", {})
            hfov = meta.get("hfov", 90.0)
            vfov = meta.get("vfov", 58.72)
            
            depth_img = Image.open(depth_path)
            point_cloud_np = _depth_to_point_cloud(depth_img, TARGET_RES, hfov, vfov)

        except Exception as e:
            print(f"[Rank {args.rank}] Error loading images {path}: {e}")
            continue

        for task_type, task_info in tasks.items():
            if task_type.startswith("_"): continue
            if task_type not in prompt_cfg: continue
            
            # --- Construct Prompt ---
            user_text_template = prompt_cfg[task_type]
            
            if task_type == "visual_grounding":
                obj_cats = task_info["obj_category"]
                # Create natural language list: "A, B, and C"
                if isinstance(obj_cats, list) and len(obj_cats) > 0:
                    cat_str = ", ".join(obj_cats[:-1]) + f", and {obj_cats[-1]}" if len(obj_cats) > 1 else obj_cats[0]
                elif isinstance(obj_cats, str):
                    cat_str = obj_cats
                else:
                    cat_str = "objects"
                user_text = user_text_template.format(obj_category=cat_str)
            else:
                user_text = user_text_template

            # --- Conversation ---
            conversation = [
                system_prompt,
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": rgb_path}, 
                        {"type": "text", "text": user_text},
                    ],
                },
            ]
            
            text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            if task_type == "visual_grounding":
                force_prefix = "bbox_0=Bbox(speaker,"
                text_input += force_prefix

            # --- Inputs ---
            inputs = processor(
                text=text_input, 
                images=[rgb_image], 
                return_tensors="pt"
            ).to(device)

            pc_tensor = torch.from_numpy(point_cloud_np).unsqueeze(0).to(device).to(torch.bfloat16)
            inputs["point_clouds"] = pc_tensor

            # --- Generate ---
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
            
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            if force_prefix:
                pred_text = f"{force_prefix}{pred_text}"

            # --- 7. Save Result (Requested Fields) ---
            results.append({
                "path": path,                           # 原始路径
                "task": task_type,
                "text": task_info["gt_text"],           # 原始 GT (bbox info)
                "obj_category": task_info["obj_category"], # 原始物体类别列表
                "pred_text": pred_text                  # 推理结果
            })

    # Save to JSON
    out_file = os.path.join(output_dir, f"visual_results_rank_{args.rank}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"[Rank {args.rank}] ✅ Saved results to {out_file}")

if __name__ == "__main__":
    main()
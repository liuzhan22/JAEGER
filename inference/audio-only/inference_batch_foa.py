# ./inference/audio-only/inference_batch_foa.py

import argparse
import yaml
import torch
import librosa
import os
import sys
import json
import math
import re
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.SALMONN3D import SALMONN3D
from transformers import Qwen2_5OmniProcessor
from safetensors.torch import load_file

# ==================== Metrics & Parsing Functions ====================

def parse_gt_doa(text: str):
    az_match = re.search(r"(?i)azimuth[:\s]+([-+]?\d*\.?\d+)", text)
    el_match = re.search(r"(?i)elevation[:\s]+([-+]?\d*\.?\d+)", text)

    az = float(az_match.group(1)) if az_match else None
    el = float(el_match.group(1)) if el_match else None
    
    return az, el

def parse_gt_distance(text: str):
    text = text.replace("m", " ")
    parts = text.replace(":", " ").split()
    for p in parts:
        try:
            return float(p)
        except ValueError:
            continue
    return None

def parse_pred_label(text: str, candidates):
    text_low = text.lower()
    best = None
    best_score = -1
    for c in candidates:
        c_low = c.lower()
        if c_low in text_low:
            score = len(c_low)
        else:
            score = 0
        if score > best_score:
            best_score = score
            best = c
    return best

def parse_pred_doa(text: str):
    return parse_gt_doa(text)

def parse_pred_distance(text: str):
    return parse_gt_distance(text)

def angular_error(az1, el1, az2, el2):
    if None in (az1, el1, az2, el2):
        return None

    def sph_to_vec(az, el):
        az_r = math.radians(az)
        el_r = math.radians(el)
        x = math.cos(el_r) * math.cos(az_r)
        y = math.cos(el_r) * math.sin(az_r)
        z = math.sin(el_r)
        return x, y, z

    x1, y1, z1 = sph_to_vec(az1, el1)
    x2, y2, z2 = sph_to_vec(az2, el2)
    dot = x1 * x2 + y1 * y2 + z1 * z2
    dot = max(min(dot, 1.0), -1.0)
    angle = math.degrees(math.acos(dot))
    return angle

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

    print(f"[Rank {args.rank}] Loading config from {args.cfg_path} ...")
    with open(args.cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    dataset_cfg = config["datasets"]
    run_cfg = config.get("run", {})

    ckpt_path = model_cfg.get("ckpt", "")
    test_ann_path = dataset_cfg.get("test_ann_path")
    test_prompt_path = model_cfg.get("test_prompt_path")
    
    output_dir = args.output_dir if args.output_dir else run_cfg.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Rank {args.rank}] Loading SALMONN3D model base ...")
    model = SALMONN3D.from_config(model_cfg).to(device)
    processor = model.processor

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[Rank {args.rank}] Loading checkpoint from {ckpt_path} ...")
        state_dict = load_file(ckpt_path) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        msg = model.load_state_dict(state_dict, strict=False)
        
        if args.rank == 0:
            foa_missing = [k for k in msg.missing_keys if "foa_iv_encoder" in k]
            if not foa_missing:
                print("FOA Encoder loaded correctly.")
            else:
                print(f"FOA Encoder missing keys: {len(foa_missing)}")

    model.eval()

    if not os.path.exists(test_ann_path):
        raise FileNotFoundError(f"Test annotation file not found: {test_ann_path}")
    if not os.path.exists(test_prompt_path):
        raise FileNotFoundError(f"Test prompt file not found: {test_prompt_path}")

    with open(test_prompt_path, "r", encoding="utf-8") as f:
        prompt_cfg = json.load(f)
    
    with open(test_ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)["annotation"]

    target_samples = args.max_samples if args.max_samples > 0 else 3000
    if 0 < target_samples < len(data):
        random.seed(42)
        data = random.sample(data, target_samples)

    if args.max_samples > 0:
        data = data[: args.max_samples]

    aed_labels = sorted({item["text"] for item in data if item["task"] == "audio_event_detection"})
    
    # Extract info
    by_path = defaultdict(dict)
    for ann in data:
        path = ann["path"]
        task = ann["task"]
        info = {
            "gt_text": ann.get("text", ""),
            "category": ann.get("category", "sound"),
            "gender": ann.get("gender", "unknown")
        }
        by_path[path][task] = info

    all_paths = sorted(list(by_path.keys())) 
    my_paths = all_paths[args.rank::args.world_size]
    my_by_path = {p: by_path[p] for p in my_paths}
    
    print(f"[Rank {args.rank}] Processing {len(my_by_path)} samples (Total: {len(all_paths)})")

    results = {
        "audio_event_detection": [],
        "spatial_doa": [],
        "multi_spatial_doa": [],
        "distance_estimation": [],
        "spatial_doa_vgg": [],
        "multi_spatial_doa_vgg": [],
        "starss23": []
    }
    
    metrics = {
        "aed_correct": 0, "aed_total": 0,
        "doa_errors": [],
        "dist_errors": []
    }

    system_prompt = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            }
        ],
    }

    for path, tasks in tqdm(my_by_path.items(), position=args.rank, desc=f"Rank {args.rank}"):
        # For starss23 task, path is already the full audio file path
        if any(task_type == "starss23" for task_type in tasks.keys()):
            foa_path = path
        else:
            foa_path = os.path.join(path, "foa_ls.wav")
            
        # Fallback handling for non-starss23 tasks
        if not os.path.exists(foa_path) and not any(task_type == "starss23" for task_type in tasks.keys()):
            foa_ls_path = os.path.join(path, "foa_ls.wav")
            if os.path.exists(foa_ls_path):
                foa_path = foa_ls_path
            elif os.path.exists(path + ".wav"):
                foa_path = path + ".wav"
            else:
                continue

        try:
            y, sr = librosa.load(foa_path, sr=16000, mono=False)
            
            # Handle starss23 mono audio by converting to pseudo-FOA format
            if any(task_type == "starss23" for task_type in tasks.keys()):
                if y.ndim == 1:
                    # Mono audio: convert to 4-channel pseudo-FOA
                    y = np.tile(y, (4, 1))
                elif y.ndim == 2 and y.shape[0] == 1:
                    # Single channel: convert to 4-channel pseudo-FOA
                    y = np.tile(y, (4, 1))
            
            if y.ndim == 1 or y.shape[0] != 4:
                if y.ndim > 1 and y.shape[1] == 4:
                    y = y.T
                else:
                    continue
            
            ch0 = y[0]  
            foa_np = y  
        except Exception:
            continue

        for task_type, ann_item in tasks.items():
            if task_type not in prompt_cfg:
                continue
            
            gt_text = ann_item["gt_text"]
            template = prompt_cfg[task_type]
            
            # Format target tasks
            if task_type == "multi_spatial_doa_vgg":
                user_text = template.format(category=ann_item["category"])
            elif task_type == "multi_spatial_doa":
                raw_gender = ann_item["gender"]
                gender_map = {"M": "male", "F": "female"}
                gender_str = gender_map.get(raw_gender, raw_gender)
                user_text = template.format(gender=gender_str)
            elif task_type == "starss23":
                raw_gender = ann_item["gender"]
                gender_map = {"M": "male", "F": "female"}
                gender_str = gender_map.get(raw_gender, raw_gender)
                user_text = template.format(gender=gender_str)
            else:
                user_text = template

            conversation = [
                system_prompt,
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": None},
                        {"type": "text", "text": user_text},
                    ],
                },
            ]

            text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            force_prefix = ""
            if task_type in ["spatial_doa", "multi_spatial_doa", "spatial_doa_vgg", "multi_spatial_doa_vgg", "starss23"]:
                force_prefix = "azimuth: "
                text_input += force_prefix  

            inputs = processor(text=text_input, audio=[ch0], sampling_rate=16000, return_tensors="pt").to(device)
            inputs["raw_wav"] = torch.from_numpy(foa_np).unsqueeze(0).to(device).to(torch.bfloat16)
            inputs["raw_wav_lens"] = torch.tensor([foa_np.shape[1]], dtype=torch.long).to(device)

            generate_cfg = {
                "max_new_tokens": 256,
                "do_sample": False,
                "num_beams": 1,
            }

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generate_cfg)
            
            input_len = inputs.input_ids.shape[1]
            generated_ids = output_ids[:, input_len:]
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            if force_prefix:
                pred_text = f"{force_prefix}{pred_text}"

            if task_type == "audio_event_detection":
                metrics["aed_total"] += 1
                pred_label = parse_pred_label(pred_text, aed_labels)
                if pred_label == gt_text:
                    metrics["aed_correct"] += 1
                
                results["audio_event_detection"].append({
                    "path": path,
                    "gt_label": gt_text,
                    "pred_label": pred_label,
                    "raw_output": pred_text
                })

            elif task_type in ["spatial_doa", "multi_spatial_doa", "spatial_doa_vgg", "multi_spatial_doa_vgg", "starss23"]:
                gt_az, gt_el = parse_gt_doa(gt_text)
                pred_az, pred_el = parse_pred_doa(pred_text)
                err = angular_error(gt_az, gt_el, pred_az, pred_el)
                
                if err is not None:
                    metrics["doa_errors"].append(err)

                record = {
                    "path": path,
                    "gt_az": gt_az,
                    "gt_el": gt_el,
                    "pred_az": pred_az,
                    "pred_el": pred_el,
                    "angular_error": err,
                    "raw_output": pred_text
                }

                if task_type in ["spatial_doa_vgg", "multi_spatial_doa_vgg"]:
                    record["category"] = ann_item["category"]
                elif task_type in ["spatial_doa", "multi_spatial_doa", "starss23"]:
                    record["gender"] = ann_item["gender"]

                if task_type == "starss23":
                    results["starss23"].append(record)
                else:
                    results[task_type].append(record)

            elif task_type == "distance_estimation":
                gt_dist = parse_gt_distance(gt_text)
                pred_dist = parse_pred_distance(pred_text)
                dist_err = None
                
                if pred_dist is not None and gt_dist is not None:
                    dist_err = abs(pred_dist - gt_dist)
                    metrics["dist_errors"].append(dist_err)

                results["distance_estimation"].append({
                    "path": path,
                    "gt_dist": gt_dist,
                    "pred_dist": pred_dist,
                    "abs_error": dist_err,
                    "raw_output": pred_text
                })

    output_obj = {
        "config": config,
        "details": results, 
    }

    out_file = os.path.join(output_dir, f"eval_results_rank_{args.rank}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
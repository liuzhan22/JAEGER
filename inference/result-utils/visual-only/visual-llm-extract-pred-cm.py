import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import json
import argparse
import re
import numpy as np
from vllm import LLM, SamplingParams

# ==================== 配置 ====================
# 请根据实际情况修改模型路径
MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct" 
DEFAULT_INPUT_FILE = "/path/to/results/file.json"

# ==================== 核心计算函数 ====================

def calculate_iou_3d(box1, box2):
    """
    计算两个 3D BBox 的 IoU (单位必须统一，建议均为米)
    box: [x, y, z, dx, dy, dz] (Center, Size)
    """
    if not box1 or not box2 or len(box1) != 6 or len(box2) != 6:
        return 0.0

    x1, y1, z1, dx1, dy1, dz1 = box1
    x2, y2, z2, dx2, dy2, dz2 = box2
    
    # Min/Max coordinates
    min1 = np.array([x1 - dx1/2, y1 - dy1/2, z1 - dz1/2])
    max1 = np.array([x1 + dx1/2, y1 + dy1/2, z1 + dz1/2])
    min2 = np.array([x2 - dx2/2, y2 - dy2/2, z2 - dz2/2])
    max2 = np.array([x2 + dx2/2, y2 + dy2/2, z2 + dz2/2])
    
    # Intersection
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(inter_max - inter_min, 0)
    intersection_vol = np.prod(inter_dims)
    
    # Union
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union_vol = vol1 + vol2 - intersection_vol
    
    if union_vol <= 0:
        return 0.0
        
    return float(intersection_vol / union_vol)

def calculate_center_offset(box1, box2):
    """计算中心点欧氏距离"""
    if not box1 or not box2:
        return None
    c1 = np.array(box1[:3])
    c2 = np.array(box2[:3])
    return float(np.linalg.norm(c1 - c2))

# ==================== LLM 辅助函数 ====================

def extract_json_from_llm_output(text: str):
    """从 LLM 输出中清洗并解析 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return None

def build_prompt(raw_text):
    """
    构造提取 Prompt。
    针对两种格式进行Few-shot提示：
    1. Float (Meters): bbox_0=Bbox(cat, 0.1, 0.2...)
    2. Int (CM): bbox 0=Bbox, cat, 10, 20...
    """
    return f"""<|im_start|>system
You are a precise data extraction assistant.
Task: Extract all bounding box information from the text.

The text may follow one of two formats:
Format A (Meters): 'bbox_0=Bbox(category, 0.14, -0.48, ...)'
Format B (CM Integers): 'bbox 0=Bbox, category, 11, -48, ...'

Output Format: 
Return ONLY a valid JSON object. 
Keys must be the integer bbox IDs (e.g., "0", "1").
Values must be objects containing:
1. "category": The object category name (string).
2. "params": A list of 6 numbers [x, y, z, dx, dy, dz]. Extract the RAW numbers as they appear in text.

Example Input:
"bbox 0=Bbox, speaker, 11, -48, -108, 31, 87, 31. bbox_1=Bbox(chair, 1.5, 0.5, 0.0, 0.5, 0.5, 1.0)"

Example Output:
{{
  "0": {{ "category": "speaker", "params": [11, -48, -108, 31, 87, 31] }},
  "1": {{ "category": "chair", "params": [1.5, 0.5, 0.0, 0.5, 0.5, 1.0] }}
}}

If no valid data is found, return {{}}.
<|im_end|>
<|im_start|>user
{raw_text}
<|im_end|>
<|im_start|>assistant
"""

# ==================== 主逻辑 ====================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Grounding (CM Prediction vs Meter GT).")
    parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = args.input_json.replace(".json", "_scored_cm.json")

    print(f"Load: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容数据结构
    target_list = []
    is_nested = False
    if isinstance(data, list):
        target_list = data
    elif isinstance(data, dict):
        if "details" in data and "visual_grounding" in data["details"]:
            target_list = data["details"]["visual_grounding"]
            is_nested = True
        else:
            target_list = data.get("visual_grounding", [])

    if not target_list:
        print("No visual_grounding data found.")
        return

    print(f"Total samples: {len(target_list)}")

    # 1. 准备 LLM 任务
    tasks = []
    task_map = {} 
    prompt_list = []
    
    for idx, item in enumerate(target_list):
        gt_text = item.get("text", "")
        pred_text = item.get("pred_text", "")
        
        # 任务 1: GT (通常是 Float/Meters)
        if gt_text:
            task_id = len(prompt_list)
            prompt_list.append(build_prompt(gt_text))
            task_map[task_id] = {"idx": idx, "type": "gt"}
            
        # 任务 2: Pred (通常是 Int/CM)
        if pred_text:
            task_id = len(prompt_list)
            prompt_list.append(build_prompt(pred_text))
            task_map[task_id] = {"idx": idx, "type": "pred"}

    print(f"LLM Extraction Tasks: {len(prompt_list)} (GT + Pred)")

    # 2. 运行 vLLM
    if prompt_list:
        print("Initializing vLLM...")
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            tensor_parallel_size=1,
            disable_custom_all_reduce=True,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

        print("Running inference...")
        outputs = llm.generate(prompt_list, sampling_params)

        # 3. 解析结果并暂存
        parsed_results = {}
        
        success_count = 0
        for i, output in enumerate(outputs):
            meta = task_map[i]
            sample_idx = meta["idx"]
            data_type = meta["type"]
            
            if sample_idx not in parsed_results:
                parsed_results[sample_idx] = {}

            llm_text = output.outputs[0].text.strip()
            parsed_json = extract_json_from_llm_output(llm_text)
            
            if parsed_json is not None:
                clean_data = {}
                for k, v in parsed_json.items():
                    try:
                        bbox_id = int(k)
                        if isinstance(v, dict) and "params" in v:
                            params = v["params"]
                            category = v.get("category", "")
                            
                            if isinstance(params, list) and len(params) >= 6:
                                # 提取原始数值
                                raw_vals = [float(x) for x in params[:6]]
                                
                                # ==============================================
                                # 关键修改：单位换算 (CM -> Meters)
                                # ==============================================
                                if data_type == "pred":
                                    # 假设预测结果是 Int(CM)，需要除以 100 还原为米
                                    final_vals = [x / 100.0 for x in raw_vals]
                                else:
                                    # GT 假设已经是米 (Float)
                                    final_vals = raw_vals
                                # ==============================================

                                clean_data[bbox_id] = {
                                    "category": str(category),
                                    "params": final_vals
                                }
                    except Exception as e:
                        continue
                
                parsed_results[sample_idx][data_type] = clean_data
                success_count += 1
            else:
                parsed_results[sample_idx][data_type] = {}

        print(f"Extraction complete. Successful parses: {success_count}/{len(prompt_list)}")
    else:
        parsed_results = {}

    # 4. 计算指标 Loop
    all_ious = []
    all_offsets = []
    total_objects = 0
    matched_objects = 0

    for idx, item in enumerate(target_list):
        res = parsed_results.get(idx, {})
        gt_data = res.get("gt", {})   
        pred_data = res.get("pred", {}) 
        
        # 将解析后的结果(米)保存回去，方便 debug
        item["parsed_gt_meters"] = gt_data
        item["parsed_pred_meters"] = pred_data
        
        item_metrics = []
        
        for bbox_id, gt_info in gt_data.items():
            total_objects += 1
            
            gt_params = gt_info["params"]
            gt_cat = gt_info["category"]
            
            entry = {
                "bbox_id": bbox_id,
                "category": gt_cat,
                "iou": 0.0,
                "center_offset": None,
                "gt_params": gt_params,
                "pred_params": None,
                "pred_category": None
            }
            
            if bbox_id in pred_data:
                pred_info = pred_data[bbox_id]
                pred_params = pred_info["params"]
                pred_cat = pred_info["category"]
                
                entry["pred_params"] = pred_params
                entry["pred_category"] = pred_cat
                
                # IoU (现在 GT 和 Pred 都是米了)
                iou = calculate_iou_3d(gt_params, pred_params)
                entry["iou"] = iou
                all_ious.append(iou)
                
                # Offset
                offset = calculate_center_offset(gt_params, pred_params)
                entry["center_offset"] = offset
                all_offsets.append(offset)
                
                matched_objects += 1
            else:
                all_ious.append(0.0)
            
            item_metrics.append(entry)
            
        item["object_metrics"] = item_metrics

    # 5. 汇总统计
    summary = {}
    
    if all_ious:
        summary["iou"] = {
            "mean": float(np.mean(all_ious)),
            "median": float(np.median(all_ious)),
            "count": len(all_ious)
        }
    else:
        summary["iou"] = {"mean": 0.0, "median": 0.0, "count": 0}

    if all_offsets:
        summary["center_offset"] = {
            "mean": float(np.mean(all_offsets)),
            "median": float(np.median(all_offsets)),
            "count": len(all_offsets)
        }
    else:
        summary["center_offset"] = {"mean": None, "median": None, "count": 0}

    print("-" * 40)
    print(f"Total Objects (GT):  {total_objects}")
    print(f"Matched Objects:     {matched_objects}")
    print("-" * 40)
    print(f"IoU Mean:            {summary['iou']['mean']:.4f}")
    print(f"IoU Median:          {summary['iou']['median']:.4f}")
    print("-" * 40)
    print(f"Offset Mean (m):     {summary['center_offset']['mean']:.4f}")
    print(f"Offset Median (m):   {summary['center_offset']['median']:.4f}")
    print("-" * 40)

    # 6. 保存结果
    if is_nested:
        if "summary" not in data: data["summary"] = {}
        data["summary"]["visual_grounding_metrics"] = summary
    else:
        output_data = {
            "summary": summary,
            "details": target_list
        }
        data = output_data

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output_json}")

if __name__ == "__main__":
    main()
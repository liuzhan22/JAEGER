import json
import argparse
import numpy as np
import os

def calculate_angular_error_vector(vec1, vec2):
    """
    计算两个 3D 向量之间的夹角 (度数)
    vec: [x, y, z]
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None
        
    dot_product = np.dot(v1, v2)
    cosine_similarity = dot_product / (norm1 * norm2)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_similarity)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)

def main():
    parser = argparse.ArgumentParser(description="Calculate Visual DoA error and save to JSON.")
    parser.add_argument("--input-json", type=str, default="/path/to/results/file.json", help="Path to the scored JSON file.")
    args = parser.parse_args()

    # 构造输出文件名: input.json -> input_DoA.json
    base, ext = os.path.splitext(args.input_json)
    output_json_path = f"{base}_DoA{ext}"

    print(f"Loading data from: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容数据结构提取 target_list
    target_list = []
    is_nested = False
    
    if isinstance(data, list):
        target_list = data
    elif isinstance(data, dict):
        if "details" in data:
            if isinstance(data["details"], list):
                 target_list = data["details"]
            elif "visual_grounding" in data["details"]:
                target_list = data["details"]["visual_grounding"]
                is_nested = True
        elif "visual_grounding" in data:
            target_list = data["visual_grounding"]

    if not target_list:
        print("Error: No sample data found.")
        return

    angular_errors = []
    total_objects = 0
    matched_objects = 0

    print(f"Processing {len(target_list)} samples...")

    # --- 核心循环 ---
    for item in target_list:
        if "object_metrics" not in item:
            continue

        for obj in item["object_metrics"]:
            total_objects += 1
            
            gt_params = obj.get("gt_params")
            pred_params = obj.get("pred_params")
            
            # 初始化 angular_error 为 None
            obj["angular_error"] = None

            if gt_params and pred_params:
                # 提取前三位 [x, y, z]
                gt_center = gt_params[:3]
                pred_center = pred_params[:3]
                
                error = calculate_angular_error_vector(gt_center, pred_center)
                
                if error is not None:
                    # === 写入误差到每个 Object 中 ===
                    obj["angular_error"] = error
                    
                    angular_errors.append(error)
                    matched_objects += 1

    # --- 统计 Summary ---
    summary_stats = {}
    if angular_errors:
        summary_stats = {
            "mean": float(np.mean(angular_errors)),
            "median": float(np.median(angular_errors)),
            "std": float(np.std(angular_errors)),
            "min": float(np.min(angular_errors)),
            "max": float(np.max(angular_errors)),
            "count": len(angular_errors)
        }
        
        print("-" * 40)
        print(f"Total Objects: {total_objects}")
        print(f"Matched Objects: {matched_objects}")
        print(f"Visual DoA Mean:   {summary_stats['mean']:.2f}°")
        print(f"Visual DoA Median: {summary_stats['median']:.2f}°")
        print("-" * 40)
    else:
        print("No valid angular errors calculated.")

    # === 更新 Summary 到 Data 字典 ===
    # 如果原始数据是 dict 结构，我们将新的 DoA 统计加进去
    if isinstance(data, dict):
        if "summary" not in data:
            data["summary"] = {}
        
        # 保存到 summary['visual_doa']
        data["summary"]["visual_doa"] = summary_stats

    # === 保存文件 ===
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved updated results with DoA metrics to: {output_json_path}")

if __name__ == "__main__":
    main()
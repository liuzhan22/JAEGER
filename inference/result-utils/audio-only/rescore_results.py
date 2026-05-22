import argparse
import json
import math
import re
import os
import numpy as np

# ==================== 1. 改进后的解析函数 ====================

def parse_doa_from_text(text: str):
    """
    使用正则稳健地提取 Azimuth 和 Elevation。
    能够处理:
    - "azimuth: 140.75; elevation: -57.25." (带句号)
    - "Azimuth 140.75, Elevation -57.25"
    - "Elev: -57.25 Az: 140.75"
    """
    if not text:
        return None, None

    # 匹配模式：查找 azimuth 或 elevation 后面的浮点数
    # (?i) 忽略大小写
    # [:\s]+ 匹配冒号或空格
    # ([-+]?\d*\.?\d+) 捕获数字（支持负号和小数点）
    az_match = re.search(r"(?i)azimuth[:\s]+([-+]?\d*\.?\d+)", text)
    el_match = re.search(r"(?i)elevation[:\s]+([-+]?\d*\.?\d+)", text)

    az = float(az_match.group(1)) if az_match else None
    el = float(el_match.group(1)) if el_match else None
    
    return az, el

def parse_distance_from_text(text: str):
    """
    解析距离，支持 "distance: 3.52m", "3.52 meters" 等
    """
    if not text:
        return None
    # 尝试正则提取 'distance' 关键词后的数字，或者直接提取第一个出现的浮点数
    dist_match = re.search(r"(?i)distance[:\s]+([-+]?\d*\.?\d+)", text)
    if dist_match:
        return float(dist_match.group(1))
    
    # 如果没有关键词，尝试提取第一个带 'm' 单位的数字
    m_match = re.search(r"([-+]?\d*\.?\d+)\s*m", text)
    if m_match:
        return float(m_match.group(1))

    # 最后的手段：简单的分割解析 (保留原有逻辑作为兜底)
    text_clean = text.replace("m", " ").replace(":", " ")
    parts = text_clean.split()
    for p in parts:
        try:
            return float(p)
        except ValueError:
            continue
    return None

def parse_pred_label(text: str, candidates):
    """AED 任务：选择匹配度最高的候选标签"""
    if not text:
        return None
    text_low = text.lower()
    best = None
    best_score = -1
    for c in candidates:
        c_low = c.lower()
        if c_low in text_low:
            score = len(c_low)
        else:
            score = 0
        if score > best_score and score > 0: # 必须有匹配
            best_score = score
            best = c
    
    # 如果没有匹配到任何候选词（best_score=0），返回 None 或 raw text
    return best if best else None 

# ==================== 2. 指标计算函数 ====================

def angular_error(az1, el1, az2, el2):
    """计算角度误差 (Great Circle Distance)"""
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
    
    #防止数值误差导致的 domain error
    dot = x1 * x2 + y1 * y2 + z1 * z2
    dot = max(min(dot, 1.0), -1.0)
    
    angle = math.degrees(math.acos(dot))
    return angle

# ==================== 3. 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="Re-score evaluation results using robust parsing.")
    parser.add_argument("--input-json", type=str, required=True, help="Path to the existing eval_results.json file.")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save the re-scored json. Defaults to input_rescored.json")
    args = parser.parse_args()

    input_path = args.input_json
    if args.output_json:
        output_path = args.output_json
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_rescored{ext}"

    print(f"Reading results from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取 details 部分
    details = data.get("details", {})
    if not details:
        print("❌ Error: 'details' key not found in JSON.")
        return

    # 准备重新计算的统计容器
    metrics = {
        "aed": {"correct": 0, "total": 0},
        "doa": {"errors": []},
        "dist": {"errors": []}
    }

    print("Re-parsing and calculating scores...")

    # --- 1. Audio Event Detection (AED) ---
    # 如果需要重新匹配标签，我们需要构建候选列表。
    # 这里我们遍历所有数据来收集所有可能的 GT label 作为 candidates
    aed_list = details.get("audio_event_detection", [])
    if aed_list:
        # 收集所有出现的 GT label
        all_gt_labels = sorted({item["gt_label"] for item in aed_list if "gt_label" in item})
        
        for item in aed_list:
            raw = item.get("raw_output", "")
            gt = item.get("gt_label")
            
            # 重新解析
            pred = parse_pred_label(raw, all_gt_labels)
            
            # 更新 Item
            item["pred_label"] = pred
            
            # 统计
            metrics["aed"]["total"] += 1
            if pred == gt:
                metrics["aed"]["correct"] += 1
                item["is_correct"] = True
            else:
                item["is_correct"] = False

    # --- 2. Spatial DOA ---
    doa_list = details.get("spatial_doa", [])
    for item in doa_list:
        raw = item.get("raw_output", "")
        gt_az = item.get("gt_az")
        gt_el = item.get("gt_el")

        # 重新解析 (使用正则)
        pred_az, pred_el = parse_doa_from_text(raw)
        
        # 计算误差
        err = angular_error(gt_az, gt_el, pred_az, pred_el)

        # 更新 Item
        item["pred_az"] = pred_az
        item["pred_el"] = pred_el
        item["angular_error"] = err
        
        if err is not None:
            metrics["doa"]["errors"].append(err)

    # --- 3. Distance Estimation ---
    dist_list = details.get("distance_estimation", [])
    for item in dist_list:
        raw = item.get("raw_output", "")
        gt_dist = item.get("gt_dist")

        # 重新解析
        pred_dist = parse_distance_from_text(raw)

        # 计算误差
        err = None
        if pred_dist is not None and gt_dist is not None:
            err = abs(pred_dist - gt_dist)
            metrics["dist"]["errors"].append(err)
        
        # 更新 Item
        item["pred_dist"] = pred_dist
        item["abs_error"] = err

    # ==================== 4. 生成新 Summary ====================
    new_summary = {}

    # AED Summary
    if metrics["aed"]["total"] > 0:
        acc = metrics["aed"]["correct"] / metrics["aed"]["total"]
        new_summary["audio_event_detection"] = {
            "accuracy": acc,
            "num_samples": metrics["aed"]["total"]
        }
        print(f"[AED] New Accuracy: {acc:.2%} ({metrics['aed']['correct']}/{metrics['aed']['total']})")

    # DOA Summary
    if metrics["doa"]["errors"]:
        mean_err = sum(metrics["doa"]["errors"]) / len(metrics["doa"]["errors"])
        new_summary["spatial_doa"] = {
            "mean_angular_error_deg": mean_err,
            "num_samples": len(metrics["doa"]["errors"])
        }
        print(f"[DOA] New Mean Angular Error: {mean_err:.2f}° (Samples: {len(metrics['doa']['errors'])})")
    else:
        print("[DOA] No valid predictions found.")

    # Distance Summary
    if metrics["dist"]["errors"]:
        mean_dist_err = sum(metrics["dist"]["errors"]) / len(metrics["dist"]["errors"])
        new_summary["distance_estimation"] = {
            "mean_abs_error_m": mean_dist_err,
            "num_samples": len(metrics["dist"]["errors"])
        }
        print(f"[Dist] New Mean Abs Error: {mean_dist_err:.3f}m (Samples: {len(metrics['dist']['errors'])})")

    # ==================== 5. 保存结果 ====================
    
    # 构造输出对象，保留原始 config，更新 summary 和 details
    output_obj = {
        "config": data.get("config", {}),
        "summary": new_summary,
        "details": details 
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Re-scored results saved to: {output_path}")

if __name__ == "__main__":
    main()
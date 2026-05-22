import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
import json
import argparse
import math
import re
import numpy as np  # 新增 numpy 用于计算中位数
from vllm import LLM, SamplingParams

# ==================== 配置 ====================
MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct"
DEFAULT_INPUT_FILE = "/path/to/results/file.json"

# ==================== 工具函数 ====================

def angular_error(az1, el1, az2, el2):
    """计算角度误差 (Great Circle Distance)"""
    if None in (az1, el1, az2, el2):
        return None
    try:
        az1_r, el1_r = math.radians(az1), math.radians(el1)
        az2_r, el2_r = math.radians(az2), math.radians(el2)
        
        x1 = math.cos(el1_r) * math.cos(az1_r)
        y1 = math.cos(el1_r) * math.sin(az1_r)
        z1 = math.sin(el1_r)
        
        x2 = math.cos(el2_r) * math.cos(az2_r)
        y2 = math.cos(el2_r) * math.sin(az2_r)
        z2 = math.sin(el2_r)
        
        dot = x1 * x2 + y1 * y2 + z1 * z2
        dot = max(min(dot, 1.0), -1.0)
        return math.degrees(math.acos(dot))
    except:
        return None

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
    """构造提取 Prompt"""
    return f"""<|im_start|>system
You are a precise data extraction assistant.
Task: Extract the 'azimuth' and 'elevation' numerical values from the user's text.
Output Format: Return ONLY a valid JSON object strictly in this format: {{"az": <float>, "el": <float>}}. 
If a value is missing, use null. Do not output any explanation.
<|im_end|>
<|im_start|>user
{raw_text}
<|im_end|>
<|im_start|>assistant
"""

# ==================== 主逻辑 ====================

def main():
    parser = argparse.ArgumentParser(description="Use LLM to re-extract DOA labels.")
    parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--force-all", action="store_true", help="If set, re-extract ALL items, not just failed ones.")
    args = parser.parse_args()

    output_path = args.input_json.replace(".json", "_rescored_llm.json")
    
    print(f"Load: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    doa_list = data.get("details", {}).get("spatial_doa", [])
    if not doa_list:
        print("No spatial_doa details found.")
        return

    # 1. 筛选需要 LLM 处理的样本
    tasks = []
    for idx, item in enumerate(doa_list):
        if (args.force_all or item.get("pred_az") is None) and item.get("raw_output"):
            tasks.append({
                "index": idx,
                "prompt": build_prompt(item["raw_output"])
            })

    print(f"Total items: {len(doa_list)}")
    print(f"Items requiring LLM extraction: {len(tasks)}")

    # 2. 运行 vLLM (如果有任务)
    if tasks:
        print("Initializing vLLM...")
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            tensor_parallel_size=1,
            disable_custom_all_reduce=True
        )
        sampling_params = SamplingParams(temperature=0.1, max_tokens=64)

        prompts = [t["prompt"] for t in tasks]
        print("Running inference...")
        outputs = llm.generate(prompts, sampling_params)

        # 3. 解析结果并回填
        success_count = 0
        for i, output in enumerate(outputs):
            idx = tasks[i]["index"]
            llm_text = output.outputs[0].text.strip()
            
            parsed = extract_json_from_llm_output(llm_text)
            
            if parsed and isinstance(parsed, dict):
                az = parsed.get("az") or parsed.get("azimuth")
                el = parsed.get("el") or parsed.get("elevation")
                try:
                    if az is not None: az = float(az)
                    if el is not None: el = float(el)
                    
                    doa_list[idx]["pred_az"] = az
                    doa_list[idx]["pred_el"] = el
                    success_count += 1
                except ValueError:
                    print(f"Warning: Failed to convert to float: {llm_text}")
            else:
                pass # 解析失败

        print(f"LLM Extraction Complete. Success: {success_count}/{len(tasks)}")

    # 4. 重新计算所有 Metric (Angular Error)
    errors = []
    for item in doa_list:
        gt_az, gt_el = item.get("gt_az"), item.get("gt_el")
        pred_az, pred_el = item.get("pred_az"), item.get("pred_el")
        
        err = angular_error(gt_az, gt_el, pred_az, pred_el)
        item["angular_error"] = err
        
        if err is not None:
            errors.append(err)

    # 5. 更新 Summary (双指标：Mean 和 Median)
    if errors:
        # 计算平均数 (Mean)
        mean_err = sum(errors) / len(errors)
        # 计算中位数 (Median)
        median_err = float(np.median(errors)) # 转为 python float

        # 保持原有结构，更新 summary
        if "summary" not in data: data["summary"] = {}
        data["summary"]["spatial_doa"] = {
            "mean_angular_error_deg": mean_err,
            "median_angular_error_deg": median_err,
            "num_samples": len(errors)
        }
        data["details"]["spatial_doa"] = doa_list

        print(f"Final Mean   Angular Error: {mean_err:.2f}°")
        print(f"Final Median Angular Error: {median_err:.2f}° (Valid Samples: {len(errors)})")
    else:
        print("No valid spatial_doa errors found.")
        data["summary"]["spatial_doa"] = {}

    # 6. 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
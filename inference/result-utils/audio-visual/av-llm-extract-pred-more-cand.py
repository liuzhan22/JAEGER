import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import json
import argparse
import re
from vllm import LLM, SamplingParams

# Configuration
MODEL_PATH = "/path/to/Qwen2.5-7B-Instruct" 
DEFAULT_INPUT_FILE = "/path/to/results/file.json"
TARGET_TASKS = ["single_source_more_cand", "dual_source_more_cand"]

def build_extraction_prompt(pred_text):
    return f"""<|im_start|>system
You are a text classification assistant.
Task: Identify which speaker corresponds to the audio source from the text.

Rules:
1. Extract the speaker position from the text (e.g., "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th")
2. If the text mentions a specific speaker position, output ONLY the ordinal number (e.g., "1st", "2nd", etc.)
3. If the text is ambiguous or mentions no specific position, output: "Unknown"

Output ONLY the ordinal number or "Unknown".
<|im_end|>
<|im_start|>user
{pred_text}
<|im_end|>
<|im_start|>assistant
"""

def clean_llm_output(text):
    if not text: return "Unknown"
    text = text.strip().rstrip(".").lower()
    
    # Extract ordinal numbers using regex
    ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    match = re.search(ordinal_pattern, text)
    
    if match:
        number = match.group(1)
        suffix = match.group(2)
        # Convert to standard ordinal format
        if suffix == "st":
            return f"{number}st"
        elif suffix == "nd":
            return f"{number}nd"
        elif suffix == "rd":
            return f"{number}rd"
        elif suffix == "th":
            return f"{number}th"
    
    # Check for spelled-out ordinals
    ordinal_words = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th"
    }
    
    for word, ordinal in ordinal_words.items():
        if word in text:
            return ordinal
    
    return "Unknown"

def normalize_gt(text):
    if not text: return "Unknown"
    text = text.strip().lower()
    
    # Extract ordinal numbers from ground truth
    ordinal_pattern = r'\b(\d+)(st|nd|rd|th)\b'
    match = re.search(ordinal_pattern, text)
    
    if match:
        number = match.group(1)
        suffix = match.group(2)
        return f"{number}{suffix}"
    
    # Check for spelled-out ordinals
    ordinal_words = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th"
    }
    
    for word, ordinal in ordinal_words.items():
        if word in text:
            return ordinal
    
    return "Unknown"

def process_task_data(task_name, target_list, llm, sampling_params):
    prompts = []
    indices = []

    for idx, item in enumerate(target_list):
        pred_text = item.get("pred_text", "")
        if not pred_text:
            continue
        prompts.append(build_extraction_prompt(pred_text))
        indices.append(idx)

    # Run vLLM
    if prompts and llm:
        outputs = llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            idx = indices[i]
            raw_out = output.outputs[0].text
            target_list[idx]["llm_extracted"] = clean_llm_output(raw_out)
            target_list[idx]["llm_raw_output"] = raw_out

    # Compute metrics
    correct_count = 0
    total_count = len(target_list)
    
    for item in target_list:
        gt_label = normalize_gt(item.get("gt_text", ""))
        pred_label = item.get("llm_extracted", "Unknown")
        
        if pred_label == "Unknown" and item.get("pred_text"):
             pred_label = clean_llm_output(item.get("pred_text"))

        item["final_pred_label"] = pred_label
        item["final_gt_label"] = gt_label
        
        is_correct = (pred_label == gt_label) and (gt_label != "Unknown")
        item["is_correct"] = is_correct
        if is_correct: 
            correct_count += 1

    acc = (correct_count / total_count * 100) if total_count > 0 else 0.0
    return {
        "total_samples": total_count,
        "correct_samples": correct_count,
        "accuracy": f"{acc:.2f}%"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    if args.output_json is None:
        args.output_json = args.input_json.replace(".json", "_llm_scored.json")

    print(f"Load: {args.input_json}")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get("details", {})
    if not details:
        print("Error: 'details' key not found in JSON.")
        return

    # Initialize vLLM only if target tasks have data
    llm = None
    sampling_params = None
    has_data = any(t in details and len(details[t]) > 0 for t in TARGET_TASKS)
    
    if has_data:
        print("Initializing vLLM...")
        llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            tensor_parallel_size=1,
            disable_custom_all_reduce=True,
        )
        sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    if "summary" not in data:
        data["summary"] = {}

    for task_name in TARGET_TASKS:
        if task_name in details:
            target_list = details[task_name]
            if not target_list: 
                continue
            
            print(f"Processing {task_name}...")
            metrics = process_task_data(task_name, target_list, llm, sampling_params)
            data["summary"][f"{task_name}_llm_metrics"] = metrics
            
            print("-" * 40)
            print(f"Task: {task_name}")
            print(f"Total: {metrics['total_samples']}")
            print(f"Correct: {metrics['correct_samples']}")
            print(f"Accuracy: {metrics['accuracy']}")
            print("-" * 40)

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {args.output_json}")

if __name__ == "__main__":
    main()
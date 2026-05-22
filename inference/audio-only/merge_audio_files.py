import os
import json
import glob
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/path/to/results/dir", help="Directory containing rank_*.json files")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output filename")
    parser.add_argument("--delete-shards", action="store_true", help="Delete original shard files after merging")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Find all shard files
    search_pattern = os.path.join(args.dir, "eval_results_rank_*.json")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"[Error] No shard files found in {args.dir}")
        return

    print(f"Found {len(files)} shard files. Merging...")

    # 2. Containers for merging (Added vgg tasks)
    merged_details = {
        "audio_event_detection": [],
        "spatial_doa": [],
        "spatial_doa_vgg": [],        
        "multi_spatial_doa": [],
        "multi_spatial_doa_vgg": [],  
        "distance_estimation": [],
        "starss23": [],
    }
    
    config_backup = None
    
    # 3. Merge Loop
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if config_backup is None:
                    config_backup = data.get("config", {})
                
                details = data.get("details", {})
                for key in merged_details:
                    if key in details:
                        merged_details[key].extend(details[key])
        except Exception as e:
            print(f"[Warning] Error reading {f_path}: {e}")

    # 4. Re-calculate Global Metrics
    summary = {}
    print("-" * 40)
    
    # --- AED Metrics ---
    aed_list = merged_details["audio_event_detection"]
    if aed_list:
        total = len(aed_list)
        correct = sum(1 for x in aed_list if x["pred_label"] == x["gt_label"])
        acc = correct / total
        summary["audio_event_detection"] = {"accuracy": acc, "count": total}
        print(f"[AED] Accuracy: {acc:.2%} ({correct}/{total})")

    # --- DOA Metrics ---
    doa_tasks = ["spatial_doa", "spatial_doa_vgg", "multi_spatial_doa", "multi_spatial_doa_vgg", "starss23"]
    for task_key in doa_tasks:
        doa_list = merged_details.get(task_key, [])
        if doa_list:
            errors = [x["angular_error"] for x in doa_list if x["angular_error"] is not None]
            if errors:
                mean_err = float(np.mean(errors))
                median_err = float(np.median(errors)) 
                
                summary[task_key] = {
                    "mean_angular_error_deg": mean_err,
                    "median_angular_error_deg": median_err, 
                    "count": len(errors)
                }
                print(f"[{task_key}] Mean: {mean_err:.2f}° | Median: {median_err:.2f}°")

    # --- Distance Metrics ---
    dist_list = merged_details["distance_estimation"]
    if dist_list:
        errors = [x["abs_error"] for x in dist_list if x["abs_error"] is not None]
        if errors:
            mean_err = float(np.mean(errors))
            summary["distance_estimation"] = {"mean_abs_error_m": mean_err, "count": len(errors)}
            print(f"[Dist] Mean Abs Error: {mean_err:.3f}m")

    print("-" * 40)

    # 5. Save Final Result
    final_output = {
        "config": config_backup,
        "summary": summary,
        "details": merged_details
    }
    
    out_path = os.path.join(args.dir, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"[Success] Merged results saved to: {out_path}")

    # 6. Cleanup
    if args.delete_shards:
        print("Cleaning up shard files...")
        for f_path in files:
            os.remove(f_path)
        print("Done.")

if __name__ == "__main__":
    main()
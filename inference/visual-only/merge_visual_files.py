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
    # 假设 visual 的分片文件命名也是 eval_results_rank_*.json
    search_pattern = os.path.join(args.dir, "visual_results_rank_*.json")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"No shard files found in {args.dir}")
        return

    print(f"Found {len(files)} shard files. Merging...")

    # 2. Containers for merging
    # 针对 Visual Grounding 任务的容器
    merged_details = {
        "visual_grounding": [],
    }
    
    config_backup = None
    
    # 3. Merge Loop
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    merged_details["visual_grounding"].extend(data)
                else:
                    print(f"⚠️ Unexpected data format in {f_path}, expected a list.")
                    
        except Exception as e:
            print(f"⚠️ Error reading {f_path}: {e}")

    # 4. Re-calculate Global Metrics
    summary = {}
    print("-" * 40)
    
    # --- Visual Metrics ---
    vis_list = merged_details["visual_grounding"]
    if vis_list:
        total = len(vis_list)
        summary["visual_grounding"] = {"count": total}
        
        print(f"[Visual] Total Samples: {total}")

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
    
    print(f"✅ Merged results saved to: {out_path}")

    # 6. Cleanup (Optional)
    if args.delete_shards:
        print("Cleaning up shard files...")
        for f_path in files:
            os.remove(f_path)
        print("Done.")

if __name__ == "__main__":
    main()
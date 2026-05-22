import os
import json
import glob
import argparse
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/path/to/results/dir", help="Directory containing av_results_rank_*.json files")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output filename")
    parser.add_argument("--delete-shards", action="store_true", help="Delete original shard files after merging")
    return parser.parse_args()

def calculate_accuracy(items):
    """
    Calculate accuracy.
    Logic: pred_text (normalized) == gt_text (normalized)
    """
    if not items:
        return 0.0, 0, 0

    correct = 0
    total = len(items)

    for item in items:
        pred = item.get("pred_text", "").strip().lower()
        gt = item.get("gt_text", "").strip().lower()
        
        pred = pred.rstrip(".")
        gt = gt.rstrip(".")

        if pred == gt:
            correct += 1
    
    return (correct / total) * 100, correct, total

def main():
    args = get_args()
    
    search_pattern = os.path.join(args.dir, "av_results_rank_*.json")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print(f"[Error] No shard files found in {args.dir}")
        return

    print(f"Found {len(files)} shard files. Merging...")

    merged_data = defaultdict(list)
    
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        task = item.get("task", "unknown")
                        merged_data[task].append(item)
                else:
                    print(f"[Warning] Unexpected data format in {f_path}, expected a list.")
                    
        except Exception as e:
            print(f"[Error] Error reading {f_path}: {e}")

    summary = {}
    print("-" * 60)
    print(f"{'Task':<25} | {'Acc (%)':<10} | {'Correct/Total':<15}")
    print("-" * 60)

    # --- Task 1: Single Source ---
    if "single_source" in merged_data:
        items = merged_data["single_source"]
        acc, corr, tot = calculate_accuracy(items)
        summary["single_source"] = {
            "accuracy": f"{acc:.2f}%",
            "correct": corr,
            "total": tot
        }
        print(f"{'Single Source':<25} | {acc:<10.2f} | {corr}/{tot}")

    # --- Task 2: Dual Source ---
    if "dual_source" in merged_data:
        items = merged_data["dual_source"]
        
        acc, corr, tot = calculate_accuracy(items)
        
        male_items = [x for x in items if x.get("gender") in ["male", "M"]]
        female_items = [x for x in items if x.get("gender") in ["female", "F"]]
        
        m_acc, m_corr, m_tot = calculate_accuracy(male_items)
        f_acc, f_corr, f_tot = calculate_accuracy(female_items)

        summary["dual_source"] = {
            "overall": {
                "accuracy": f"{acc:.2f}%",
                "correct": corr,
                "total": tot
            },
            "breakdown": {
                "male": {"accuracy": f"{m_acc:.2f}%", "correct": m_corr, "total": m_tot},
                "female": {"accuracy": f"{f_acc:.2f}%", "correct": f_corr, "total": f_tot}
            }
        }
        print(f"{'Dual Source (Overall)':<25} | {acc:<10.2f} | {corr}/{tot}")
        print(f"{'  - Male Target':<25} | {m_acc:<10.2f} | {m_corr}/{m_tot}")
        print(f"{'  - Female Target':<25} | {f_acc:<10.2f} | {f_corr}/{f_tot}")

    # --- Task 2: Dual Source VGG ---
    if "dual_source_vgg" in merged_data:
        items = merged_data["dual_source_vgg"]
        
        acc, corr, tot = calculate_accuracy(items)
        
        summary["dual_source_vgg"] = {
            "overall": {
                "accuracy": f"{acc:.2f}%",
                "correct": corr,
                "total": tot
            }
        }
        print(f"{'Dual Source VGG':<25} | {acc:<10.2f} | {corr}/{tot}")

    print("-" * 60)

    final_output = {
        "summary": summary,
        "details": merged_data
    }
    
    out_path = os.path.join(args.dir, args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"[Success] Merged results saved to: {out_path}")

    if args.delete_shards:
        print("Cleaning up shard files...")
        for f_path in files:
            os.remove(f_path)
        print("Done.")

if __name__ == "__main__":
    main()
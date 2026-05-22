import json
import numpy as np
import os

def process_evaluation(input_file, output_file):
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get("details", {}).get("spatial_doa", [])
    
    az_errors = []
    el_errors = []
    
    for item in items:
        # 1. Azimuth Error (circular)
        if item["gt_az"] is None or item["pred_az"] is None or item["gt_el"] is None or item["pred_el"] is None:
            continue
        gt_az = item["gt_az"]
        pred_az = item["pred_az"]
        
        diff_az = abs(pred_az - gt_az) % 360
        if diff_az > 180:
            diff_az = 360 - diff_az
        az_errors.append(diff_az)
        
        # 2. Elevation Error
        gt_el = item["gt_el"]
        pred_el = item["pred_el"]
        
        diff_el = abs(pred_el - gt_el)
        el_errors.append(diff_el)

    # Convert to numpy for stats
    az_errors = np.array(az_errors)
    el_errors = np.array(el_errors)

    # Calculate stats
    stats = {
        "count": len(items),
        "az_mae": float(np.mean(az_errors)),
        "az_median": float(np.median(az_errors)),
        "el_mae": float(np.mean(el_errors)),
        "el_median": float(np.median(el_errors))
    }

    # Print for verification
    print(f"Processed {len(items)} items.")
    print("Stats:", stats)

    # Update data with new stats
    # Ensure 'summary' key exists or update it
    if "summary" not in data:
        data["summary"] = {}
    
    # You can merge into existing summary or add a specific sub-key
    data["summary"].update(stats)

    # Save to output path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Saved results to: {output_file}")

if __name__ == "__main__":
    input_path = "/path/to/results/file.json"
    output_path = "/path/to/results/file_az_el.json"
    
    process_evaluation(input_path, output_path)
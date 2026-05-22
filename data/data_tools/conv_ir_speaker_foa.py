import os
import json
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import math

# ================= Configuration =================
CONFIG = {
    "hm3d_root": "/path/to/data/simulation_ds/hm3d/hm3d_foa_av_v2",
    # 定义 split 对应的 LibriSpeech 标注文件
    "splits": {
        "train": "/path/to/data/LibriSpeech/train_clean_100_ann_librispeech.json", 
        "val": "/path/to/data/LibriSpeech/dev_clean_ann_librispeech.json",
        "test": "/path/to/data/LibriSpeech/test_clean_ann_librispeech.json"
    },
    "num_workers": max(1, cpu_count() - 8), # 留一些核给系统
    "output_filename": "foa_ls.wav"
}
# =================================================

def load_librispeech_data(json_path):
    """
    加载 LibriSpeech 数据，并按性别分类
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_data = data['annotation']
    male_data = [item for item in all_data if item.get('gender') == 'M']
    female_data = [item for item in all_data if item.get('gender') == 'F']
    
    print(f"Loaded LibriSpeech: Total {len(all_data)} | Male {len(male_data)} | Female {len(female_data)}")
    return {
        "all": all_data,
        "M": male_data,
        "F": female_data
    }

def find_task_dirs(split_root):
    """
    寻找该 split 下所有的 task 文件夹
    """
    # 结构: split_root / scene_id / task_id
    # task_id 可能是 task1_xxx 或 task2_xxx
    task_dirs = glob(os.path.join(split_root, "*", "task*"))
    task_dirs.sort() # 保证顺序一致
    return task_dirs

def convolve_audio(audio_path, ir_path):
    """
    读取单通道音频和多通道 IR，进行卷积
    返回: (samples, channels) 的 numpy array
    """
    # 1. Load Audio (Mono)
    vocal, sr = sf.read(audio_path)
    if len(vocal.shape) > 1:
        vocal = vocal.mean(axis=1) # Ensure mono
        
    # 2. Load IR (4 channels)
    ir = np.load(ir_path)
    
    # Ensure IR shape is (4, N) for iteration, or handle accordingly
    # 假设 ir shape 是 (4, N) 或者 (N, 4)，我们需要 (4, N)
    if ir.shape[0] != 4 and ir.shape[1] == 4:
        ir = ir.T
    elif ir.shape[0] != 4:
        raise ValueError(f"IR shape error: {ir.shape}")

    # 3. Convolution
    # Result shape: (N_samples, 4)
    convolved = np.array([fftconvolve(vocal, ir_channel, mode='full') for ir_channel in ir]).T
    return convolved, sr

def process_single_task(args):
    """
    Worker function
    args: (task_dir, task_type, audio_items)
    audio_items: 
       - For Task1: [audio_item]
       - For Task2: [male_item, female_item]
    """
    task_dir, task_type, audio_items = args
    
    output_path = os.path.join(task_dir, CONFIG["output_filename"])
    meta_path = os.path.join(task_dir, "metadata.json")
    
    try:
        if task_type == 1:
            # === Task 1: Single Source (Random M/F) ===
            item = audio_items[0]
            ir_path = os.path.join(task_dir, "ir.npy")
            
            if not os.path.exists(ir_path):
                return f"[Error] Missing ir.npy in {task_dir}"
                
            final_audio, sr = convolve_audio(item['path'], ir_path)
            
            # Prepare metadata update
            meta_update = {
                "audio_source_info": {
                    "source_path": item['path'],
                    "text": item['text'],
                    "gender": item['gender']
                }
            }

        elif task_type == 2:
            # === Task 2: Dual Source (Male + Female) ===
            male_item = audio_items[0]
            female_item = audio_items[1]
            
            ir_male_path = os.path.join(task_dir, "ir_male.npy")
            ir_female_path = os.path.join(task_dir, "ir_female.npy")
            
            if not os.path.exists(ir_male_path) or not os.path.exists(ir_female_path):
                return f"[Error] Missing ir_male/female.npy in {task_dir}"
            
            # Convolve separately
            conv_m, sr_m = convolve_audio(male_item['path'], ir_male_path)
            conv_f, sr_f = convolve_audio(female_item['path'], ir_female_path)
            
            if sr_m != sr_f:
                return f"[Error] Sample rate mismatch in {task_dir}"
            sr = sr_m
            
            # Pad and Mix
            len_m = conv_m.shape[0]
            len_f = conv_f.shape[0]
            
            if len_m < len_f:
                pad_width = ((0, len_f - len_m), (0, 0)) # Pad time axis, keep channel axis
                conv_m = np.pad(conv_m, pad_width, mode='constant')
            elif len_m > len_f:
                pad_width = ((0, len_m - len_f), (0, 0))
                conv_f = np.pad(conv_f, pad_width, mode='constant')
            
            final_audio = conv_m + conv_f
            
            # Prepare metadata update
            meta_update = {
                "audio_source_info": {
                    "male": {
                        "source_path": male_item['path'],
                        "text": male_item['text'],
                        "gender": "M"
                    },
                    "female": {
                        "source_path": female_item['path'],
                        "text": female_item['text'],
                        "gender": "F"
                    }
                }
            }
        
        else:
            return f"[Error] Unknown task type for {task_dir}"

        # 4. Save Audio
        sf.write(output_path, final_audio, sr)
        
        # 5. Update Metadata (Load -> Update -> Save)
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            meta.update(meta_update)
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
        
        return None # Success

    except Exception as e:
        return f"[Exception] Processing {task_dir}: {str(e)}"

def main():
    for split_name, json_path in CONFIG["splits"].items():
        print(f"\n{'='*20} Processing Split: {split_name} {'='*20}")
        
        hm3d_split_root = os.path.join(CONFIG["hm3d_root"], split_name)
        if not os.path.exists(hm3d_split_root):
            print(f"[Warning] Directory {hm3d_split_root} not found. Skipping.")
            continue
            
        # 1. Load Audio Lists
        audio_db = load_librispeech_data(json_path)
        
        # 2. Find Tasks
        print("Scanning task directories...")
        task_dirs = find_task_dirs(hm3d_split_root)
        print(f"Found {len(task_dirs)} task directories.")
        
        # 3. Assign Jobs
        jobs = []
        
        # Counters for sequential usage
        idx_all = 0
        idx_m = 0
        idx_f = 0
        
        len_all = len(audio_db['all'])
        len_m = len(audio_db['M'])
        len_f = len(audio_db['F'])
        
        if len_all == 0:
            print("[Error] No audio data found. Skipping split.")
            continue

        for d in task_dirs:
            folder_name = os.path.basename(d)
            
            if folder_name.startswith("task1"):
                # --- Task 1 ---
                # Select next audio from 'all' list
                audio_item = audio_db['all'][idx_all % len_all]
                idx_all += 1
                
                jobs.append((d, 1, [audio_item]))
                
            elif folder_name.startswith("task2"):
                # --- Task 2 ---
                if len_m == 0 or len_f == 0:
                    print("[Error] Task 2 requires both Male and Female audio. Skipping.")
                    continue
                    
                # Select next Male
                audio_m = audio_db['M'][idx_m % len_m]
                idx_m += 1
                
                # Select next Female
                audio_f = audio_db['F'][idx_f % len_f]
                idx_f += 1
                
                jobs.append((d, 2, [audio_m, audio_f]))
        
        print(f"Prepared {len(jobs)} jobs. Starting multiprocessing...")
        
        # 4. Execute
        with Pool(CONFIG['num_workers']) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_task, jobs), total=len(jobs)))
            
        # 5. Report
        errors = [r for r in results if r is not None]
        if errors:
            print(f"Finished with {len(errors)} errors.")
            err_log_path = f"error_log_{split_name}.txt"
            with open(err_log_path, "w") as f:
                for err in errors:
                    f.write(err + "\n")
            print(f"Errors saved to {err_log_path}")
        else:
            print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
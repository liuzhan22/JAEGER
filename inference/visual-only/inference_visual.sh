#!/bin/bash

# Configuration
# 修改为您的视觉推理 Python 脚本路径
PY_SCRIPT="/path/to/inference/visual-only/inference_batch_visual.py"
# 配置文件保持不变
CFG_PATH="/path/to/configs/decode_config.yaml"
# 修改为视觉任务的输出目录
OUTPUT_DIR="/path/to/results/dir"
WORLD_SIZE=8

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ">>> Starting parallel VISUAL inference on $WORLD_SIZE GPUs..."

# 1. Launch Loop
for ((i=0; i<WORLD_SIZE; i++)); do
    echo "Launching Rank $i on GPU $i..."
    
    CUDA_VISIBLE_DEVICES=$i python3 $PY_SCRIPT \
        --cfg-path "$CFG_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --rank $i \
        --world-size $WORLD_SIZE &
        
    sleep 1
done

# 2. Wait for all processes
echo ">>> Waiting for all processes to finish..."
wait

echo ">>> All inference processes finished."
echo ">>> Done!"
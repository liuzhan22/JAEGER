#!/bin/bash

# Configuration
# 修改为上面Python脚本的实际路径
PY_SCRIPT="/path/to/inference/audio-visual/inference_batch_av.py"

# 配置文件
CFG_PATH="/path/to/configs/decode_config.yaml"

# 输出目录 (修改为你想要的路径)
OUTPUT_DIR="/path/to/results/dir"

# GPU数量
WORLD_SIZE=8

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ">>> Starting parallel AUDIO-VISUAL inference on $WORLD_SIZE GPUs..."

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
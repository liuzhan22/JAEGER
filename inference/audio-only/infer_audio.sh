#!/bin/bash

# Configuration
PY_SCRIPT="/path/to/inference/audio-only/inference_batch_foa.py"
CFG_PATH="/path/to/configs/decode_config.yaml"
OUTPUT_DIR="/path/to/output"
WORLD_SIZE=8

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ">>> Starting parallel inference on $WORLD_SIZE GPUs..."

# 1. Launch Loop
for ((i=0; i<WORLD_SIZE; i++)); do
    echo "Launching Rank $i on GPU $i..."
    
    # CUDA_VISIBLE_DEVICES isolates the GPU for each process
    # '&' puts the process in the background
    CUDA_VISIBLE_DEVICES=$i python3 $PY_SCRIPT \
        --cfg-path "$CFG_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --rank $i \
        --world-size $WORLD_SIZE &
        
    # Optional: sleep slightly to avoid file read race conditions on startup
    sleep 1
done

# 2. Wait for all processes
echo ">>> Waiting for all processes to finish..."
wait

echo ">>> All inference processes finished."

echo ">>> Done!"
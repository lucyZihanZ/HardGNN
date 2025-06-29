#!/bin/bash
set -e  # exit on any error

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Enable Python unbuffered mode for immediate output
export PYTHONUNBUFFERED=1

# Activate venv if using one
# source /path/to/your/python37/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p logs

# Get current timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/gowalla_${timestamp}.log"

echo "Starting experiment at $(date)" | tee -a "$log_file"
echo "Logging to: $log_file" | tee -a "$log_file"

# Run the Python script with arguments and redirect output to log
python main.py \
  --data gowalla \
  --lr 2e-3 \
  --reg 1e-2 \
  --temp 0.1 \
  --save_path gowalla \
  --epoch 150 \
  --batch 512 \
  --graphNum 3 \
  --gnn_layer 2 \
  --att_layer 1 \
  --test True \
  --testSize 1000 \
  --sampNum 40 \
  --use_hard_neg True \
  --hard_neg_top_k 5 \
  --contrastive_weight 0.1 2>&1 | tee -a "$log_file"

echo "Experiment completed at $(date)" | tee -a "$log_file" 
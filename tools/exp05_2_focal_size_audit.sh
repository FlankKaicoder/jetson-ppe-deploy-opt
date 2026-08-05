#!/usr/bin/env bash

set -uo pipefail

repo_dir="/root/autodl-tmp/jetson-ppe-deploy-opt-exp05"
python_bin="/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python"
weights="/root/autodl-tmp/jetson-ppe-outputs/exp05_2_focal_e100_20260805_162350/weights/best.pt"
data_yaml="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"
run_name="${1:-exp05_2_focal_size_audit_$(date +%Y%m%d_%H%M%S)}"
output_dir="$repo_dir/results/evaluation/$run_name"

cd "$repo_dir" || exit 1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -u tools/exp02_8_baseline_error_size_audit.py \
    --weights "$weights" \
    --data "$data_yaml" \
    --output-dir "$output_dir" \
    --imgsz 640 \
    --batch 16 \
    --conf .25 \
    --nms-iou .70 \
    --match-iou .50 \
    --max-visuals 30

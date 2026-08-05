#!/usr/bin/env bash

set -uo pipefail

repo_dir="/root/autodl-tmp/jetson-ppe-deploy-opt-exp05"
python_bin="/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python"
weights="/root/autodl-tmp/jetson-ppe-outputs/exp05_2_focal_e100_20260805_162350/weights/best.pt"
data_yaml="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"
eval_root="/root/autodl-tmp/jetson-ppe-outputs"
run_name="${1:-exp05_2_focal_test_$(date +%Y%m%d_%H%M%S)}"
report_dir="$repo_dir/results/evaluation/$run_name"

mkdir -p "$report_dir"
cd "$repo_dir" || exit 1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" -u tools/exp05_2_focal_test.py \
    --weights "$weights" \
    --data "$data_yaml" \
    --eval-root "$eval_root" \
    --run-name "$run_name" \
    --report-dir "$report_dir" \
    2>&1 | tee "$report_dir/run.log"

return_code=${PIPESTATUS[0]}
exit "$return_code"

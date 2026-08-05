#!/usr/bin/env bash

set -uo pipefail

repo_dir="/root/autodl-tmp/jetson-ppe-deploy-opt-exp05"
python_bin="/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python"
pretrained="/root/autodl-tmp/models/ultralytics/yolo11n.pt"
data_yaml="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"
output_root="/root/autodl-tmp/jetson-ppe-outputs"
run_name="${1:-exp05_1_attention_e100_$(date +%Y%m%d_%H%M%S)}"
report_dir="$repo_dir/results/training/$run_name"

mkdir -p "$report_dir"
cd "$repo_dir" || {
    echo "ERROR: Exp05 worktree not found"
    exit 1
}

export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

echo "experiment=Exp05.1 attention formal training"
echo "run_name=$run_name"
echo "report_dir=$report_dir"

"$python_bin" -u tools/exp05_1_attention_train.py \
    --pretrained "$pretrained" \
    --data "$data_yaml" \
    --output-root "$output_root" \
    --run-name "$run_name" \
    --report-dir "$report_dir" \
    2>&1 | tee "$report_dir/run.log"

python_return_code=${PIPESTATUS[0]}

if [ "$python_return_code" -eq 0 ]; then
    echo "exp05_1_attention_train=PASS"
else
    echo "exp05_1_attention_train=FAIL"
fi

exit "$python_return_code"

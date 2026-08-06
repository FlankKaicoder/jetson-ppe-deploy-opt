#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

WEIGHTS="/root/autodl-tmp/jetson-ppe-outputs/exp02_6_yolo11n_baseline_e100_20260804_185444/weights/best.pt"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_ROOT="$REPO_DIR/results/model_design/exp04_1b_strict_numerical_replay_${TIMESTAMP}"

mkdir -p "$OUTPUT_ROOT"

cd "$REPO_DIR" || exit 1

if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON="$(command -v python3)"
fi

if [ ! -f "$WEIGHTS" ]; then
    echo "ERROR: weights not found:"
    echo "$WEIGHTS"
    exit 1
fi

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

run_case()
{
    local CASE_NAME="$1"
    local DEVICE="$2"

    local CASE_DIR="$OUTPUT_ROOT/$CASE_NAME"
    local LOG="$CASE_DIR/console.log"
    local RC_FILE="$CASE_DIR/return_code.txt"

    mkdir -p "$CASE_DIR"

    echo
    echo "============================================================"
    echo " case=$CASE_NAME"
    echo " device=$DEVICE"
    echo "============================================================"

    "$PYTHON" -u - \
        "$DEVICE" \
        "$CASE_DIR" \
        "$WEIGHTS" <<'PY' \
        2>&1 | tee "$LOG"

import runpy
import sys

import torch

device = sys.argv[1]
output_dir = sys.argv[2]
weights = sys.argv[3]

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass

print("=" * 60)
print(" Exp04.1b Strict Numerical Runtime Settings")
print("=" * 60)
print(f"requested_device={device}")
print(
    "cuda_available="
    f"{torch.cuda.is_available()}"
)
print(
    "cudnn_benchmark="
    f"{torch.backends.cudnn.benchmark}"
)
print(
    "cudnn_deterministic="
    f"{torch.backends.cudnn.deterministic}"
)
print(
    "cudnn_allow_tf32="
    f"{torch.backends.cudnn.allow_tf32}"
)
print(
    "matmul_allow_tf32="
    f"{torch.backends.cuda.matmul.allow_tf32}"
)
print(
    "float32_matmul_precision="
    f"{torch.get_float32_matmul_precision()}"
)
print()

sys.argv = [
    "exp04_1_yolo11n_reparam_probe.py",
    "--weights",
    weights,
    "--output-dir",
    output_dir,
    "--imgsz",
    "640",
    "--device",
    device,
]

runpy.run_path(
    "tools/exp04_1_yolo11n_reparam_probe.py",
    run_name="__main__",
)
PY

    local PYTHON_RC=${PIPESTATUS[0]}

    echo "$PYTHON_RC" > "$RC_FILE"

    echo
    echo "case=$CASE_NAME"
    echo "python_return_code=$PYTHON_RC"

    if [ -f "$CASE_DIR/summary.txt" ]; then
        grep -E \
            'target_indices=|baseline_parameter_count=|rep_training_parameter_count=|rep_deploy_parameter_count=|baseline_vs_training_max_abs_error=|training_vs_deploy_max_abs_error=|training_vs_deploy_mean_abs_error=|training_vs_deploy_relative_l2_error=|baseline_vs_deploy_max_abs_error=|baseline_vs_deploy_relative_l2_error=|overall=' \
            "$CASE_DIR/summary.txt" || true
    else
        echo "summary_not_generated=YES"
    fi

    return 0
}

run_case "cuda_fp32_tf32_off" "0"
run_case "cpu_fp32" "cpu"

echo
echo "============================================================"
echo " Exp04.1b Strict Numerical Replay Summary"
echo "============================================================"
echo "output_root=$OUTPUT_ROOT"

for CASE_NAME in \
    cuda_fp32_tf32_off \
    cpu_fp32
do
    CASE_DIR="$OUTPUT_ROOT/$CASE_NAME"

    echo
    echo "---------- $CASE_NAME ----------"

    if [ -f "$CASE_DIR/summary.txt" ]; then
        grep -E \
            'baseline_vs_training_max_abs_error=|training_vs_deploy_max_abs_error=|training_vs_deploy_mean_abs_error=|training_vs_deploy_relative_l2_error=|baseline_vs_deploy_max_abs_error=|baseline_vs_deploy_relative_l2_error=|overall=' \
            "$CASE_DIR/summary.txt" || true
    else
        echo "summary_not_generated=YES"
    fi

    if [ -f "$CASE_DIR/return_code.txt" ]; then
        echo -n "python_return_code="
        cat "$CASE_DIR/return_code.txt"
    fi
done

echo
echo "exp04_1b_command_completed=YES"

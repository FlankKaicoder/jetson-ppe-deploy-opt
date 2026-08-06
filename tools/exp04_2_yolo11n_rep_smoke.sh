#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

PRETRAINED="/root/autodl-tmp/models/ultralytics/yolo11n.pt"

DATA_YAML="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"

OUTPUT_ROOT="/root/autodl-tmp/jetson-ppe-outputs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="exp04_2_yolo11n_rep_smoke_${TIMESTAMP}"

REPORT_DIR="$REPO_DIR/results/training/${RUN_NAME}"

RUN_LOG="$REPORT_DIR/run.log"
ENVIRONMENT="$REPORT_DIR/environment.txt"
ABNORMAL="$REPORT_DIR/abnormal.txt"
RUNNER_SUMMARY="$REPORT_DIR/runner_summary.txt"

mkdir -p "$REPORT_DIR"

cd "$REPO_DIR" || {
    echo "ERROR: repository directory not found"
    exit 1
}

if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON="$VENV_DIR/bin/python"
else
    PYTHON="$(command -v python3)"
fi

for REQUIRED in \
    "$PRETRAINED" \
    "$DATA_YAML" \
    "$PYTHON"
do
    if [ ! -e "$REQUIRED" ]; then
        echo "ERROR: required path not found:"
        echo "$REQUIRED"
        exit 1
    fi
done

{
    echo "experiment=Exp04.2"
    echo "timestamp=$TIMESTAMP"
    echo "repo_dir=$REPO_DIR"
    echo "run_name=$RUN_NAME"
    echo "report_dir=$REPORT_DIR"
    echo "output_root=$OUTPUT_ROOT"
    echo "pretrained=$PRETRAINED"
    echo "data_yaml=$DATA_YAML"
    echo "python=$PYTHON"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo
    "$PYTHON" --version

    "$PYTHON" - <<'PY'
import torch
import ultralytics

print(f"torch_version={torch.__version__}")
print(f"ultralytics_version={ultralytics.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(
        "gpu_name="
        f"{torch.cuda.get_device_name(0)}"
    )
PY
} > "$ENVIRONMENT" 2>&1

export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "============================================================"
echo " Exp04.2 YOLO11n-Rep One-Epoch Training Smoke Test"
echo "============================================================"
echo "run_name=$RUN_NAME"
echo "report_dir=$REPORT_DIR"
echo "pretrained=$PRETRAINED"
echo "data_yaml=$DATA_YAML"
echo "python=$PYTHON"
echo

"$PYTHON" -u \
    tools/exp04_2_yolo11n_rep_smoke.py \
    --pretrained "$PRETRAINED" \
    --data "$DATA_YAML" \
    --output-root "$OUTPUT_ROOT" \
    --run-name "$RUN_NAME" \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$RUN_LOG"

PYTHON_RETURN_CODE=${PIPESTATUS[0]}

grep -nE \
'Traceback|AssertionError|RuntimeError|FileNotFoundError|overall=FAIL|result=FAIL' \
"$RUN_LOG" \
> "$ABNORMAL" || true

if (
    [ "$PYTHON_RETURN_CODE" -eq 0 ] &&
    [ -f "$REPORT_DIR/summary.txt" ] &&
    grep -q '^overall=PASS$' \
        "$REPORT_DIR/summary.txt" &&
    [ ! -s "$ABNORMAL" ]
); then
    FINAL_RESULT="PASS"
else
    FINAL_RESULT="FAIL"
fi

{
    echo "============================================================"
    echo " Exp04.2 Runner Summary"
    echo "============================================================"
    echo "result=$FINAL_RESULT"
    echo "python_return_code=$PYTHON_RETURN_CODE"
    echo "run_name=$RUN_NAME"
    echo "report_dir=$REPORT_DIR"
    echo "summary=$REPORT_DIR/summary.txt"
    echo "summary_json=$REPORT_DIR/summary.json"
    echo "train_arguments=$REPORT_DIR/train_arguments.json"
    echo "environment=$ENVIRONMENT"
    echo "abnormal=$ABNORMAL"
} | tee "$RUNNER_SUMMARY"

echo
echo "========== experiment summary =========="

if [ -f "$REPORT_DIR/summary.txt" ]; then
    cat "$REPORT_DIR/summary.txt"
else
    echo "summary.txt was not generated"
fi

echo
echo "========== abnormal =========="

if [ -s "$ABNORMAL" ]; then
    cat "$ABNORMAL"
else
    echo "No abnormal messages detected."
fi

echo
echo "exp04_2_command_completed=YES"

if [ "$FINAL_RESULT" = "PASS" ]; then
    exit 0
else
    exit 1
fi

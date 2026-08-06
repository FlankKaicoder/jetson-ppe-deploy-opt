#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

TRAINING_WEIGHTS="/root/autodl-tmp/jetson-ppe-outputs/exp04_3_yolo11n_rep_e100_20260804_223547/weights/best.pt"

DATA_YAML="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"

OUTPUT_ROOT="/root/autodl-tmp/jetson-ppe-outputs"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

RUN_NAME="exp04_4_yolo11n_rep_deploy_${TIMESTAMP}"

ARTIFACT_DIR="$OUTPUT_ROOT/$RUN_NAME"

REPORT_DIR="$REPO_DIR/results/model_conversion/$RUN_NAME"

RUN_LOG="$REPORT_DIR/run.log"
ENVIRONMENT="$REPORT_DIR/environment.txt"
ABNORMAL="$REPORT_DIR/abnormal.txt"
RUNNER_SUMMARY="$REPORT_DIR/runner_summary.txt"

mkdir -p \
  "$REPORT_DIR" \
  "$ARTIFACT_DIR"

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
    "$TRAINING_WEIGHTS" \
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
    echo "experiment=Exp04.4"
    echo "timestamp=$TIMESTAMP"
    echo "repo_dir=$REPO_DIR"
    echo "run_name=$RUN_NAME"
    echo "training_weights=$TRAINING_WEIGHTS"
    echo "data_yaml=$DATA_YAML"
    echo "artifact_dir=$ARTIFACT_DIR"
    echo "report_dir=$REPORT_DIR"
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
echo " Exp04.4 Rep Training-to-Deploy Conversion and Validation"
echo "============================================================"
echo "training_weights=$TRAINING_WEIGHTS"
echo "artifact_dir=$ARTIFACT_DIR"
echo "report_dir=$REPORT_DIR"
echo

"$PYTHON" -u \
    tools/exp04_4_convert_and_validate.py \
    --training-weights "$TRAINING_WEIGHTS" \
    --data "$DATA_YAML" \
    --artifact-dir "$ARTIFACT_DIR" \
    --report-dir "$REPORT_DIR" \
    2>&1 | tee "$RUN_LOG"

PYTHON_RETURN_CODE=${PIPESTATUS[0]}

grep -nE \
'Traceback|AssertionError|RuntimeError|FileNotFoundError|torch.OutOfMemoryError|CUDA out of memory|overall=FAIL|result=FAIL' \
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
    echo " Exp04.4 Runner Summary"
    echo "============================================================"
    echo "result=$FINAL_RESULT"
    echo "python_return_code=$PYTHON_RETURN_CODE"
    echo "run_name=$RUN_NAME"
    echo "artifact_dir=$ARTIFACT_DIR"
    echo "report_dir=$REPORT_DIR"
    echo "summary=$REPORT_DIR/summary.txt"
    echo "summary_json=$REPORT_DIR/summary.json"
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
echo "========== generated weights =========="

find "$ARTIFACT_DIR/weights" \
    -maxdepth 1 \
    -type f \
    -printf '%f %s bytes\n' \
    2>/dev/null || true

echo
echo "========== abnormal =========="

if [ -s "$ABNORMAL" ]; then
    cat "$ABNORMAL"
else
    echo "No abnormal messages detected."
fi

echo
echo "exp04_4_command_completed=YES"

if [ "$FINAL_RESULT" = "PASS" ]; then
    exit 0
else
    exit 1
fi

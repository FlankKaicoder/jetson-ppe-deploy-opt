#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

BASELINE_WEIGHTS="/root/autodl-tmp/jetson-ppe-outputs/exp02_6_yolo11n_baseline_e100_20260804_185444/weights/best.pt"

REP_TRAINING_WEIGHTS="/root/autodl-tmp/jetson-ppe-outputs/exp04_3_yolo11n_rep_e100_20260804_223547/weights/best.pt"

REP_DEPLOY_WEIGHTS="/root/autodl-tmp/jetson-ppe-outputs/exp04_4_yolo11n_rep_deploy_20260804_225458/weights/best_deploy_fp32.pt"

DATA_YAML="/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_DIR="$REPO_DIR/results/evaluation/exp04_5a_reparam_independent_test_${TIMESTAMP}"

RUN_LOG="$OUTPUT_DIR/run.log"
ENVIRONMENT="$OUTPUT_DIR/environment.txt"
ABNORMAL="$OUTPUT_DIR/abnormal.txt"
RUNNER_SUMMARY="$OUTPUT_DIR/runner_summary.txt"

mkdir -p "$OUTPUT_DIR"

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
    "$BASELINE_WEIGHTS" \
    "$REP_TRAINING_WEIGHTS" \
    "$REP_DEPLOY_WEIGHTS" \
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
    echo "experiment=Exp04.5a"
    echo "timestamp=$TIMESTAMP"
    echo "repo_dir=$REPO_DIR"
    echo "output_dir=$OUTPUT_DIR"
    echo "baseline_weights=$BASELINE_WEIGHTS"
    echo "rep_training_weights=$REP_TRAINING_WEIGHTS"
    echo "rep_deploy_weights=$REP_DEPLOY_WEIGHTS"
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
echo " Exp04.5a Reparameterization Independent Test Evaluation"
echo "============================================================"
echo "output_dir=$OUTPUT_DIR"
echo "baseline_weights=$BASELINE_WEIGHTS"
echo "rep_training_weights=$REP_TRAINING_WEIGHTS"
echo "rep_deploy_weights=$REP_DEPLOY_WEIGHTS"
echo

"$PYTHON" -u \
    tools/exp04_5a_reparam_independent_test.py \
    --baseline-weights "$BASELINE_WEIGHTS" \
    --rep-training-weights "$REP_TRAINING_WEIGHTS" \
    --rep-deploy-weights "$REP_DEPLOY_WEIGHTS" \
    --data "$DATA_YAML" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$RUN_LOG"

PYTHON_RETURN_CODE=${PIPESTATUS[0]}

grep -nE \
'Traceback|AssertionError|RuntimeError|FileNotFoundError|torch.OutOfMemoryError|CUDA out of memory|overall=FAIL|result=FAIL' \
"$RUN_LOG" \
> "$ABNORMAL" || true

if (
    [ "$PYTHON_RETURN_CODE" -eq 0 ] &&
    [ -f "$OUTPUT_DIR/summary.txt" ] &&
    grep -q '^overall=PASS$' \
        "$OUTPUT_DIR/summary.txt" &&
    [ ! -s "$ABNORMAL" ]
); then
    FINAL_RESULT="PASS"
else
    FINAL_RESULT="FAIL"
fi

{
    echo "============================================================"
    echo " Exp04.5a Runner Summary"
    echo "============================================================"
    echo "result=$FINAL_RESULT"
    echo "python_return_code=$PYTHON_RETURN_CODE"
    echo "output_dir=$OUTPUT_DIR"
    echo "summary=$OUTPUT_DIR/summary.txt"
    echo "summary_json=$OUTPUT_DIR/summary.json"
    echo "overall_metrics=$OUTPUT_DIR/overall_metrics.csv"
    echo "per_class_metrics=$OUTPUT_DIR/per_class_metrics.csv"
    echo "environment=$ENVIRONMENT"
    echo "abnormal=$ABNORMAL"
} | tee "$RUNNER_SUMMARY"

echo
echo "========== experiment summary =========="

if [ -f "$OUTPUT_DIR/summary.txt" ]; then
    cat "$OUTPUT_DIR/summary.txt"
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
echo "exp04_5a_command_completed=YES"

if [ "$FINAL_RESULT" = "PASS" ]; then
    exit 0
else
    exit 1
fi

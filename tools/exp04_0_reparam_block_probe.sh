#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

OUTPUT_DIR="$REPO_DIR/results/model_design/exp04_0_reparam_block_probe_${TIMESTAMP}"

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

{
    echo "experiment=Exp04.0"
    echo "timestamp=$TIMESTAMP"
    echo "repo_dir=$REPO_DIR"
    echo "output_dir=$OUTPUT_DIR"
    echo "python=$PYTHON"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo
    "$PYTHON" --version
    "$PYTHON" - <<'PY'
import torch

print(f"torch_version={torch.__version__}")
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
echo " Exp04.0 Reparameterization Block Probe"
echo "============================================================"
echo "output_dir=$OUTPUT_DIR"
echo "python=$PYTHON"
echo

"$PYTHON" -u \
    tools/exp04_0_reparam_block_probe.py \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$RUN_LOG"

PYTHON_RETURN_CODE=${PIPESTATUS[0]}

grep -nE \
'Traceback|AssertionError|RuntimeError|overall=FAIL|result=FAIL' \
"$RUN_LOG" \
> "$ABNORMAL" || true

if (
    [ "$PYTHON_RETURN_CODE" -eq 0 ] &&
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
    echo " Exp04.0 Runner Summary"
    echo "============================================================"
    echo "result=$FINAL_RESULT"
    echo "python_return_code=$PYTHON_RETURN_CODE"
    echo "output_dir=$OUTPUT_DIR"
    echo "summary=$OUTPUT_DIR/summary.txt"
    echo "metrics=$OUTPUT_DIR/metrics.csv"
    echo "environment=$ENVIRONMENT"
    echo "abnormal=$ABNORMAL"
} | tee "$RUNNER_SUMMARY"

echo
echo "========== experiment summary =========="
cat "$OUTPUT_DIR/summary.txt"

echo
echo "========== abnormal =========="
if [ -s "$ABNORMAL" ]; then
    cat "$ABNORMAL"
else
    echo "No abnormal messages detected."
fi

echo
echo "exp04_0_command_completed=YES"

if [ "$FINAL_RESULT" = "PASS" ]; then
    exit 0
else
    exit 1
fi

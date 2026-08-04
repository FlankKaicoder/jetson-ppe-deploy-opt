#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="${REPO_DIR}/.venv-autodl"

DATASET_ROOT="/root/autodl-tmp/datasets/sources/construction-ppe_ultralytics_2025_v1"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/results/dataset_audit/exp02_3_raw_dataset_deep_audit_${TIMESTAMP}"

LOG="${OUT_DIR}/run.log"
ABNORMAL="${OUT_DIR}/abnormal.txt"

mkdir -p "$OUT_DIR"

cd "$REPO_DIR" || {
    echo "ERROR: repository not found: $REPO_DIR"
    exit 2
}

if [ ! -x "$VENV_DIR/bin/python" ]; then
    echo "ERROR: virtual environment not found: $VENV_DIR"
    exit 3
fi

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo " Exp02.3: Raw Dataset Deep Audit"
echo "============================================================"
echo "timestamp=$(date --iso-8601=seconds)"
echo "repo_dir=$REPO_DIR"
echo "dataset_root=$DATASET_ROOT"
echo "out_dir=$OUT_DIR"
echo "python=$(command -v python)"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: dataset root not found: $DATASET_ROOT"
    exit 4
fi

echo
echo "========== Python syntax check =========="

python -m py_compile \
    tools/exp02_3_raw_dataset_deep_audit.py

SYNTAX_RC=$?

echo "python_syntax_return_code=$SYNTAX_RC"

if [ "$SYNTAX_RC" -ne 0 ]; then
    echo "ERROR: Python syntax check failed"
    exit "$SYNTAX_RC"
fi

echo
echo "========== run deep audit =========="

PYTHONUNBUFFERED=1 \
python tools/exp02_3_raw_dataset_deep_audit.py \
    --dataset-root "$DATASET_ROOT" \
    --out-dir "$OUT_DIR" \
    2>&1 | tee "$LOG"

AUDIT_RC=${PIPESTATUS[0]}

{
    grep -nEi \
        'traceback|segmentation fault|memoryerror|permission denied|runtimeerror|syntaxerror|ERROR:' \
        "$LOG" || true
} > "$ABNORMAL"

if [ ! -s "$ABNORMAL" ]; then
    echo "No abnormal execution messages detected." \
        > "$ABNORMAL"
fi

echo
echo "========== return code =========="
echo "exp02_3_return_code=$AUDIT_RC"

if [ -f "$OUT_DIR/summary.txt" ]; then
    echo
    echo "========== summary =========="
    cat "$OUT_DIR/summary.txt"
fi

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "exp02_3_out_dir=$OUT_DIR"

if [ "$AUDIT_RC" -eq 0 ]; then
    echo "exp02_3_command_completed=YES"
else
    echo "exp02_3_command_completed=NO"
fi

exit "$AUDIT_RC"

#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="${REPO_DIR}/.venv-autodl"

SOURCE_ROOT="/root/autodl-tmp/datasets/sources/construction-ppe_ultralytics_2025_v1"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DATASET_OUT="/root/autodl-tmp/datasets/derived/construction_ppe3_canonical_v1_${TIMESTAMP}"

REPORT_OUT="${REPO_DIR}/results/dataset_audit/exp02_4a_ppe3_canonical_${TIMESTAMP}"

LOG="${REPORT_OUT}/run.log"
ABNORMAL="${REPORT_OUT}/abnormal.txt"

mkdir -p "$REPORT_OUT"

cd "$REPO_DIR" || {
    echo "ERROR: repository not found"
    exit 2
}

source "$VENV_DIR/bin/activate"

echo "============================================================"
echo " Exp02.4a: Build Canonical PPE3 Pool"
echo "============================================================"
echo "source_root=$SOURCE_ROOT"
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"
echo "python=$(command -v python)"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

python -m py_compile \
    tools/exp02_4a_build_ppe3_canonical.py

SYNTAX_RC=$?

echo "python_syntax_return_code=$SYNTAX_RC"

if [ "$SYNTAX_RC" -ne 0 ]; then
    exit "$SYNTAX_RC"
fi

PYTHONUNBUFFERED=1 \
python tools/exp02_4a_build_ppe3_canonical.py \
    --source-root "$SOURCE_ROOT" \
    --dataset-out "$DATASET_OUT" \
    --report-out "$REPORT_OUT" \
    2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}

grep -nEi \
    'traceback|runtimeerror|permission denied|syntaxerror|ERROR:' \
    "$LOG" \
    > "$ABNORMAL" || true

if [ ! -s "$ABNORMAL" ]; then
    echo "No abnormal execution messages detected." \
        > "$ABNORMAL"
fi

echo
echo "========== return code =========="
echo "exp02_4a_return_code=$RC"

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"

if [ "$RC" -eq 0 ]; then
    echo "exp02_4a_command_completed=YES"
else
    echo "exp02_4a_command_completed=NO"
fi

exit "$RC"

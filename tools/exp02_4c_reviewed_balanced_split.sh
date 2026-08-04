#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="$REPO_DIR/.venv-autodl"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

cd "$REPO_DIR" || {
    echo "ERROR: repository not found"
    exit 2
}

source "$VENV_DIR/bin/activate"

V2_REPORT="$(
    find \
        "$REPO_DIR/results/dataset_audit" \
        -maxdepth 1 \
        -type d \
        -name 'exp02_4b_group_split_v2_*' \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

DEEP_AUDIT="$(
    find \
        "$REPO_DIR/results/dataset_audit" \
        -maxdepth 1 \
        -type d \
        -name 'exp02_3_raw_dataset_deep_audit_*' \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

V2_MANIFEST="$V2_REPORT/group_split_manifest.csv"
REVIEWED_PAIRS="$DEEP_AUDIT/cross_split_near_duplicates.csv"

DATASET_OUT="/root/autodl-tmp/datasets/derived/construction_ppe3_reviewed_split_v1_${TIMESTAMP}"

REPORT_OUT="$REPO_DIR/results/dataset_audit/exp02_4c_reviewed_split_${TIMESTAMP}"

LOG="$REPORT_OUT/run.log"
ABNORMAL="$REPORT_OUT/abnormal.txt"

mkdir -p "$REPORT_OUT"

echo "============================================================"
echo " Exp02.4c: Reviewed and Balanced Dataset Split"
echo "============================================================"
echo "v2_report=$V2_REPORT"
echo "v2_manifest=$V2_MANIFEST"
echo "deep_audit=$DEEP_AUDIT"
echo "reviewed_pairs=$REVIEWED_PAIRS"
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"
echo "python=$(command -v python)"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

if [ ! -f "$V2_MANIFEST" ]; then
    echo "ERROR: v2 manifest not found"
    exit 3
fi

if [ ! -f "$REVIEWED_PAIRS" ]; then
    echo "ERROR: reviewed pair CSV not found"
    exit 4
fi

python -m py_compile \
    tools/exp02_4c_reviewed_balanced_split.py

SYNTAX_RC=$?

echo "python_syntax_return_code=$SYNTAX_RC"

if [ "$SYNTAX_RC" -ne 0 ]; then
    exit "$SYNTAX_RC"
fi

PYTHONUNBUFFERED=1 \
python tools/exp02_4c_reviewed_balanced_split.py \
    --v2-manifest "$V2_MANIFEST" \
    --reviewed-pairs "$REVIEWED_PAIRS" \
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
echo "exp02_4c_return_code=$RC"

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"

if [ "$RC" -eq 0 ]; then
    echo "exp02_4c_command_completed=YES"
else
    echo "exp02_4c_command_completed=NO"
fi

exit "$RC"

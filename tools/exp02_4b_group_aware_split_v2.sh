#!/usr/bin/env bash

set -uo pipefail

REPO_DIR="/root/autodl-tmp/jetson-ppe-deploy-opt"
VENV_DIR="${REPO_DIR}/.venv-autodl"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

cd "$REPO_DIR" || {
    echo "ERROR: repository not found"
    exit 2
}

source "$VENV_DIR/bin/activate"

CANONICAL_REPORT="$(
    find \
        "$REPO_DIR/results/dataset_audit" \
        -maxdepth 1 \
        -type d \
        -name 'exp02_4a_ppe3_canonical_*' \
        -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
)"

if [ -z "$CANONICAL_REPORT" ]; then
    echo "ERROR: completed Exp02.4a report not found"
    exit 3
fi

if [ ! -f "$CANONICAL_REPORT/summary.json" ]; then
    echo "ERROR: Exp02.4a summary.json not found"
    exit 4
fi

CANONICAL_ROOT="$(
    python - "$CANONICAL_REPORT/summary.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])

data = json.loads(
    path.read_text(encoding="utf-8")
)

print(data["dataset_out"])
PY
)"

CANONICAL_MANIFEST="$CANONICAL_REPORT/canonical_manifest.csv"

DATASET_OUT="/root/autodl-tmp/datasets/derived/construction_ppe3_group_split_v2_${TIMESTAMP}"

REPORT_OUT="$REPO_DIR/results/dataset_audit/exp02_4b_group_split_v2_${TIMESTAMP}"

LOG="$REPORT_OUT/run.log"
ABNORMAL="$REPORT_OUT/abnormal.txt"

mkdir -p "$REPORT_OUT"

echo "============================================================"
echo " Exp02.4b-v2: Conservative Group-Aware Split"
echo "============================================================"
echo "canonical_report=$CANONICAL_REPORT"
echo "canonical_root=$CANONICAL_ROOT"
echo "canonical_manifest=$CANONICAL_MANIFEST"
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"
echo "python=$(command -v python)"
echo "git_branch=$(git branch --show-current)"
echo "git_commit=$(git rev-parse HEAD)"

python -m py_compile \
    tools/exp02_4b_group_aware_split_v2.py

SYNTAX_RC=$?

echo "python_syntax_return_code=$SYNTAX_RC"

if [ "$SYNTAX_RC" -ne 0 ]; then
    exit "$SYNTAX_RC"
fi

PYTHONUNBUFFERED=1 \
python tools/exp02_4b_group_aware_split_v2.py \
    --canonical-root "$CANONICAL_ROOT" \
    --canonical-manifest "$CANONICAL_MANIFEST" \
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
echo "exp02_4b_v2_return_code=$RC"

echo
echo "========== abnormal =========="
cat "$ABNORMAL"

echo
echo "dataset_out=$DATASET_OUT"
echo "report_out=$REPORT_OUT"

if [ "$RC" -eq 0 ]; then
    echo "exp02_4b_v2_command_completed=YES"
else
    echo "exp02_4b_v2_command_completed=NO"
fi

exit "$RC"

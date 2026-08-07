#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_AUTODL_PYTHON:-$repo_dir/.venv-autodl/bin/python}"
dataset_root="${PPE_DATASET_ROOT:-/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104}"
artifact_root="${PPE_INT8_ARTIFACT_ROOT:-/root/autodl-tmp/jetson-ppe-artifacts/exp08}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp08_0_prepare_calibration_${timestamp}"
report_dir="$repo_dir/results/int8/$run_name"
artifact_dir="$artifact_root/$run_name"

if [ -e "$report_dir" ] || [ -e "$artifact_dir" ]; then
    echo "ERROR: timestamp output already exists"
    exit 1
fi
mkdir -p "$report_dir" "$artifact_dir"
run_log="$report_dir/run.log"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    printf '%s\n' 1 > "$report_dir/return_code.txt"
    printf 'result=FAIL\nreason=%s\n' "$message" > "$report_dir/failure_summary.txt"
    exit 1
}

[ -d "$repo_dir" ] || fail_early "repository directory missing: $repo_dir"
[ -x "$python_bin" ] || fail_early "Python missing: $python_bin"
[ -d "$dataset_root/images/train" ] || fail_early "train image directory missing"
[ -d "$dataset_root/labels/train" ] || fail_early "train label directory missing"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/08-*) ;;
    *) fail_early "unexpected Git branch: $(git branch --show-current)" ;;
esac

image_count="$(find "$dataset_root/images/train" -maxdepth 1 -type f | wc -l)"
label_count="$(find "$dataset_root/labels/train" -maxdepth 1 -type f | wc -l)"
[ "$image_count" -eq 980 ] || fail_early "train image count is $image_count"
[ "$label_count" -eq 980 ] || fail_early "train label count is $label_count"

{
    echo "experiment=Exp08.0 train-only INT8 calibration preparation"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "whoami=$(whoami)"
    echo "uname=$(uname -a)"
    echo "repo_dir=$repo_dir"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$python_bin"
    "$python_bin" --version
    echo "dataset_root=$dataset_root"
    echo "source_split=train"
    echo "train_image_count=$image_count"
    echo "train_label_count=$label_count"
    echo "sample_count=256"
    echo "seed=42"
    echo "candidate_count=1024"
    echo "max_proportion_delta=0.02"
    echo "test_split_used=false"
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "$python_bin tools/exp08_prepare_calibration.py --dataset-root $dataset_root --report-dir $report_dir --artifact-dir $artifact_dir --sample-count 256 --seed 42 --candidate-count 1024 --max-proportion-delta 0.02" \
    > "$report_dir/command.txt"

"$python_bin" -u tools/exp08_prepare_calibration.py \
    --dataset-root "$dataset_root" \
    --report-dir "$report_dir" \
    --artifact-dir "$artifact_dir" \
    --sample-count 256 \
    --seed 42 \
    --candidate-count 1024 \
    --max-proportion-delta 0.02 \
    2>&1 | tee "$run_log"
python_return_code=${PIPESTATUS[0]}
printf '%s\n' "$python_return_code" > "$report_dir/python_return_code.txt"

archive_path="$artifact_dir/construction_ppe3_train_calibration_256.tar.gz"
grep -nE 'Traceback|FATAL:|ERROR:|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$python_return_code" -eq 0 ] && \
    [ -s "$report_dir/summary.json" ] && \
    [ -s "$report_dir/calibration_manifest.json" ] && \
    [ -s "$archive_path" ] && \
    grep -q '"result": "PASS"' "$report_dir/summary.json" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi

{
    echo "result=$final_result"
    echo "python_return_code=$python_return_code"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    if [ -s "$archive_path" ]; then
        echo "archive_sha256=$(sha256sum "$archive_path" | cut -d ' ' -f 1)"
    fi
} > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"

if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
    exit "$final_code"
fi

echo "exp08_0_prepare_calibration=PASS"
exit 0

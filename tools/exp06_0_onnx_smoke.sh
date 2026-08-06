#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_PYTHON_BIN:-/root/autodl-tmp/jetson-ppe-deploy-opt/.venv-autodl/bin/python}"
weights="${PPE_BASELINE_WEIGHTS:-/root/autodl-tmp/jetson-ppe-outputs/exp02_6_yolo11n_baseline_e100_20260804_185444/weights/best.pt}"
expected_weight_sha256="79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6"
data_yaml="${PPE_DATA_YAML:-/root/autodl-tmp/datasets/derived/construction_ppe3_final_split_v1_20260804_175104/construction_ppe3.yaml}"
artifact_root="${PPE_ARTIFACT_ROOT:-/root/autodl-tmp/jetson-ppe-artifacts}"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp06_0_onnx_smoke_${timestamp}"
report_dir="$repo_dir/results/onnx/$run_name"
artifact_dir="$artifact_root/$run_name"

run_log="$report_dir/run.log"
environment_file="$report_dir/environment.txt"
abnormal_file="$report_dir/abnormal.txt"
runner_summary="$report_dir/runner_summary.txt"
return_code_file="$report_dir/return_code.txt"

if [ -e "$report_dir" ] || [ -e "$artifact_dir" ]; then
    echo "ERROR: timestamp output already exists"
    exit 1
fi

mkdir -p "$report_dir" "$artifact_dir"

if [ ! -d "$repo_dir" ]; then
    echo "ERROR: repository directory not found: $repo_dir"
    exit 1
fi

if [ ! -x "$python_bin" ]; then
    echo "ERROR: Python interpreter not executable: $python_bin"
    exit 1
fi

if [ ! -s "$weights" ]; then
    echo "ERROR: frozen baseline weight not found: $weights"
    exit 1
fi

if [ ! -f "$data_yaml" ]; then
    echo "ERROR: dataset YAML not found: $data_yaml"
    exit 1
fi

actual_weight_sha256="$(sha256sum "$weights" | awk '{print $1}')"
if [ "$actual_weight_sha256" != "$expected_weight_sha256" ]; then
    echo "ERROR: frozen baseline SHA256 mismatch"
    echo "expected=$expected_weight_sha256"
    echo "actual=$actual_weight_sha256"
    exit 1
fi

cd "$repo_dir" || exit 1
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

{
    echo "experiment=Exp06.0 ONNX smoke"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "whoami=$(whoami)"
    echo "repo_dir=$repo_dir"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$python_bin"
    echo "weights=$weights"
    echo "weight_sha256=$actual_weight_sha256"
    echo "data_yaml=$data_yaml"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    "$python_bin" --version
    "$python_bin" -m pip show torch ultralytics onnx onnxruntime
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} > "$environment_file" 2>&1

{
    echo "$python_bin -u tools/exp06_export_onnx.py"
    echo "--run-kind smoke --imgsz 640 --batch 1 --opset 17"
    echo "--weights $weights"
    echo "--data $data_yaml"
    echo "--report-dir $report_dir"
    echo "--artifact-dir $artifact_dir"
} > "$report_dir/command.txt"

echo "============================================================"
echo " Exp06.0 ONNX Smoke Test"
echo "============================================================"
echo "report_dir=$report_dir"
echo "artifact_dir=$artifact_dir"

"$python_bin" -u tools/exp06_export_onnx.py \
    --repo-root "$repo_dir" \
    --weights "$weights" \
    --expected-weight-sha256 "$expected_weight_sha256" \
    --data "$data_yaml" \
    --report-dir "$report_dir" \
    --artifact-dir "$artifact_dir" \
    --run-kind smoke \
    --imgsz 640 \
    --batch 1 \
    --opset 17 \
    2>&1 | tee "$run_log"

python_return_code=${PIPESTATUS[0]}
echo "$python_return_code" > "$return_code_file"

grep -nE \
    'Traceback|FATAL:|result=FAIL|raw_tensor_result=FAIL|detection_result=FAIL' \
    "$run_log" > "$abnormal_file"

if [ "$python_return_code" -eq 0 ] && \
    [ -s "$report_dir/summary.json" ] && \
    grep -q '^result=PASS$' "$report_dir/summary.txt" && \
    [ ! -s "$abnormal_file" ]; then
    final_result="PASS"
else
    final_result="FAIL"
fi

{
    echo "result=$final_result"
    echo "python_return_code=$python_return_code"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    echo "summary=$report_dir/summary.txt"
    echo "manifest=$report_dir/artifact_manifest.json"
} | tee "$runner_summary"

if [ "$final_result" != "PASS" ]; then
    {
        echo "result=FAIL"
        echo "python_return_code=$python_return_code"
        echo "last_log_lines:"
        tr '\r' '\n' < "$run_log" | tail -n 80
    } > "$report_dir/failure_summary.txt"
    exit 1
fi

echo "exp06_0_onnx_smoke=PASS"
exit 0

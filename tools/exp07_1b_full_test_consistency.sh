#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
onnx_path="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
onnx_reference_summary="$repo_dir/results/onnx/exp06_1_onnx_formal_20260806_155847/summary.json"
pytorch_reference="/home/nvidia/models/jetson-ppe/exp02/yolo11n_baseline_exp02_best.pt"
fp32_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp32.engine"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
data_yaml="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/construction_ppe3_jetson_test.yaml"
dataset_archive="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_split.tar.gz"
expected_onnx_sha256="305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8"
expected_pytorch_sha256="79dad73ccad09d46299083078f6d7e19c38541bc19ac86a8d3f11e49661d6ae6"
expected_fp32_sha256="01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_dataset_sha256="3bf3addcb79e7ac46163f7a294265a92c5f84c7e633f56da0b16e22f33400f4a"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp07_1b_full_test_consistency_${timestamp}"
report_dir="$repo_dir/results/tensorrt/$run_name"
if [ -e "$report_dir" ]; then
    echo "ERROR: report directory already exists: $report_dir"
    exit 1
fi
mkdir -p "$report_dir"
run_log="$report_dir/run.log"
return_code_file="$report_dir/return_code.txt"
abnormal_file="$report_dir/abnormal.txt"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    echo 1 > "$return_code_file"
    {
        echo "result=FAIL"
        echo "reason=$message"
    } > "$report_dir/failure_summary.txt"
    exit 1
}

[ -x "$python_bin" ] || fail_early "Python not executable: $python_bin"
[ -s "$onnx_path" ] || fail_early "ONNX missing"
[ -s "$onnx_reference_summary" ] || fail_early "Exp06 ONNX reference summary missing"
[ -s "$pytorch_reference" ] || fail_early "PyTorch reference missing"
[ -s "$fp32_engine" ] || fail_early "FP32 engine missing"
[ -s "$fp16_engine" ] || fail_early "FP16 engine missing"
[ -s "$data_yaml" ] || fail_early "dataset YAML missing"
[ -s "$dataset_archive" ] || fail_early "dataset archive missing"

[ "$(sha256sum "$onnx_path" | cut -d ' ' -f 1)" = "$expected_onnx_sha256" ] || \
    fail_early "ONNX SHA256 mismatch"
[ "$(sha256sum "$pytorch_reference" | cut -d ' ' -f 1)" = "$expected_pytorch_sha256" ] || \
    fail_early "PyTorch SHA256 mismatch"
[ "$(sha256sum "$fp32_engine" | cut -d ' ' -f 1)" = "$expected_fp32_sha256" ] || \
    fail_early "FP32 engine SHA256 mismatch"
[ "$(sha256sum "$fp16_engine" | cut -d ' ' -f 1)" = "$expected_fp16_sha256" ] || \
    fail_early "FP16 engine SHA256 mismatch"
[ "$(sha256sum "$dataset_archive" | cut -d ' ' -f 1)" = "$expected_dataset_sha256" ] || \
    fail_early "dataset archive SHA256 mismatch"

image_count="$(find /home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/images/test -maxdepth 1 -type f | wc -l)"
label_count="$(find /home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/labels/test -maxdepth 1 -type f | wc -l)"
[ "$image_count" -eq 219 ] || fail_early "test image count is $image_count"
[ "$label_count" -eq 219 ] || fail_early "test label count is $label_count"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/07-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp07.1b full test consistency"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "onnx=$onnx_path"
    echo "onnx_sha256=$expected_onnx_sha256"
    echo "onnx_reference_summary=$onnx_reference_summary"
    echo "onnx_reference_summary_sha256=$(sha256sum "$onnx_reference_summary" | cut -d ' ' -f 1)"
    echo "pytorch_reference=$pytorch_reference"
    echo "pytorch_reference_sha256=$expected_pytorch_sha256"
    echo "fp32_engine=$fp32_engine"
    echo "fp32_engine_sha256=$expected_fp32_sha256"
    echo "fp16_engine=$fp16_engine"
    echo "fp16_engine_sha256=$expected_fp16_sha256"
    echo "data_yaml=$data_yaml"
    echo "dataset_archive_sha256=$expected_dataset_sha256"
    echo "test_image_count=$image_count"
    echo "test_label_count=$label_count"
    "$python_bin" --version
    "$python_bin" -m pip show ultralytics onnxruntime-gpu torch tensorrt
} > "$report_dir/environment.txt" 2>&1

{
    echo "$python_bin tools/exp07_full_test_consistency.py"
    echo "PyTorch reference: $pytorch_reference (same Jetson validator runtime)"
    echo "ONNX provenance: $onnx_reference_summary (AutoDL Exp06 formal)"
    echo "--imgsz 640 --batch 1 --workers 2 --rect false --split test"
    echo "FP32 thresholds: PR=2e-2 AP=5e-4 (post-failure calibration)"
    echo "FP16 thresholds: PR=2e-2 AP=1e-3 (post-failure max-F1 calibration)"
} > "$report_dir/command.txt"

YOLO_AUTOINSTALL=false "$python_bin" -u tools/exp07_full_test_consistency.py \
    --repo-root "$repo_dir" \
    --onnx "$onnx_path" \
    --onnx-reference-summary "$onnx_reference_summary" \
    --pytorch-reference "$pytorch_reference" \
    --fp32-engine "$fp32_engine" \
    --fp16-engine "$fp16_engine" \
    --data "$data_yaml" \
    --report-dir "$report_dir" \
    --imgsz 640 \
    --batch 1 \
    --workers 2 \
    --fp32-pr-atol 2e-2 \
    --fp32-ap-atol 5e-4 \
    --fp16-pr-atol 2e-2 \
    --fp16-ap-atol 1e-3 \
    2>&1 | tee "$run_log"
python_return_code=${PIPESTATUS[0]}
echo "$python_return_code" > "$return_code_file"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|fp32_result=FAIL|fp16_result=FAIL' \
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
} > "$report_dir/runner_summary.txt"

if [ "$final_result" != "PASS" ]; then
    {
        echo "result=FAIL"
        echo "python_return_code=$python_return_code"
        echo "last_log_lines:"
        tr '\r' '\n' < "$run_log" | tail -n 160
    } > "$report_dir/failure_summary.txt"
    exit 1
fi

echo "exp07_1b_full_test_consistency=PASS"
exit 0

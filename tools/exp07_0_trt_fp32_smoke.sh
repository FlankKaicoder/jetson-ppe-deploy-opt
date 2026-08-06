#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
trtexec_bin="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
onnx_path="${PPE_ONNX_PATH:-/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx}"
probe_image="${PPE_PROBE_IMAGE:-/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg}"
artifact_root="${PPE_TRT_ARTIFACT_ROOT:-/home/nvidia/models/jetson-ppe/exp07}"
expected_onnx_sha256="305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8"
expected_probe_sha256="39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp07_0_trt_fp32_smoke_${timestamp}"
report_dir="$repo_dir/results/tensorrt/$run_name"
artifact_dir="$artifact_root/$run_name"
engine_path="$artifact_dir/yolo11n_baseline_exp07_b1_640_fp32.engine"

if [ -e "$report_dir" ] || [ -e "$artifact_dir" ]; then
    echo "ERROR: timestamp output already exists"
    exit 1
fi

mkdir -p "$report_dir" "$artifact_dir"
run_log="$report_dir/run.log"
build_log="$report_dir/build.log"
environment_file="$report_dir/environment.txt"
command_file="$report_dir/command.txt"
return_code_file="$report_dir/return_code.txt"
runner_summary="$report_dir/runner_summary.txt"
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

[ -d "$repo_dir" ] || fail_early "repository directory not found: $repo_dir"
[ -x "$python_bin" ] || fail_early "Python not executable: $python_bin"
[ -x "$trtexec_bin" ] || fail_early "trtexec not executable: $trtexec_bin"
[ -s "$onnx_path" ] || fail_early "ONNX not found: $onnx_path"
[ -s "$probe_image" ] || fail_early "probe image not found: $probe_image"

actual_onnx_sha256="$(sha256sum "$onnx_path" | cut -d ' ' -f 1)"
actual_probe_sha256="$(sha256sum "$probe_image" | cut -d ' ' -f 1)"
[ "$actual_onnx_sha256" = "$expected_onnx_sha256" ] || \
    fail_early "ONNX SHA256 mismatch: $actual_onnx_sha256"
[ "$actual_probe_sha256" = "$expected_probe_sha256" ] || \
    fail_early "probe image SHA256 mismatch: $actual_probe_sha256"

cd "$repo_dir" || fail_early "cannot enter repository"
git_branch="$(git branch --show-current)"
case "$git_branch" in
    exp/07-*) ;;
    *) fail_early "unexpected Git branch: $git_branch" ;;
esac

{
    echo "experiment=Exp07.0 TensorRT FP32 smoke"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "whoami=$(whoami)"
    echo "uname=$(uname -a)"
    echo "repo_dir=$repo_dir"
    echo "git_branch=$git_branch"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$python_bin"
    "$python_bin" --version
    echo "trtexec=$trtexec_bin"
    dpkg-query -W libnvinfer10 libnvinfer-dev python3-libnvinfer
    /usr/local/cuda/bin/nvcc --version
    nvpmodel -q
    echo "jetson_clocks_state=NOT_CHECKED_NON_ROOT"
    echo "onnx=$onnx_path"
    echo "onnx_sha256=$actual_onnx_sha256"
    echo "probe_image=$probe_image"
    echo "probe_image_sha256=$actual_probe_sha256"
    echo "engine=$engine_path"
    echo "workspace_mib=1024"
    echo "builder_optimization_level=3"
    echo "tf32=disabled"
} > "$environment_file" 2>&1

{
    echo "$trtexec_bin"
    echo "--onnx=$onnx_path"
    echo "--saveEngine=$engine_path"
    echo "--noTF32 --memPoolSize=workspace:1024"
    echo "--builderOptimizationLevel=3"
    echo "--profilingVerbosity=detailed --skipInference --verbose"
    echo "$python_bin tools/exp07_trt_consistency.py"
    echo "--precision fp32 --imgsz 640 --confidence 0.25 --nms-iou 0.70"
    echo "raw thresholds: max=1e-3 mean=2e-5 relative_l2=1e-5"
    echo "detection thresholds: box=1e-3 confidence=1e-5"
} > "$command_file"

echo "============================================================" | tee "$run_log"
echo " Exp07.0 TensorRT FP32 Smoke Test" | tee -a "$run_log"
echo "============================================================" | tee -a "$run_log"
echo "report_dir=$report_dir" | tee -a "$run_log"
echo "artifact_dir=$artifact_dir" | tee -a "$run_log"

build_start="$(date +%s)"
"$trtexec_bin" \
    --onnx="$onnx_path" \
    --saveEngine="$engine_path" \
    --noTF32 \
    --memPoolSize=workspace:1024 \
    --builderOptimizationLevel=3 \
    --profilingVerbosity=detailed \
    --skipInference \
    --verbose \
    2>&1 | tee "$build_log" | tee -a "$run_log"
build_return_code=${PIPESTATUS[0]}
build_end="$(date +%s)"
echo "$build_return_code" > "$report_dir/build_return_code.txt"
echo "$((build_end - build_start))" > "$report_dir/build_seconds.txt"

consistency_return_code=99
if [ "$build_return_code" -eq 0 ] && [ -s "$engine_path" ]; then
    "$python_bin" -u tools/exp07_trt_consistency.py \
        --repo-root "$repo_dir" \
        --onnx "$onnx_path" \
        --engine "$engine_path" \
        --image "$probe_image" \
        --report-dir "$report_dir" \
        --precision fp32 \
        --imgsz 640 \
        --confidence 0.25 \
        --nms-iou 0.70 \
        --max-abs-atol 1e-3 \
        --mean-abs-atol 2e-5 \
        --relative-l2-atol 1e-5 \
        --box-atol 1e-3 \
        --confidence-atol 1e-5 \
        2>&1 | tee -a "$run_log"
    consistency_return_code=${PIPESTATUS[0]}
fi
echo "$consistency_return_code" > "$report_dir/consistency_return_code.txt"

grep -nE \
    'Traceback|FATAL:|ERROR:|result=FAIL|raw_tensor_result=FAIL|detection_result=FAIL' \
    "$run_log" > "$abnormal_file"

if [ "$build_return_code" -eq 0 ] && \
    [ "$consistency_return_code" -eq 0 ] && \
    [ -s "$engine_path" ] && \
    [ -s "$report_dir/summary.json" ] && \
    grep -q '^result=PASS$' "$report_dir/summary.txt" && \
    [ ! -s "$abnormal_file" ]; then
    final_result="PASS"
    final_return_code=0
else
    final_result="FAIL"
    final_return_code=1
fi

echo "$final_return_code" > "$return_code_file"
{
    echo "result=$final_result"
    echo "build_return_code=$build_return_code"
    echo "consistency_return_code=$consistency_return_code"
    echo "build_seconds=$((build_end - build_start))"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    echo "engine_path=$engine_path"
    if [ -s "$engine_path" ]; then
        echo "engine_size_bytes=$(stat -c %s "$engine_path")"
        echo "engine_sha256=$(sha256sum "$engine_path" | cut -d ' ' -f 1)"
    fi
} | tee "$runner_summary"

if [ "$final_result" != "PASS" ]; then
    {
        echo "result=FAIL"
        echo "build_return_code=$build_return_code"
        echo "consistency_return_code=$consistency_return_code"
        echo "last_log_lines:"
        tr '\r' '\n' < "$run_log" | tail -n 120
    } > "$report_dir/failure_summary.txt"
    exit 1
fi

echo "exp07_0_trt_fp32_smoke=PASS"
exit 0

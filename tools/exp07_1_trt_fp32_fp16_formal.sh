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
run_name="exp07_1_trt_fp32_fp16_formal_${timestamp}"
report_dir="$repo_dir/results/tensorrt/$run_name"
artifact_dir="$artifact_root/$run_name"
fp32_engine="$artifact_dir/yolo11n_baseline_exp07_b1_640_fp32.engine"
fp16_engine="$artifact_dir/yolo11n_baseline_exp07_b1_640_fp16.engine"

if [ -e "$report_dir" ] || [ -e "$artifact_dir" ]; then
    echo "ERROR: timestamp output already exists"
    exit 1
fi
mkdir -p "$report_dir/fp32" "$report_dir/fp16" "$artifact_dir"

run_log="$report_dir/run.log"
environment_file="$report_dir/environment.txt"
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
    echo "experiment=Exp07.1 TensorRT FP32 FP16 formal"
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
    echo "workspace_mib=1024"
    echo "builder_optimization_level=3"
    echo "tf32=disabled"
    echo "benchmark_scope=GPU_COMPUTE_ONLY_NO_H2D_D2H"
    echo "benchmark_warmup_ms=500"
    echo "benchmark_iterations=200"
    echo "benchmark_duration_seconds=0"
} > "$environment_file" 2>&1

{
    echo "FP32 build: --noTF32 --memPoolSize=workspace:1024 --builderOptimizationLevel=3"
    echo "FP16 build: --fp16 --noTF32 --memPoolSize=workspace:1024 --builderOptimizationLevel=3"
    echo "FP32 raw thresholds: max=1e-3 mean=2e-5 relative_l2=1e-5"
    echo "FP32 detection thresholds: box=1e-3 confidence=1e-5"
    echo "FP16 raw thresholds: max=0.5 mean=0.01 relative_l2=1e-3"
    echo "FP16 detection thresholds: box=0.5 confidence=1e-3"
    echo "Benchmark: warmUp=500ms iterations=200 duration=0 CUDA graph spin wait no data transfers"
} > "$report_dir/command.txt"

echo "============================================================" | tee "$run_log"
echo " Exp07.1 TensorRT FP32 / FP16 Formal" | tee -a "$run_log"
echo "============================================================" | tee -a "$run_log"
echo "report_dir=$report_dir" | tee -a "$run_log"
echo "artifact_dir=$artifact_dir" | tee -a "$run_log"

build_engine() {
    local precision="$1"
    local engine_path="$2"
    local log_path="$3"
    local start end return_code
    start="$(date +%s)"
    if [ "$precision" = "fp16" ]; then
        "$trtexec_bin" \
            --onnx="$onnx_path" \
            --saveEngine="$engine_path" \
            --fp16 \
            --noTF32 \
            --memPoolSize=workspace:1024 \
            --builderOptimizationLevel=3 \
            --profilingVerbosity=detailed \
            --skipInference \
            2>&1 | tee "$log_path" | tee -a "$run_log"
    else
        "$trtexec_bin" \
            --onnx="$onnx_path" \
            --saveEngine="$engine_path" \
            --noTF32 \
            --memPoolSize=workspace:1024 \
            --builderOptimizationLevel=3 \
            --profilingVerbosity=detailed \
            --skipInference \
            2>&1 | tee "$log_path" | tee -a "$run_log"
    fi
    return_code=${PIPESTATUS[0]}
    end="$(date +%s)"
    echo "$return_code" > "$report_dir/${precision}_build_return_code.txt"
    echo "$((end - start))" > "$report_dir/${precision}_build_seconds.txt"
}

build_engine fp32 "$fp32_engine" "$report_dir/fp32_build.log"
fp32_build_return_code="$(cat "$report_dir/fp32_build_return_code.txt")"
fp32_build_seconds="$(cat "$report_dir/fp32_build_seconds.txt")"
build_engine fp16 "$fp16_engine" "$report_dir/fp16_build.log"
fp16_build_return_code="$(cat "$report_dir/fp16_build_return_code.txt")"
fp16_build_seconds="$(cat "$report_dir/fp16_build_seconds.txt")"

run_consistency() {
    local precision="$1"
    local engine_path="$2"
    local max_abs="$3"
    local mean_abs="$4"
    local relative_l2="$5"
    local box_atol="$6"
    local confidence_atol="$7"
    "$python_bin" -u tools/exp07_trt_consistency.py \
        --repo-root "$repo_dir" \
        --onnx "$onnx_path" \
        --engine "$engine_path" \
        --image "$probe_image" \
        --report-dir "$report_dir/$precision" \
        --precision "$precision" \
        --imgsz 640 \
        --confidence 0.25 \
        --nms-iou 0.70 \
        --max-abs-atol "$max_abs" \
        --mean-abs-atol "$mean_abs" \
        --relative-l2-atol "$relative_l2" \
        --box-atol "$box_atol" \
        --confidence-atol "$confidence_atol" \
        2>&1 | tee -a "$run_log"
    return ${PIPESTATUS[0]}
}

fp32_consistency_return_code=99
if [ "$fp32_build_return_code" -eq 0 ] && [ -s "$fp32_engine" ]; then
    run_consistency fp32 "$fp32_engine" 1e-3 2e-5 1e-5 1e-3 1e-5
    fp32_consistency_return_code=$?
fi
echo "$fp32_consistency_return_code" > "$report_dir/fp32_consistency_return_code.txt"

fp16_consistency_return_code=99
if [ "$fp16_build_return_code" -eq 0 ] && [ -s "$fp16_engine" ]; then
    run_consistency fp16 "$fp16_engine" 0.5 0.01 1e-3 0.5 1e-3
    fp16_consistency_return_code=$?
fi
echo "$fp16_consistency_return_code" > "$report_dir/fp16_consistency_return_code.txt"

run_benchmark() {
    local precision="$1"
    local engine_path="$2"
    "$trtexec_bin" \
        --loadEngine="$engine_path" \
        --warmUp=500 \
        --duration=0 \
        --iterations=200 \
        --useCudaGraph \
        --useSpinWait \
        --noDataTransfers \
        --percentile=50,95,99 \
        --exportTimes="$report_dir/${precision}_times.json" \
        2>&1 | tee "$report_dir/${precision}_benchmark.log" | tee -a "$run_log"
    return ${PIPESTATUS[0]}
}

fp32_benchmark_return_code=99
if [ "$fp32_consistency_return_code" -eq 0 ]; then
    run_benchmark fp32 "$fp32_engine"
    fp32_benchmark_return_code=$?
fi
echo "$fp32_benchmark_return_code" > "$report_dir/fp32_benchmark_return_code.txt"

fp16_benchmark_return_code=99
if [ "$fp16_consistency_return_code" -eq 0 ]; then
    run_benchmark fp16 "$fp16_engine"
    fp16_benchmark_return_code=$?
fi
echo "$fp16_benchmark_return_code" > "$report_dir/fp16_benchmark_return_code.txt"

"$python_bin" -u tools/exp07_collect_formal.py \
    --report-dir "$report_dir" \
    --fp32-engine "$fp32_engine" \
    --fp16-engine "$fp16_engine" \
    --fp32-build-return-code "$fp32_build_return_code" \
    --fp16-build-return-code "$fp16_build_return_code" \
    --fp32-build-seconds "$fp32_build_seconds" \
    --fp16-build-seconds "$fp16_build_seconds" \
    --fp32-consistency-return-code "$fp32_consistency_return_code" \
    --fp16-consistency-return-code "$fp16_consistency_return_code" \
    --fp32-benchmark-return-code "$fp32_benchmark_return_code" \
    --fp16-benchmark-return-code "$fp16_benchmark_return_code" \
    2>&1 | tee -a "$run_log"
collector_return_code=${PIPESTATUS[0]}

grep -nE \
    'Traceback|FATAL:|ERROR:|result=FAIL|raw_tensor_result=FAIL|detection_result=FAIL' \
    "$run_log" > "$abnormal_file"

if [ "$collector_return_code" -eq 0 ] && \
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
    echo "collector_return_code=$collector_return_code"
    echo "report_dir=$report_dir"
    echo "artifact_dir=$artifact_dir"
    echo "fp32_engine=$fp32_engine"
    echo "fp16_engine=$fp16_engine"
} > "$report_dir/runner_summary.txt"

if [ "$final_result" != "PASS" ]; then
    {
        echo "result=FAIL"
        echo "collector_return_code=$collector_return_code"
        echo "last_log_lines:"
        tr '\r' '\n' < "$run_log" | tail -n 160
    } > "$report_dir/failure_summary.txt"
    exit 1
fi

echo "exp07_1_trt_fp32_fp16_formal=PASS"
exit 0

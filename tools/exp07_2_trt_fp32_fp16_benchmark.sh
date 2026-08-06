#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
trtexec_bin="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
engine_root="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501"
fp32_engine="$engine_root/yolo11n_baseline_exp07_b1_640_fp32.engine"
fp16_engine="$engine_root/yolo11n_baseline_exp07_b1_640_fp16.engine"
expected_fp32_sha256="01616a8144228db5edbf8948227e3bbaee43b22c495aba3c6c44212e43efe0f1"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp07_2_trt_fp32_fp16_benchmark_${timestamp}"
report_dir="$repo_dir/results/tensorrt/$run_name"
if [ -e "$report_dir" ]; then
    echo "ERROR: report directory already exists: $report_dir"
    exit 1
fi
mkdir -p "$report_dir"
run_log="$report_dir/run.log"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    printf '%s\n' 1 > "$report_dir/return_code.txt"
    printf 'result=FAIL\nreason=%s\n' "$message" > "$report_dir/failure_summary.txt"
    exit 1
}

[ -x "$python_bin" ] || fail_early "Python not executable"
[ -x "$trtexec_bin" ] || fail_early "trtexec not executable"
[ -s "$fp32_engine" ] || fail_early "FP32 engine missing"
[ -s "$fp16_engine" ] || fail_early "FP16 engine missing"
[ "$(sha256sum "$fp32_engine" | cut -d ' ' -f 1)" = "$expected_fp32_sha256" ] || fail_early "FP32 SHA256 mismatch"
[ "$(sha256sum "$fp16_engine" | cut -d ' ' -f 1)" = "$expected_fp16_sha256" ] || fail_early "FP16 SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/07-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp07.2 TensorRT diagnostic benchmark"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "fp32_engine=$fp32_engine"
    echo "fp32_sha256=$expected_fp32_sha256"
    echo "fp16_engine=$fp16_engine"
    echo "fp16_sha256=$expected_fp16_sha256"
    echo "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END"
    echo "warmup_ms=500"
    echo "iterations=200"
    echo "duration_seconds=0"
    echo "cuda_graph=true"
    echo "spin_wait=true"
    echo "jetson_clocks=NOT_CHECKED_NON_ROOT"
    nvpmodel -q
    dpkg-query -W libnvinfer10 nvinfer-bin
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "$trtexec_bin --loadEngine=<engine> --warmUp=500 --duration=0 --iterations=200 --useCudaGraph --useSpinWait --noDataTransfers --percentile=50,95,99 --exportTimes=<json>" \
    > "$report_dir/command.txt"

run_benchmark() {
    local precision="$1"
    local engine="$2"
    "$trtexec_bin" \
        --loadEngine="$engine" \
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

run_benchmark fp32 "$fp32_engine"
fp32_return_code=$?
printf '%s\n' "$fp32_return_code" > "$report_dir/fp32_return_code.txt"

run_benchmark fp16 "$fp16_engine"
fp16_return_code=$?
printf '%s\n' "$fp16_return_code" > "$report_dir/fp16_return_code.txt"

"$python_bin" -u tools/exp07_collect_benchmark.py \
    --report-dir "$report_dir" \
    --fp32-engine "$fp32_engine" \
    --fp16-engine "$fp16_engine" \
    --fp32-return-code "$fp32_return_code" \
    --fp16-return-code "$fp16_return_code" \
    2>&1 | tee -a "$run_log"
collector_return_code=${PIPESTATUS[0]}

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL' "$run_log" > "$report_dir/abnormal.txt"
if [ "$fp32_return_code" -eq 0 ] && \
    [ "$fp16_return_code" -eq 0 ] && \
    [ "$collector_return_code" -eq 0 ] && \
    grep -q '^result=PASS$' "$report_dir/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi
printf 'result=%s\nfp32_return_code=%s\nfp16_return_code=%s\ncollector_return_code=%s\n' \
    "$final_result" "$fp32_return_code" "$fp16_return_code" "$collector_return_code" \
    > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"
if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
fi
exit "$final_code"

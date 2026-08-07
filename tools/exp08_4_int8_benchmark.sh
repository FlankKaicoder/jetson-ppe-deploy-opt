#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
trtexec_bin="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
int8_engine="/home/nvidia/models/jetson-ppe/exp08/exp08_2_int8_formal_20260807_153244/yolo11n_baseline_exp08_b1_640_int8.engine"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_int8_sha256="5787fb3bae4dbd00909c1762efc9263566044bc4dc35a836c950312e85895f26"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp08_4_int8_benchmark_${timestamp}"
report_dir="$repo_dir/results/int8/$run_name"

if [ -e "$report_dir" ]; then
    echo "ERROR: report directory already exists"
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

[ -x "$python_bin" ] || fail_early "Python missing"
[ -x "$trtexec_bin" ] || fail_early "trtexec missing"
[ -s "$fp16_engine" ] || fail_early "FP16 engine missing"
[ -s "$int8_engine" ] || fail_early "INT8 engine missing"
[ "$(sha256sum "$fp16_engine" | cut -d ' ' -f 1)" = "$expected_fp16_sha256" ] || fail_early "FP16 SHA256 mismatch"
[ "$(sha256sum "$int8_engine" | cut -d ' ' -f 1)" = "$expected_int8_sha256" ] || fail_early "INT8 SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/08-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp08.4 FP16 vs INT8 diagnostic benchmark"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "fp16_sha256=$expected_fp16_sha256"
    echo "int8_sha256=$expected_int8_sha256"
    echo "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END"
    echo "warmup_ms=500"
    echo "iterations=200"
    echo "duration_seconds=0"
    echo "cuda_graph=true"
    echo "spin_wait=true"
    echo "jetson_clocks=NOT_CHECKED_NON_ROOT"
    echo "min_latency_reduction=0.05"
    echo "min_size_reduction=0.10"
    nvpmodel -q
    dpkg-query -W libnvinfer10 nvinfer-bin
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "$trtexec_bin --loadEngine=<engine> --warmUp=500 --duration=0 --iterations=200 --useCudaGraph --useSpinWait --noDataTransfers --percentile=50,95,99 --exportTimes=<json>" \
    > "$report_dir/command.txt"

run_benchmark() {
    local name="$1"
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
        --exportTimes="$report_dir/${name}_times.json" \
        2>&1 | tee "$report_dir/${name}_benchmark.log" | tee -a "$run_log"
    return ${PIPESTATUS[0]}
}

run_benchmark fp16 "$fp16_engine"
fp16_return_code=$?
run_benchmark int8 "$int8_engine"
int8_return_code=$?
printf '%s\n' "$fp16_return_code" > "$report_dir/fp16_return_code.txt"
printf '%s\n' "$int8_return_code" > "$report_dir/int8_return_code.txt"

"$python_bin" -u tools/exp08_collect_benchmark.py \
    --report-dir "$report_dir" \
    --fp16-engine "$fp16_engine" \
    --int8-engine "$int8_engine" \
    --fp16-return-code "$fp16_return_code" \
    --int8-return-code "$int8_return_code" \
    --min-latency-reduction 0.05 \
    --min-size-reduction 0.10 \
    2>&1 | tee -a "$run_log"
collector_return_code=${PIPESTATUS[0]}
printf '%s\n' "$collector_return_code" > "$report_dir/collector_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$fp16_return_code" -eq 0 ] && \
    [ "$int8_return_code" -eq 0 ] && \
    [ "$collector_return_code" -eq 0 ] && \
    grep -q '^result=PASS$' "$report_dir/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi
printf 'result=%s\nfp16_return_code=%s\nint8_return_code=%s\ncollector_return_code=%s\nreport_dir=%s\n' \
    "$final_result" "$fp16_return_code" "$int8_return_code" "$collector_return_code" "$report_dir" \
    > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"
if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
fi
exit "$final_code"

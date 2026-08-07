#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
probe_image="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
expected_engine_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_image_sha256="39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp09_0_cpp_runtime_smoke_${timestamp}"
report_dir="$repo_dir/results/runtime/$run_name"
build_dir="$report_dir/build"
run_log="$report_dir/run.log"

mkdir -p "$report_dir"

fail_early() {
    local message="$1"
    echo "ERROR: $message" | tee -a "$run_log"
    printf 'result=FAIL\nreason=%s\n' "$message" > "$report_dir/failure_summary.txt"
    printf '%s\n' 1 > "$report_dir/return_code.txt"
    exit 1
}

[ -x "$python_bin" ] || fail_early "Python missing"
[ -s "$engine" ] || fail_early "FP16 Engine missing"
[ -s "$probe_image" ] || fail_early "probe image missing"
[ "$(sha256sum "$engine" | cut -d ' ' -f 1)" = "$expected_engine_sha256" ] || fail_early "Engine SHA256 mismatch"
[ "$(sha256sum "$probe_image" | cut -d ' ' -f 1)" = "$expected_image_sha256" ] || fail_early "probe SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/09-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp09.0 TensorRT C++ Runtime smoke"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "engine_sha256=$expected_engine_sha256"
    echo "probe_image_sha256=$expected_image_sha256"
    echo "warmup=2"
    echo "iterations=5"
    echo "timing_scope=H2D_ENQUEUE_D2H_SYNCHRONIZE"
    /usr/local/cuda/bin/nvcc --version | tail -1
    dpkg-query -W libnvinfer10 2>/dev/null
    cmake --version | head -1
    g++ --version | head -1
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "cmake -S runtime -B <build_dir> -DCMAKE_BUILD_TYPE=Release" \
    "cmake --build <build_dir> --parallel 2" \
    "python tools/exp09_prepare_reference.py --engine <engine> --image <image> --report-dir <report_dir>" \
    "exp09_trt_infer --engine <engine> --input <input.bin> --output <output.bin> --summary <runtime.txt> --warmup 2 --iterations 5" \
    "python tools/exp09_compare_outputs.py --reference <python.bin> --candidate <cxx.bin> --report-dir <comparison>" \
    > "$report_dir/command.txt"

cmake -S runtime -B "$build_dir" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "$run_log"
configure_code=${PIPESTATUS[0]}
printf '%s\n' "$configure_code" > "$report_dir/configure_return_code.txt"
if [ "$configure_code" -ne 0 ]; then
    fail_early "CMake configure failed"
fi

cmake --build "$build_dir" --parallel 2 2>&1 | tee -a "$run_log"
build_code=${PIPESTATUS[0]}
printf '%s\n' "$build_code" > "$report_dir/build_return_code.txt"
if [ "$build_code" -ne 0 ]; then
    fail_early "C++ build failed"
fi

"$python_bin" -u tools/exp09_prepare_reference.py \
    --engine "$engine" \
    --image "$probe_image" \
    --report-dir "$report_dir/reference" \
    2>&1 | tee -a "$run_log"
reference_code=${PIPESTATUS[0]}
printf '%s\n' "$reference_code" > "$report_dir/reference_return_code.txt"
if [ "$reference_code" -ne 0 ]; then
    fail_early "reference preparation failed"
fi

"$build_dir/exp09_trt_infer" \
    --engine "$engine" \
    --input "$report_dir/reference/input_fp32_nchw.bin" \
    --output "$report_dir/cxx_output_fp32.bin" \
    --summary "$report_dir/runtime_summary.txt" \
    --warmup 2 \
    --iterations 5 \
    2>&1 | tee -a "$run_log"
runtime_code=${PIPESTATUS[0]}
printf '%s\n' "$runtime_code" > "$report_dir/runtime_return_code.txt"
if [ "$runtime_code" -ne 0 ]; then
    fail_early "C++ runtime failed"
fi

"$python_bin" -u tools/exp09_compare_outputs.py \
    --reference "$report_dir/reference/python_trt_output_fp32.bin" \
    --candidate "$report_dir/cxx_output_fp32.bin" \
    --report-dir "$report_dir/comparison" \
    2>&1 | tee -a "$run_log"
compare_code=${PIPESTATUS[0]}
printf '%s\n' "$compare_code" > "$report_dir/compare_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$compare_code" -eq 0 ] && \
    grep -q '^result=PASS$' "$report_dir/runtime_summary.txt" && \
    grep -q '^result=PASS$' "$report_dir/comparison/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi
printf 'result=%s\nconfigure_return_code=%s\nbuild_return_code=%s\nreference_return_code=%s\nruntime_return_code=%s\ncompare_return_code=%s\nreport_dir=%s\n' \
    "$final_result" "$configure_code" "$build_code" "$reference_code" "$runtime_code" "$compare_code" "$report_dir" \
    > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"
if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
fi
exit "$final_code"

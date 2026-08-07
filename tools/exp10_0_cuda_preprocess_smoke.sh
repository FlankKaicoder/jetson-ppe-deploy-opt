#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
source_image="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
expected_image_sha256="39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp10_0_cuda_preprocess_smoke_${timestamp}"
report_dir="$repo_dir/results/cuda_preprocess/$run_name"
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
[ -s "$source_image" ] || fail_early "source image missing"
[ "$(sha256sum "$source_image" | cut -d ' ' -f 1)" = "$expected_image_sha256" ] || fail_early "source SHA256 mismatch"

cd "$repo_dir" || fail_early "cannot enter repository"
case "$(git branch --show-current)" in
    exp/10-*) ;;
    *) fail_early "unexpected Git branch" ;;
esac

{
    echo "experiment=Exp10.0 CUDA fused preprocess smoke"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "source_image_sha256=$expected_image_sha256"
    echo "fixture=hd_wide_720x1280"
    echo "warmup=2"
    echo "iterations=5"
    echo "timing_scope=CPU_vs_KERNEL_ONLY_vs_PAGEABLE_TRANSFERS"
    /usr/local/cuda/bin/nvcc --version | tail -1
    pkg-config --modversion opencv4
    cmake --version | head -1
    g++ --version | head -1
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "python tools/exp10_make_fixtures.py --source <image> --output-dir <fixtures>" \
    "cmake -S cuda -B <build> -DCMAKE_BUILD_TYPE=Release" \
    "cmake --build <build> --parallel 2" \
    "exp10_preprocess --image <hd_wide.png> --warmup 2 --iterations 5" \
    "python tools/exp10_compare_preprocess.py --cpu <cpu.bin> --cuda <cuda.bin>" \
    > "$report_dir/command.txt"

"$python_bin" -u tools/exp10_make_fixtures.py \
    --source "$source_image" \
    --output-dir "$report_dir/fixtures" \
    2>&1 | tee -a "$run_log"
fixture_code=${PIPESTATUS[0]}
printf '%s\n' "$fixture_code" > "$report_dir/fixture_return_code.txt"
if [ "$fixture_code" -ne 0 ]; then
    fail_early "fixture generation failed"
fi

cmake -S cuda -B "$build_dir" -DCMAKE_BUILD_TYPE=Release 2>&1 | tee -a "$run_log"
configure_code=${PIPESTATUS[0]}
printf '%s\n' "$configure_code" > "$report_dir/configure_return_code.txt"
if [ "$configure_code" -ne 0 ]; then
    fail_early "CMake configure failed"
fi

cmake --build "$build_dir" --parallel 2 2>&1 | tee -a "$run_log"
build_code=${PIPESTATUS[0]}
printf '%s\n' "$build_code" > "$report_dir/build_return_code.txt"
if [ "$build_code" -ne 0 ]; then
    fail_early "CUDA build failed"
fi

"$build_dir/exp10_preprocess" \
    --image "$report_dir/fixtures/hd_wide.png" \
    --cpu-output "$report_dir/cpu_output.bin" \
    --cuda-output "$report_dir/cuda_output.bin" \
    --summary "$report_dir/runtime_summary.txt" \
    --warmup 2 \
    --iterations 5 \
    2>&1 | tee -a "$run_log"
runtime_code=${PIPESTATUS[0]}
printf '%s\n' "$runtime_code" > "$report_dir/runtime_return_code.txt"
if [ "$runtime_code" -ne 0 ]; then
    fail_early "CUDA runtime failed"
fi

"$python_bin" -u tools/exp10_compare_preprocess.py \
    --cpu "$report_dir/cpu_output.bin" \
    --cuda "$report_dir/cuda_output.bin" \
    --runtime-summary "$report_dir/runtime_summary.txt" \
    --report-dir "$report_dir/comparison" \
    2>&1 | tee -a "$run_log"
compare_code=${PIPESTATUS[0]}
printf '%s\n' "$compare_code" > "$report_dir/compare_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$compare_code" -eq 0 ] && \
    grep -q '^result=PASS$' "$report_dir/comparison/summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi
printf 'result=%s\nfixture_return_code=%s\nconfigure_return_code=%s\nbuild_return_code=%s\nruntime_return_code=%s\ncompare_return_code=%s\nreport_dir=%s\n' \
    "$final_result" "$fixture_code" "$configure_code" "$build_code" "$runtime_code" "$compare_code" "$report_dir" \
    > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"
if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
fi
exit "$final_code"

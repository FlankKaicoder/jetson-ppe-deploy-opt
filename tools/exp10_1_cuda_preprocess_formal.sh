#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PPE_REPO_DIR:-$(cd "$script_dir/.." && pwd)}"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
source_image="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
expected_image_sha256="39a263dd6931e7ca70b85348cdd35c3fed9ca5c938c391023d438b24fbe8910e"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp10_1_cuda_preprocess_formal_${timestamp}"
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
    echo "experiment=Exp10.1 CUDA fused preprocess formal"
    echo "timestamp=$timestamp"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git branch --show-current)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "source_image_sha256=$expected_image_sha256"
    echo "correctness_fixtures=square,wide,tall,hd_wide,small_tall"
    echo "correctness_warmup=2"
    echo "correctness_iterations=5"
    echo "performance_fixture=hd_wide"
    echo "performance_warmup=20"
    echo "performance_iterations=200"
    echo "jetson_clocks=not_locked"
    echo "timing_scope=CPU_vs_KERNEL_ONLY_vs_PAGEABLE_TRANSFERS"
    /usr/local/cuda/bin/nvcc --version | tail -1
    pkg-config --modversion opencv4
    cmake --version | head -1
    g++ --version | head -1
} > "$report_dir/environment.txt" 2>&1

printf '%s\n' \
    "python tools/exp10_make_fixtures.py --source <image> --output-dir <fixtures>" \
    "cmake -S cuda -B <build> -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
    "cmake --build <build> --parallel 2" \
    "exp10_preprocess --image <fixture> --warmup <2|20> --iterations <5|200>" \
    "python tools/exp10_compare_preprocess.py --cpu <cpu.bin> --cuda <cuda.bin>" \
    "python tools/exp10_collect_formal.py --run-dir <report_dir>" \
    > "$report_dir/command.txt"

"$python_bin" -u tools/exp10_make_fixtures.py \
    --source "$source_image" --output-dir "$report_dir/fixtures" \
    2>&1 | tee -a "$run_log"
fixture_code=${PIPESTATUS[0]}
printf '%s\n' "$fixture_code" > "$report_dir/fixture_return_code.txt"
[ "$fixture_code" -eq 0 ] || fail_early "fixture generation failed"

cmake -S cuda -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    2>&1 | tee -a "$run_log"
configure_code=${PIPESTATUS[0]}
printf '%s\n' "$configure_code" > "$report_dir/configure_return_code.txt"
[ "$configure_code" -eq 0 ] || fail_early "CMake configure failed"

cmake --build "$build_dir" --parallel 2 2>&1 | tee -a "$run_log"
build_code=${PIPESTATUS[0]}
printf '%s\n' "$build_code" > "$report_dir/build_return_code.txt"
[ "$build_code" -eq 0 ] || fail_early "CUDA build failed"

step_failure=0
for fixture in square wide tall hd_wide small_tall; do
    warmup=2
    iterations=5
    if [ "$fixture" = "hd_wide" ]; then
        warmup=20
        iterations=200
    fi
    fixture_dir="$report_dir/fixture_results/$fixture"
    comparison_dir="$report_dir/comparison/$fixture"
    mkdir -p "$fixture_dir" "$comparison_dir"
    "$build_dir/exp10_preprocess" \
        --image "$report_dir/fixtures/$fixture.png" \
        --cpu-output "$fixture_dir/cpu_output.bin" \
        --cuda-output "$fixture_dir/cuda_output.bin" \
        --summary "$fixture_dir/runtime_summary.txt" \
        --warmup "$warmup" --iterations "$iterations" \
        2>&1 | tee -a "$run_log"
    runtime_code=${PIPESTATUS[0]}
    printf '%s\n' "$runtime_code" > "$fixture_dir/runtime_return_code.txt"
    if [ "$runtime_code" -ne 0 ]; then
        step_failure=1
        continue
    fi
    compare_args=(
        --cpu "$fixture_dir/cpu_output.bin"
        --cuda "$fixture_dir/cuda_output.bin"
        --runtime-summary "$fixture_dir/runtime_summary.txt"
        --report-dir "$comparison_dir"
    )
    if [ "$fixture" = "square" ]; then
        compare_args+=(--require-exact)
    fi
    "$python_bin" -u tools/exp10_compare_preprocess.py "${compare_args[@]}" \
        2>&1 | tee -a "$run_log"
    compare_code=${PIPESTATUS[0]}
    printf '%s\n' "$compare_code" > "$fixture_dir/compare_return_code.txt"
    if [ "$compare_code" -ne 0 ]; then
        step_failure=1
    fi
done

collector_code=1
if [ "$step_failure" -eq 0 ]; then
    "$python_bin" -u tools/exp10_collect_formal.py --run-dir "$report_dir" \
        2>&1 | tee -a "$run_log"
    collector_code=${PIPESTATUS[0]}
fi
printf '%s\n' "$collector_code" > "$report_dir/collector_return_code.txt"

grep -nE 'Traceback|FATAL:|ERROR:|result=FAIL|"result": "FAIL"' "$run_log" > "$report_dir/abnormal.txt"
if [ "$step_failure" -eq 0 ] && [ "$collector_code" -eq 0 ] && \
    grep -q '^result=PASS$' "$report_dir/formal_summary.txt" && \
    [ ! -s "$report_dir/abnormal.txt" ]; then
    final_result="PASS"
    final_code=0
else
    final_result="FAIL"
    final_code=1
fi
printf 'result=%s\nfixture_return_code=%s\nconfigure_return_code=%s\nbuild_return_code=%s\nstep_failure=%s\ncollector_return_code=%s\nreport_dir=%s\n' \
    "$final_result" "$fixture_code" "$configure_code" "$build_code" \
    "$step_failure" "$collector_code" "$report_dir" \
    > "$report_dir/runner_summary.txt"
printf '%s\n' "$final_code" > "$report_dir/return_code.txt"
if [ "$final_code" -ne 0 ]; then
    cp "$report_dir/runner_summary.txt" "$report_dir/failure_summary.txt"
fi
exit "$final_code"

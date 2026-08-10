#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
trtexec_bin="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
qdq_engine="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.engine"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_qdq_sha256="43db95c68e9dd23d00b2c35e0cfe19a9d61ca75a1a92ffbf70245f530ceb66c9"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_4_qdq_benchmark_${timestamp}}"

fail() {
    local reason="$1"
    printf 'result=FAIL\nreason=%s\n' "${reason}" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    echo "ERROR: ${reason}" >&2
    exit 1
}

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}"
[[ -x "${python_bin}" ]] || fail "Python missing: ${python_bin}"
[[ -x "${trtexec_bin}" ]] || fail "trtexec missing: ${trtexec_bin}"
[[ -s "${fp16_engine}" ]] || fail "FP16 Engine missing"
[[ -s "${qdq_engine}" ]] || fail "QDQ Engine missing"
[[ "$(sha256sum "${fp16_engine}" | cut -d ' ' -f 1)" == "${expected_fp16_sha256}" ]] || fail "FP16 hash mismatch"
[[ "$(sha256sum "${qdq_engine}" | cut -d ' ' -f 1)" == "${expected_qdq_sha256}" ]] || fail "QDQ hash mismatch"
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected Git branch"

{
    echo "experiment=Exp17 R4 paired/interleaved GPU-only diagnostic"
    echo "timestamp=${timestamp}"
    echo "hostname=$(hostname)"
    echo "git_branch=$(git -C "${repo_root}" branch --show-current)"
    echo "git_commit=$(git -C "${repo_root}" rev-parse HEAD)"
    echo "scope=GPU_COMPUTE_ONLY_NO_H2D_D2H_NOT_END_TO_END"
    echo "order=R1 FP16->QDQ; R2 QDQ->FP16; R3 FP16->QDQ"
    echo "warmup_ms=500 iterations=200 duration_seconds=0"
    echo "cuda_graph=true spin_wait=true data_transfers=false"
    echo "clock_policy=dynamic_25W_paired_interleaved"
    sha256sum "${fp16_engine}" "${qdq_engine}"
    nvpmodel -q
    dpkg-query -W libnvinfer10 nvinfer-bin
    cat /sys/devices/virtual/thermal/thermal_zone0/temp 2>/dev/null || true
} > "${report_dir}/environment.txt" 2>&1
printf '%s\n' \
    "${trtexec_bin} --loadEngine=<engine> --warmUp=500 --duration=0 --iterations=200 --useCudaGraph --useSpinWait --noDataTransfers --percentile=50,95,99 --exportTimes=<json>" \
    > "${report_dir}/command.txt"

declare -A return_codes
run_one() {
    local name="$1"
    local engine="$2"
    set +e
    "${trtexec_bin}" --loadEngine="${engine}" --warmUp=500 --duration=0 \
        --iterations=200 --useCudaGraph --useSpinWait --noDataTransfers \
        --percentile=50,95,99 --exportTimes="${report_dir}/${name}_times.json" \
        > "${report_dir}/${name}.log" 2>&1
    local rc=$?
    set -e
    return_codes["${name}"]="${rc}"
}

set -e
run_one r1_1_fp16 "${fp16_engine}"
run_one r1_2_qdq "${qdq_engine}"
run_one r2_1_qdq "${qdq_engine}"
run_one r2_2_fp16 "${fp16_engine}"
run_one r3_1_fp16 "${fp16_engine}"
run_one r3_2_qdq "${qdq_engine}"

{
    echo "{"
    echo '  "r1_1_fp16": '"${return_codes[r1_1_fp16]},"
    echo '  "r1_2_qdq": '"${return_codes[r1_2_qdq]},"
    echo '  "r2_1_qdq": '"${return_codes[r2_1_qdq]},"
    echo '  "r2_2_fp16": '"${return_codes[r2_2_fp16]},"
    echo '  "r3_1_fp16": '"${return_codes[r3_1_fp16]},"
    echo '  "r3_2_qdq": '"${return_codes[r3_2_qdq]}"
    echo "}"
} > "${report_dir}/return_codes.json"

set +e
"${python_bin}" "${repo_root}/tools/exp17_collect_benchmark.py" \
    --report-dir "${report_dir}" --fp16-engine "${fp16_engine}" \
    --qdq-engine "${qdq_engine}" --return-codes "${report_dir}/return_codes.json" \
    --min-latency-reduction 0.05 --min-size-reduction 0.10 --min-favorable-pairs 2 \
    > "${report_dir}/collector.log" 2>&1
collector_rc=$?
set -e
printf '%s\n' "${collector_rc}" > "${report_dir}/collector_return_code.txt"
if [[ "${collector_rc}" -eq 0 ]]; then
    printf '0\n' > "${report_dir}/return_code.txt"
    echo "exp17_4_qdq_benchmark=PASS output=${report_dir}"
    exit 0
fi
[[ -s "${report_dir}/summary.json" ]] || fail "collector failed without summary"
printf '%s\n' "${collector_rc}" > "${report_dir}/return_code.txt"
cp "${report_dir}/summary.txt" "${report_dir}/rejection_summary.txt"
echo "exp17_4_qdq_benchmark=REJECTED output=${report_dir}"
exit "${collector_rc}"

#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
trtexec="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
artifact="/home/nvidia/models/jetson-ppe/exp17/exp17_8_mixed_build_20260810_152337"
fp16="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
full_qdq="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.engine"
p3_classification="${artifact}/yolo11n_exp17_mixed_p3_classification.engine"
classification="${artifact}/yolo11n_exp17_mixed_classification.engine"
dfl="${artifact}/yolo11n_exp17_mixed_dfl.engine"
detect_head="${artifact}/yolo11n_exp17_mixed_detect_head.engine"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_9_mixed_perf_screen_${timestamp}}"

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}"
fail() { printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"; printf '1\n' > "${report_dir}/return_code.txt"; exit 1; }
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
for name in fp16 full_qdq p3_classification classification dfl detect_head; do
    path="${!name}"
    [[ -s "${path}" ]] || fail "missing Engine: ${name}"
done
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    nvpmodel -q
    for name in fp16 full_qdq p3_classification classification dfl detect_head; do sha256sum "${!name}"; done
    echo "scope=GPU_ONLY_SCREEN_NOT_FINAL_ADOPTION"
    echo "r1=fp16,full_qdq,p3_classification,classification,dfl,detect_head"
    echo "r2=detect_head,dfl,classification,p3_classification,full_qdq,fp16"
} > "${report_dir}/environment.txt" 2>&1

run_one() {
    local run_name="$1"
    local engine="$2"
    set +e
    "${trtexec}" --loadEngine="${engine}" --warmUp=500 --duration=0 --iterations=200 \
        --useCudaGraph --useSpinWait --noDataTransfers --percentile=50,95,99 \
        --exportTimes="${report_dir}/${run_name}_times.json" > "${report_dir}/${run_name}.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${report_dir}/${run_name}_return_code.txt"
    [[ "${rc}" -eq 0 ]] || fail "trtexec failed: ${run_name}"
}

set -e
order1=(fp16 full_qdq p3_classification classification dfl detect_head)
order2=(detect_head dfl classification p3_classification full_qdq fp16)
sequence=0
for name in "${order1[@]}"; do sequence=$((sequence + 1)); run_one "r1_${sequence}_${name}" "${!name}"; done
sequence=0
for name in "${order2[@]}"; do sequence=$((sequence + 1)); run_one "r2_${sequence}_${name}" "${!name}"; done
/usr/bin/python3 "${repo_root}/tools/exp17_collect_mixed_screen.py" --report-dir "${report_dir}" \
    > "${report_dir}/collector.log" 2>&1 || fail "collector failed"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp17_9_mixed_perf_screen=PASS output=${report_dir}"

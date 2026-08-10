#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fp16="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
implicit="/home/nvidia/models/jetson-ppe/exp08/exp08_2_int8_formal_20260807_153244/yolo11n_baseline_exp08_b1_640_int8.engine"
qdq="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.engine"
images="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/images/test"
limit="${PPE_LIMIT:-8}"
stage="smoke"
[[ "${limit}" -eq 219 ]] && stage="formal"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_7_raw_scale_${stage}_${timestamp}}"

[[ ! -e "${output_dir}" ]] || { echo "ERROR: output exists: ${output_dir}" >&2; exit 1; }
mkdir -p "${output_dir}"
fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${output_dir}/failure_summary.txt"
    printf '1\n' > "${output_dir}/return_code.txt"
    exit 1
}
[[ "${limit}" =~ ^[0-9]+$ ]] && [[ "${limit}" -gt 0 ]] && [[ "${limit}" -le 219 ]] || fail "invalid limit"
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
for input in "${fp16}" "${implicit}" "${qdq}"; do [[ -s "${input}" ]] || fail "missing Engine: ${input}"; done
[[ "$(sha256sum "${fp16}" | cut -d ' ' -f 1)" == "88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83" ]] || fail "FP16 hash mismatch"
[[ "$(sha256sum "${implicit}" | cut -d ' ' -f 1)" == "5787fb3bae4dbd00909c1762efc9263566044bc4dc35a836c950312e85895f26" ]] || fail "implicit INT8 hash mismatch"
[[ "$(sha256sum "${qdq}" | cut -d ' ' -f 1)" == "43db95c68e9dd23d00b2c35e0cfe19a9d61ca75a1a92ffbf70245f530ceb66c9" ]] || fail "QDQ hash mismatch"
[[ "$(find "${images}" -maxdepth 1 -type f | wc -l)" -eq 219 ]] || fail "test image count mismatch"
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    sha256sum "${fp16}" "${implicit}" "${qdq}"
    echo "stage=${stage} images=${limit} confidence=0.25"
} > "${output_dir}/environment.txt" 2>&1
set +e
/usr/bin/python3 "${repo_root}/tools/exp17_compare_raw_scales.py" \
    --fp16-engine "${fp16}" --implicit-engine "${implicit}" --qdq-engine "${qdq}" \
    --images "${images}" --output-dir "${output_dir}/audit" --limit "${limit}" \
    --confidence 0.25 > "${output_dir}/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${output_dir}/audit_return_code.txt"
if [[ "${rc}" -eq 0 ]]; then
    cp "${output_dir}/audit/summary.json" "${output_dir}/summary.json"
    cp "${output_dir}/audit/summary.txt" "${output_dir}/summary.txt"
    printf '0\n' > "${output_dir}/return_code.txt"
    echo "exp17_7_raw_scale_${stage}=PASS output=${output_dir}"
    exit 0
fi
fail "raw-scale audit failed with ${rc}"

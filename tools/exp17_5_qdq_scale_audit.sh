#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
onnx_path="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.onnx"
expected_sha256="5a28c30b0f92db1a94be7f290a781ff182df757fb71e36d749a5b64d1daf8325"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_5_qdq_scale_audit_${timestamp}}"

[[ ! -e "${output_dir}" ]] || { echo "ERROR: output exists: ${output_dir}" >&2; exit 1; }
mkdir -p "${output_dir}"
fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${output_dir}/failure_summary.txt"
    printf '1\n' > "${output_dir}/return_code.txt"
    exit 1
}
[[ -s "${onnx_path}" ]] || fail "QDQ ONNX missing"
[[ "$(sha256sum "${onnx_path}" | cut -d ' ' -f 1)" == "${expected_sha256}" ]] || fail "QDQ ONNX hash mismatch"
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    sha256sum "${onnx_path}"
} > "${output_dir}/environment.txt" 2>&1
set +e
/usr/bin/python3 "${repo_root}/tools/exp17_audit_qdq_scales.py" \
    --onnx "${onnx_path}" --output-dir "${output_dir}/audit" \
    > "${output_dir}/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${output_dir}/audit_return_code.txt"
if [[ "${rc}" -eq 0 ]]; then
    cp "${output_dir}/audit/summary.json" "${output_dir}/summary.json"
    cp "${output_dir}/audit/summary.txt" "${output_dir}/summary.txt"
    printf '0\n' > "${output_dir}/return_code.txt"
    echo "exp17_5_qdq_scale_audit=PASS output=${output_dir}"
    exit 0
fi
fail "static QDQ scale audit failed with ${rc}"

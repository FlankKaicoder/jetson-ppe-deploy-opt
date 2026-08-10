#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fp32_onnx="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
qdq_onnx="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.onnx"
manifest="/home/nvidia/models/jetson-ppe/exp08/calibration_20260807_145356/calibration/calibration_manifest.json"
builder="${repo_root}/tools/exp08_build_int8.py"
limit="${PPE_LIMIT:-8}"
stage="smoke"
[[ "${limit}" -eq 256 ]] && stage="formal"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_6_head_activation_${stage}_${timestamp}}"

[[ ! -e "${output_dir}" ]] || { echo "ERROR: output exists: ${output_dir}" >&2; exit 1; }
mkdir -p "${output_dir}"
fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${output_dir}/failure_summary.txt"
    printf '1\n' > "${output_dir}/return_code.txt"
    exit 1
}
[[ "${limit}" =~ ^[0-9]+$ ]] && [[ "${limit}" -gt 0 ]] && [[ "${limit}" -le 256 ]] || fail "invalid limit"
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
for input in "${fp32_onnx}" "${qdq_onnx}" "${manifest}" "${builder}"; do
    [[ -s "${input}" ]] || fail "missing input: ${input}"
done
[[ "$(sha256sum "${fp32_onnx}" | cut -d ' ' -f 1)" == "305e23c65aa3b1d01e7b1a784c355665228f435f0b904b92ed2618954736d1f8" ]] || fail "FP32 ONNX hash mismatch"
[[ "$(sha256sum "${qdq_onnx}" | cut -d ' ' -f 1)" == "5a28c30b0f92db1a94be7f290a781ff182df757fb71e36d749a5b64d1daf8325" ]] || fail "QDQ ONNX hash mismatch"
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    sha256sum "${fp32_onnx}" "${qdq_onnx}" "${manifest}"
    echo "stage=${stage} limit=${limit} sample_per_image_per_tensor=2048"
} > "${output_dir}/environment.txt" 2>&1
set +e
/usr/bin/python3 "${repo_root}/tools/exp17_audit_head_activations.py" \
    --fp32-onnx "${fp32_onnx}" --qdq-onnx "${qdq_onnx}" \
    --manifest "${manifest}" --exp08-builder "${builder}" \
    --output-dir "${output_dir}/audit" --limit "${limit}" --sample-per-image 2048 \
    > "${output_dir}/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${output_dir}/audit_return_code.txt"
if [[ "${rc}" -eq 0 ]]; then
    cp "${output_dir}/audit/summary.json" "${output_dir}/summary.json"
    cp "${output_dir}/audit/summary.txt" "${output_dir}/summary.txt"
    printf '0\n' > "${output_dir}/return_code.txt"
    echo "exp17_6_head_activation_${stage}=PASS output=${output_dir}"
    exit 0
fi
fail "head activation audit failed with ${rc}"

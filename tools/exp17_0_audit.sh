#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_0_audit_${timestamp}}"
python_path="${PPE_EXP17_PYTHON:-python3}"
python_packages="${PPE_EXP17_PYTHONPATH:-/home/nvidia/.local/jetson-ppe-exp16-py}"
onnx_path="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
int8_root="/home/nvidia/models/jetson-ppe/exp08/exp08_2_int8_formal_20260807_153244"
int8_engine="${int8_root}/yolo11n_baseline_exp08_b1_640_int8.engine"
calibration_cache="${int8_root}/yolo11n_baseline_exp08_int8.cache"
calibration_manifest="/home/nvidia/models/jetson-ppe/exp08/calibration_20260807_145356/calibration/calibration_manifest.json"

fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}"
for input in "${onnx_path}" "${fp16_engine}" "${int8_engine}" \
             "${calibration_cache}" "${calibration_manifest}" \
             "${repo_root}/tools/exp08_build_int8.py"; do
    [[ -s "${input}" ]] || fail "missing or empty input: ${input}"
done

{
    hostname
    whoami
    pwd
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    /usr/local/cuda/bin/nvcc --version
    dpkg-query -W libnvinfer10
    sha256sum "${onnx_path}" "${fp16_engine}" "${int8_engine}" \
        "${calibration_cache}" "${calibration_manifest}"
} > "${report_dir}/environment.txt" 2>&1

printf '%q ' env "PYTHONPATH=${python_packages}" "${python_path}" \
    "${repo_root}/tools/exp17_audit_quantization.py" \
    --onnx "${onnx_path}" \
    --exp08-builder "${repo_root}/tools/exp08_build_int8.py" \
    --calibration-cache "${calibration_cache}" \
    --calibration-manifest "${calibration_manifest}" \
    --fp16-engine "${fp16_engine}" \
    --int8-engine "${int8_engine}" \
    --output-dir "${report_dir}" --expect-mode implicit \
    > "${report_dir}/command.txt"
printf '\n' >> "${report_dir}/command.txt"

set +e
PYTHONPATH="${python_packages}" "${python_path}" \
    "${repo_root}/tools/exp17_audit_quantization.py" \
    --onnx "${onnx_path}" \
    --exp08-builder "${repo_root}/tools/exp08_build_int8.py" \
    --calibration-cache "${calibration_cache}" \
    --calibration-manifest "${calibration_manifest}" \
    --fp16-engine "${fp16_engine}" \
    --int8-engine "${int8_engine}" \
    --output-dir "${report_dir}" --expect-mode implicit \
    > "${report_dir}/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${report_dir}/return_code.txt"
[[ "${rc}" -eq 0 ]] || fail "quantization audit failed with ${rc}"
[[ -s "${report_dir}/summary.json" ]] || fail "summary.json missing"
[[ -s "${report_dir}/summary.txt" ]] || fail "summary.txt missing"

echo "exp17_0_audit=PASS output=${report_dir}"

#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_onnx="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138/yolo11n_exp17_qdq_full.onnx"
manifest="/home/nvidia/models/jetson-ppe/exp08/calibration_20260807_145356/calibration/calibration_manifest.json"
probe="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
trtexec="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp17_8_mixed_build_${timestamp}"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/${run_name}}"
artifact_dir="${PPE_ARTIFACT_DIR:-/home/nvidia/models/jetson-ppe/exp17/${run_name}}"
candidates=(p3_classification classification dfl detect_head)

[[ ! -e "${report_dir}" ]] || { echo "ERROR: report exists: ${report_dir}" >&2; exit 1; }
[[ ! -e "${artifact_dir}" ]] || { echo "ERROR: artifact exists: ${artifact_dir}" >&2; exit 1; }
mkdir -p "${report_dir}"
fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
[[ "$(sha256sum "${source_onnx}" | cut -d ' ' -f 1)" == "5a28c30b0f92db1a94be7f290a781ff182df757fb71e36d749a5b64d1daf8325" ]] || fail "source QDQ hash mismatch"
for input in "${manifest}" "${probe}" "${trtexec}"; do [[ -e "${input}" ]] || fail "missing input: ${input}"; done
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    /usr/local/cuda/bin/nvcc --version
    dpkg-query -W libnvinfer10
    sha256sum "${source_onnx}" "${manifest}" "${probe}"
    echo "candidates=${candidates[*]}"
    echo "build=stronglyTyped,noTF32,workspace1024MiB,optimizationLevel3"
} > "${report_dir}/environment.txt" 2>&1

set +e
/usr/bin/python3 "${repo_root}/tools/exp17_make_mixed_candidates.py" \
    --qdq-onnx "${source_onnx}" --output-dir "${artifact_dir}" \
    > "${report_dir}/generate.log" 2>&1
generate_rc=$?
set -e
printf '%s\n' "${generate_rc}" > "${report_dir}/generate_return_code.txt"
[[ "${generate_rc}" -eq 0 ]] || fail "mixed candidate generation failed with ${generate_rc}"
cp "${artifact_dir}/summary.json" "${report_dir}/graph_summary.json"
cp "${artifact_dir}/summary.txt" "${report_dir}/graph_summary.txt"

declare -A build_codes smoke_codes
for name in "${candidates[@]}"; do
    onnx_path="${artifact_dir}/yolo11n_exp17_mixed_${name}.onnx"
    engine_path="${artifact_dir}/yolo11n_exp17_mixed_${name}.engine"
    [[ -s "${onnx_path}" ]] || fail "candidate ONNX missing: ${name}"
    start="$(date +%s)"
    set +e
    "${trtexec}" --onnx="${onnx_path}" --saveEngine="${engine_path}" \
        --stronglyTyped --noTF32 --memPoolSize=workspace:1024 \
        --builderOptimizationLevel=3 --profilingVerbosity=detailed --skipInference \
        > "${report_dir}/${name}_build.log" 2>&1
    build_codes["${name}"]=$?
    set -e
    printf '%s\n' "$(( $(date +%s) - start ))" > "${report_dir}/${name}_build_seconds.txt"
    printf '%s\n' "${build_codes[${name}]}" > "${report_dir}/${name}_build_return_code.txt"
    [[ "${build_codes[${name}]}" -eq 0 ]] || fail "TensorRT build failed: ${name}"
    set +e
    /usr/bin/python3 "${repo_root}/tools/exp08_int8_smoke.py" \
        --onnx "${onnx_path}" --engine "${engine_path}" --image "${probe}" \
        --report-dir "${report_dir}/${name}_smoke" --imgsz 640 \
        --confidence 0.25 --nms-iou 0.70 \
        > "${report_dir}/${name}_smoke.log" 2>&1
    smoke_codes["${name}"]=$?
    set -e
    printf '%s\n' "${smoke_codes[${name}]}" > "${report_dir}/${name}_smoke_return_code.txt"
    [[ "${smoke_codes[${name}]}" -eq 0 ]] || fail "execution smoke failed: ${name}"
    sha256sum "${onnx_path}" "${engine_path}" > "${report_dir}/${name}_sha256.txt"
    echo "candidate=${name} build=PASS smoke=PASS" | tee -a "${report_dir}/progress.log"
done
{
    echo "result=PASS"
    for name in "${candidates[@]}"; do
        echo "${name}_build_seconds=$(cat "${report_dir}/${name}_build_seconds.txt")"
        stat -c "${name}_engine_bytes=%s" "${artifact_dir}/yolo11n_exp17_mixed_${name}.engine"
        cat "${report_dir}/${name}_sha256.txt"
    done
} > "${report_dir}/summary.txt"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp17_8_mixed_build=PASS output=${report_dir} artifacts=${artifact_dir}"

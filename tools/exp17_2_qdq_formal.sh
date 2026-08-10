#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="exp17_2_qdq_formal_${timestamp}"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/${run_name}}"
artifact_dir="${PPE_ARTIFACT_DIR:-/home/nvidia/models/jetson-ppe/exp17/${run_name}}"
onnx_path="/home/nvidia/models/jetson-ppe/exp06/yolo11n_baseline_exp06_b1_640_opset17.onnx"
manifest="/home/nvidia/models/jetson-ppe/exp08/calibration_20260807_145356/calibration/calibration_manifest.json"
probe_image="/home/nvidia/models/jetson-ppe/exp06/probe_train__image784.jpg"
trtexec="${PPE_TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
qdq_onnx="${artifact_dir}/yolo11n_exp17_qdq_full.onnx"
engine="${artifact_dir}/yolo11n_exp17_qdq_full.engine"

fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
[[ ! -e "${artifact_dir}" ]] || { echo "ERROR: artifact dir exists: ${artifact_dir}" >&2; exit 1; }
mkdir -p "${report_dir}" "${artifact_dir}"
for input in "${onnx_path}" "${manifest}" "${probe_image}" \
             "${repo_root}/tools/exp08_build_int8.py" \
             "${repo_root}/tools/exp17_make_qdq.py"; do
    [[ -s "${input}" ]] || fail "missing or empty input: ${input}"
done
[[ -x "${trtexec}" ]] || fail "trtexec missing: ${trtexec}"

{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    /usr/local/cuda/bin/nvcc --version
    dpkg-query -W libnvinfer10
    sha256sum "${onnx_path}" "${manifest}" "${probe_image}"
    echo "calibration_images=256"
    echo "calibration_method=Entropy"
    echo "quant_format=QDQ"
    echo "op_types=Conv,MatMul,Softmax"
    echo "types=activation_QInt8_weight_QInt8"
    echo "weight_granularity=per_channel"
    echo "bias=FP32_not_quantized"
    echo "calibration_chunk_size=8"
    echo "exclude_output_quantization=Conv,MatMul"
    echo "network=strongly_typed"
} > "${report_dir}/environment.txt" 2>&1

quantize_start="$(date +%s)"
set +e
python3 "${repo_root}/tools/exp17_make_qdq.py" \
    --onnx "${onnx_path}" --manifest "${manifest}" \
    --exp08-builder "${repo_root}/tools/exp08_build_int8.py" \
    --output-onnx "${qdq_onnx}" --report-dir "${report_dir}" \
    --limit 256 --calibration-chunk-size 8 --calibration-method entropy \
    > "${report_dir}/quantize.log" 2>&1
quantize_rc=$?
set -e
quantize_end="$(date +%s)"
printf '%s\n' "${quantize_rc}" > "${report_dir}/quantize_return_code.txt"
printf '%s\n' "$((quantize_end - quantize_start))" > "${report_dir}/quantize_wall_seconds.txt"
[[ "${quantize_rc}" -eq 0 ]] || fail "formal QDQ quantization failed with ${quantize_rc}"
[[ -s "${qdq_onnx}" ]] || fail "formal QDQ ONNX missing"

build_start="$(date +%s)"
set +e
"${trtexec}" \
    --onnx="${qdq_onnx}" \
    --saveEngine="${engine}" \
    --stronglyTyped \
    --noTF32 \
    --memPoolSize=workspace:1024 \
    --builderOptimizationLevel=3 \
    --profilingVerbosity=detailed \
    --skipInference \
    --verbose \
    > "${report_dir}/build.log" 2>&1
build_rc=$?
set -e
build_end="$(date +%s)"
printf '%s\n' "${build_rc}" > "${report_dir}/build_return_code.txt"
printf '%s\n' "$((build_end - build_start))" > "${report_dir}/build_seconds.txt"
[[ "${build_rc}" -eq 0 ]] || fail "formal TensorRT QDQ build failed with ${build_rc}"
[[ -s "${engine}" ]] || fail "formal QDQ Engine missing"

set +e
python3 "${repo_root}/tools/exp08_int8_smoke.py" \
    --onnx "${qdq_onnx}" --engine "${engine}" --image "${probe_image}" \
    --report-dir "${report_dir}/execution" --imgsz 640 \
    --confidence 0.25 --nms-iou 0.70 \
    > "${report_dir}/execution.log" 2>&1
execution_rc=$?
set -e
printf '%s\n' "${execution_rc}" > "${report_dir}/execution_return_code.txt"
[[ "${execution_rc}" -eq 0 ]] || fail "formal QDQ execution smoke failed with ${execution_rc}"

sha256sum "${qdq_onnx}" "${engine}" > "${report_dir}/artifact_sha256.txt"
{
    echo "result=PASS"
    echo "quantize_return_code=${quantize_rc}"
    echo "build_return_code=${build_rc}"
    echo "execution_return_code=${execution_rc}"
    echo "qdq_onnx=${qdq_onnx}"
    echo "engine=${engine}"
} > "${report_dir}/summary.txt"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp17_2_qdq_formal=PASS output=${report_dir} artifact=${artifact_dir}"

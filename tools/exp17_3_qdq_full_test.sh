#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PPE_JETSON_PYTHON:-/usr/bin/python3}"
fp16_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
qdq_root="/home/nvidia/models/jetson-ppe/exp17/exp17_2_qdq_formal_20260809_224138"
qdq_engine="${qdq_root}/yolo11n_exp17_qdq_full.engine"
qdq_onnx="${qdq_root}/yolo11n_exp17_qdq_full.onnx"
data_yaml="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/construction_ppe3_jetson_test.yaml"
dataset_archive="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_split.tar.gz"
expected_fp16_sha256="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
expected_qdq_engine_sha256="43db95c68e9dd23d00b2c35e0cfe19a9d61ca75a1a92ffbf70245f530ceb66c9"
expected_qdq_onnx_sha256="5a28c30b0f92db1a94be7f290a781ff182df757fb71e36d749a5b64d1daf8325"
expected_dataset_sha256="3bf3addcb79e7ac46163f7a294265a92c5f84c7e633f56da0b16e22f33400f4a"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_3_qdq_full_test_${timestamp}}"

fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}/backend_metrics" "${report_dir}/val_runs"
for input in "${fp16_engine}" "${qdq_engine}" "${qdq_onnx}" \
             "${data_yaml}" "${dataset_archive}"; do
    [[ -s "${input}" ]] || fail "missing or empty input: ${input}"
done
[[ "$(sha256sum "${fp16_engine}" | cut -d ' ' -f 1)" == "${expected_fp16_sha256}" ]] || fail "FP16 hash mismatch"
[[ "$(sha256sum "${qdq_engine}" | cut -d ' ' -f 1)" == "${expected_qdq_engine_sha256}" ]] || fail "QDQ Engine hash mismatch"
[[ "$(sha256sum "${qdq_onnx}" | cut -d ' ' -f 1)" == "${expected_qdq_onnx_sha256}" ]] || fail "QDQ ONNX hash mismatch"
[[ "$(sha256sum "${dataset_archive}" | cut -d ' ' -f 1)" == "${expected_dataset_sha256}" ]] || fail "dataset hash mismatch"
[[ "$(find "$(dirname "${data_yaml}")/images/test" -maxdepth 1 -type f | wc -l)" -eq 219 ]] || fail "test image count mismatch"

{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    sha256sum "${fp16_engine}" "${qdq_engine}" "${qdq_onnx}" "${dataset_archive}"
    echo "test_images=219"
    echo "test_instances=840"
    echo "imgsz=640 batch=1 rect=false"
    echo "conf=0.25 nms_iou=0.70 match_iou=0.50"
} > "${report_dir}/environment.txt" 2>&1

run_backend() {
    local name="$1"
    local engine="$2"
    set +e
    YOLO_AUTOINSTALL=false "${python_bin}" -u "${repo_root}/tools/exp07_eval_backend.py" \
        --model "${engine}" --data "${data_yaml}" \
        --output "${report_dir}/backend_metrics/${name}.json" \
        --project "${report_dir}/val_runs" --name "${name}" --device 0 \
        --imgsz 640 --batch 1 --workers 2 \
        > "${report_dir}/${name}_backend.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${report_dir}/${name}_backend_return_code.txt"
    [[ "${rc}" -eq 0 ]] || fail "${name} backend evaluation failed with ${rc}"
}

run_scale() {
    local name="$1"
    local engine="$2"
    set +e
    YOLO_AUTOINSTALL=false "${python_bin}" -u "${repo_root}/tools/exp02_8_baseline_error_size_audit.py" \
        --weights "${engine}" --data "${data_yaml}" \
        --output-dir "${report_dir}/${name}_scale" --imgsz 640 --batch 1 \
        --conf 0.25 --nms-iou 0.70 --match-iou 0.50 --max-visuals 0 \
        --serial-static-engine \
        > "${report_dir}/${name}_scale.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${report_dir}/${name}_scale_return_code.txt"
    [[ "${rc}" -eq 0 ]] || fail "${name} scale audit failed with ${rc}"
}

set -e
run_backend fp16 "${fp16_engine}"
run_backend explicit_qdq "${qdq_engine}"
run_scale fp16 "${fp16_engine}"
run_scale explicit_qdq "${qdq_engine}"

set +e
"${python_bin}" "${repo_root}/tools/exp17_collect_full_test.py" \
    --report-dir "${report_dir}" \
    --fp16-metrics "${report_dir}/backend_metrics/fp16.json" \
    --qdq-metrics "${report_dir}/backend_metrics/explicit_qdq.json" \
    --fp16-scale "${report_dir}/fp16_scale/summary.json" \
    --qdq-scale "${report_dir}/explicit_qdq_scale/summary.json" \
    --fp16-engine "${fp16_engine}" --qdq-engine "${qdq_engine}" \
    --map50-95-max-drop 0.010 --map50-max-drop 0.015 \
    --tiny-small-max-drop 0.050 \
    > "${report_dir}/collector.log" 2>&1
collector_rc=$?
set -e
printf '%s\n' "${collector_rc}" > "${report_dir}/collector_return_code.txt"
if [[ "${collector_rc}" -eq 0 ]]; then
    printf '0\n' > "${report_dir}/return_code.txt"
    echo "exp17_3_qdq_full_test=PASS output=${report_dir}"
    exit 0
fi
[[ -s "${report_dir}/summary.json" ]] || fail "collector failed without summary"
printf '1\n' > "${report_dir}/return_code.txt"
echo "exp17_3_qdq_full_test=REJECTED output=${report_dir}"
exit 1

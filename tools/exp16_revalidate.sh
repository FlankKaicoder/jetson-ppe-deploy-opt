#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/plugin/exp16_revalidate_r3_${timestamp}}"
build_root="${PPE_BUILD_ROOT:-${repo_root}/results/plugin/exp16_revalidate_build_matrix_20260809_210100}"
app_build="${PPE_APP_BUILD:-${repo_root}/results/plugin/exp16_revalidate_r3_app_build_20260809_211500/build}"
images="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/images/test"
f0_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
plugin_so="${repo_root}/results/plugin/exp16_8_rebuild_20260809_170133/build/libppe_yolo_decode_plugin.so"

fail() {
    printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
}

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}"
[[ "$(find "${images}" -maxdepth 1 -type f | wc -l)" -eq 219 ]] || fail "test image count mismatch"
[[ -x "${app_build}/exp15_gpu_postprocess" ]] || fail "Exp15 C++ collector missing"
[[ -x "${app_build}/exp16_plugin_video_infer" ]] || fail "Plugin C++ collector missing"
for engine in "${f0_engine}" "${build_root}/B1.engine" "${build_root}/B2.engine" \
              "${build_root}/P.engine"; do
    [[ -s "${engine}" ]] || fail "engine missing: ${engine}"
done
[[ -s "${plugin_so}" ]] || fail "Plugin shared library missing"

{
    hostname
    uname -m
    git rev-parse HEAD
    git branch --show-current
    nvpmodel -q
    sha256sum "${f0_engine}" "${build_root}/B1.engine" "${build_root}/B2.engine" \
        "${build_root}/P.engine" "${plugin_so}"
    echo "image_count=219"
    echo "confidence_floor=0.25"
    echo "nms_iou=0.70"
    echo "gt_match_iou=0.50"
    echo "preprocess=C++_CUDA_preprocess"
} > "${report_dir}/environment.txt" 2>&1

run_checked() {
    local name="$1"
    shift
    set +e
    "$@" > "${report_dir}/${name}.log" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${report_dir}/${name}_return_code.txt"
    [[ "${rc}" -eq 0 ]] || fail "${name} failed with ${rc}"
}

collect_raw() {
    local name="$1"
    local engine="$2"
    run_checked "collect_${name}" "${app_build}/exp15_gpu_postprocess" \
        --engine "${engine}" --source-type images --source "${images}" \
        --output-dir "${report_dir}/${name}" --postprocess cub --max-frames 219 \
        --warmup 2 --confidence 0.25 --nms-iou 0.70
}

set -e
collect_raw F0 "${f0_engine}"
collect_raw B1 "${build_root}/B1.engine"
run_checked collect_P "${app_build}/exp16_plugin_video_infer" \
    --engine "${build_root}/P.engine" --plugin "${plugin_so}" \
    --source-type images --source "${images}" --output-dir "${report_dir}/P" \
    --postprocess plugin --max-frames 219 --warmup 2 --confidence 0.25 --nms-iou 0.70
collect_raw B2 "${build_root}/B2.engine"

for name in F0 B1 P B2; do
    run_checked "metrics_${name}" python3 "${repo_root}/tools/exp16_revalidate_metrics.py" \
        evaluate --predictions "${report_dir}/${name}/predictions.csv" \
        --images "${images}" --output "${report_dir}/${name}/metrics"
done

mkdir -p "${report_dir}/matches"
run_match() {
    local left="$1"
    local right="$2"
    local name="${left}_${right}"
    run_checked "match_${name}" python3 "${repo_root}/tools/exp16_revalidate_match.py" \
        --left "${report_dir}/${left}/predictions.csv" \
        --right "${report_dir}/${right}/predictions.csv" \
        --output-dir "${report_dir}/matches/${name}" --iou-threshold 0.50
}
run_match F0 B1
run_match F0 B2
run_match B1 B2
run_match F0 P
run_match B1 P
run_match B2 P

set +e
python3 "${repo_root}/tools/exp16_revalidate_metrics.py" gate --root "${report_dir}" \
    > "${report_dir}/r3_gate.log" 2>&1
gate_rc=$?
set -e
printf '%s\n' "${gate_rc}" > "${report_dir}/r3_gate_return_code.txt"
if [[ "${gate_rc}" -ne 0 ]]; then
    cp "${report_dir}/r3_gate_summary.json" "${report_dir}/failure_summary.json"
    printf 'result=REJECTED\nperformance_authorized=false\n' > "${report_dir}/summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
    exit 1
fi
printf 'result=PASS\nperformance_authorized=true\n' > "${report_dir}/summary.txt"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp16_revalidate_r3=PASS output=${report_dir}"

#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/plugin/exp16_revalidate_r4_${timestamp}}"
app_build="${repo_root}/results/plugin/exp16_revalidate_r3_app_build_20260809_211500/build"
build_root="${repo_root}/results/plugin/exp16_revalidate_build_matrix_20260809_210100"
f0_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
plugin_so="${repo_root}/results/plugin/exp16_8_rebuild_20260809_170133/build/libppe_yolo_decode_plugin.so"
video="/home/nvidia/imx219_test_sensor0.mp4"

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists" >&2; exit 1; }
mkdir -p "${report_dir}"
printf 'round,order_index,variant,run_dir,return_code\n' > "${report_dir}/run_registry.csv"
{
    hostname
    uname -m
    git rev-parse HEAD
    git branch --show-current
    nvpmodel -q
    sha256sum "${f0_engine}" "${build_root}/P.engine" "${plugin_so}" "${video}"
    stat -c '%n %s bytes' "${f0_engine}" "${build_root}/P.engine" "${plugin_so}"
    echo "clock_policy=dynamic"
    echo "order=F0_P__P_F0__F0_P"
} > "${report_dir}/environment.txt" 2>&1

run_one() {
    local round="$1" order="$2" variant="$3"
    local run_dir="${report_dir}/round_${round}_${order}_${variant}"
    local binary engine mode
    local extra=()
    if [[ "${variant}" == "F0" ]]; then
        binary="${app_build}/exp15_gpu_postprocess"
        engine="${f0_engine}"
        mode="cub"
    else
        binary="${app_build}/exp16_plugin_video_infer"
        engine="${build_root}/P.engine"
        mode="plugin"
        extra=(--plugin "${plugin_so}")
    fi
    set +e
    python3 "${repo_root}/tools/exp12_monitor.py" \
        --output-dir "${run_dir}" --interval-seconds 0.25 --temperature-stop-c 90 \
        -- "${binary}" --engine "${engine}" "${extra[@]}" \
        --source-type file --source "${video}" --output-dir "${run_dir}/app_output" \
        --postprocess "${mode}" --max-frames 150 --warmup 2 \
        --confidence 0.25 --nms-iou 0.70
    local rc=$?
    set -e
    printf '%s,%s,%s,%s,%s\n' "${round}" "${order}" "${variant}" "${run_dir}" "${rc}" \
        >> "${report_dir}/run_registry.csv"
    if [[ "${rc}" -ne 0 ]]; then
        printf 'result=FAIL\nround=%s\nvariant=%s\nreturn_code=%s\n' \
            "${round}" "${variant}" "${rc}" > "${report_dir}/failure_summary.txt"
        printf '1\n' > "${report_dir}/return_code.txt"
        exit 1
    fi
}

set -e
run_one 1 1 F0
run_one 1 2 P
run_one 2 1 P
run_one 2 2 F0
run_one 3 1 F0
run_one 3 2 P

set +e
python3 "${repo_root}/tools/exp16_revalidate_performance.py" "${report_dir}" \
    > "${report_dir}/performance_gate.log" 2>&1
gate_rc=$?
set -e
printf '%s\n' "${gate_rc}" > "${report_dir}/performance_gate_return_code.txt"
if [[ "${gate_rc}" -eq 0 ]]; then
    printf 'result=PASS\n' > "${report_dir}/summary.txt"
    printf '0\n' > "${report_dir}/return_code.txt"
else
    cp "${report_dir}/performance_gate.json" "${report_dir}/failure_summary.json"
    printf 'result=REJECTED\n' > "${report_dir}/summary.txt"
    printf '1\n' > "${report_dir}/return_code.txt"
fi
sha256sum "${report_dir}/run_registry.csv" "${report_dir}/performance_gate.json" \
    > "${report_dir}/sha256.txt"
exit "${gate_rc}"

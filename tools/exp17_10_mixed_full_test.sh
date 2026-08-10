#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifact="/home/nvidia/models/jetson-ppe/exp17/exp17_8_mixed_build_20260810_152337"
fp16="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
p3_classification="${artifact}/yolo11n_exp17_mixed_p3_classification.engine"
classification="${artifact}/yolo11n_exp17_mixed_classification.engine"
dfl="${artifact}/yolo11n_exp17_mixed_dfl.engine"
data_yaml="/home/nvidia/datasets/jetson-ppe-exp07/construction_ppe3_test_v1/construction_ppe3_jetson_test.yaml"
performance_summary="${repo_root}/results/quantization/exp17_9_mixed_perf_screen_20260810_153903/summary.json"
timestamp="$(date +%Y%m%d_%H%M%S)"
report_dir="${PPE_OUTPUT_DIR:-${repo_root}/results/quantization/exp17_10_mixed_full_test_${timestamp}}"
names=(fp16 p3_classification classification dfl)

[[ ! -e "${report_dir}" ]] || { echo "ERROR: output exists: ${report_dir}" >&2; exit 1; }
mkdir -p "${report_dir}/backend_metrics" "${report_dir}/val_runs"
fail() { printf 'result=FAIL\nreason=%s\n' "$1" > "${report_dir}/failure_summary.txt"; printf '1\n' > "${report_dir}/return_code.txt"; exit 1; }
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/17-explicit-qdq-mixed-precision" ]] || fail "unexpected branch"
for name in "${names[@]}"; do [[ -s "${!name}" ]] || fail "missing Engine: ${name}"; done
[[ -s "${data_yaml}" && -s "${performance_summary}" ]] || fail "missing data or performance summary"
[[ "$(find "$(dirname "${data_yaml}")/images/test" -maxdepth 1 -type f | wc -l)" -eq 219 ]] || fail "test image count mismatch"
{
    hostname
    whoami
    uname -m
    git -C "${repo_root}" rev-parse HEAD
    git -C "${repo_root}" branch --show-current
    for name in "${names[@]}"; do sha256sum "${!name}"; done
    echo "images=219 instances=840 imgsz=640 batch=1"
    echo "conf=0.25 nms_iou=0.70 match_iou=0.50"
} > "${report_dir}/environment.txt" 2>&1

for name in "${names[@]}"; do
    engine="${!name}"
    set +e
    YOLO_AUTOINSTALL=false /usr/bin/python3 -u "${repo_root}/tools/exp07_eval_backend.py" \
        --model "${engine}" --data "${data_yaml}" \
        --output "${report_dir}/backend_metrics/${name}.json" \
        --project "${report_dir}/val_runs" --name "${name}" --device 0 \
        --imgsz 640 --batch 1 --workers 2 > "${report_dir}/${name}_backend.log" 2>&1
    backend_rc=$?
    set -e
    printf '%s\n' "${backend_rc}" > "${report_dir}/${name}_backend_return_code.txt"
    [[ "${backend_rc}" -eq 0 ]] || fail "backend evaluation failed: ${name}"
    set +e
    YOLO_AUTOINSTALL=false /usr/bin/python3 -u "${repo_root}/tools/exp02_8_baseline_error_size_audit.py" \
        --weights "${engine}" --data "${data_yaml}" --output-dir "${report_dir}/${name}_scale" \
        --imgsz 640 --batch 1 --conf 0.25 --nms-iou 0.70 --match-iou 0.50 \
        --max-visuals 0 --serial-static-engine > "${report_dir}/${name}_scale.log" 2>&1
    scale_rc=$?
    set -e
    printf '%s\n' "${scale_rc}" > "${report_dir}/${name}_scale_return_code.txt"
    [[ "${scale_rc}" -eq 0 ]] || fail "scale audit failed: ${name}"
    echo "candidate=${name} backend=PASS scale=PASS" | tee -a "${report_dir}/progress.log"
done
/usr/bin/python3 "${repo_root}/tools/exp17_collect_mixed_accuracy.py" \
    --report-dir "${report_dir}" --performance-summary "${performance_summary}" \
    > "${report_dir}/collector.log" 2>&1 || fail "collector failed"
printf '0\n' > "${report_dir}/return_code.txt"
echo "exp17_10_mixed_full_test=PASS output=${report_dir}"

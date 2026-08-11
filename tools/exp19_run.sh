#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 smoke|formal|diagnostic|stability v0|vfinal file|camera frame_count" >&2
  exit 2
fi
phase="$1"
variant="$2"
source_type="$3"
frame_count="$4"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
video="/home/nvidia/imx219_test_sensor0.mp4"
engine_sha="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
video_sha="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"
binary="${repo_root}/build/exp19_plain/exp15_gpu_postprocess"

[[ "${phase}" == "smoke" || "${phase}" == "formal" || "${phase}" == "diagnostic" || "${phase}" == "stability" ]] || exit 2
[[ "${variant}" == "v0" || "${variant}" == "vfinal" ]] || exit 2
[[ "${source_type}" == "file" || "${source_type}" == "camera" ]] || exit 2
[[ "${frame_count}" =~ ^[1-9][0-9]*$ ]] || exit 2
if [[ "${phase}" == "stability" && ( "${variant}" != "vfinal" || "${source_type}" != "camera" || "${frame_count}" != "54000" ) ]]; then
  echo "stability is frozen to: vfinal camera 54000" >&2
  exit 2
fi
if [[ "${phase}" == "diagnostic" && ( "${source_type}" != "file" || "${frame_count}" != "150" ) ]]; then
  echo "diagnostic is frozen to file 150" >&2
  exit 2
fi
mode="baseline"
[[ "${variant}" == "vfinal" ]] && mode="cub"
warmup=2
[[ "${phase}" != "smoke" ]] && warmup=20
interval=0.1
[[ "${source_type}" == "camera" ]] && interval=1
timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="exp19_${phase}_${variant}_${source_type}_${timestamp}"
output_dir="${repo_root}/results/final_benchmark/${run_id}"
app_output="${output_dir}/app_output"
mkdir -p "${app_output}"

fail_early() {
  printf 'result=FAIL reason=%s\n' "$1" | tee "${output_dir}/failure_summary.txt"
  printf '1\n' > "${output_dir}/return_code.txt"
  exit 1
}
for path in "${binary}" "${engine}" "${repo_root}/tools/exp12_monitor.py" \
  "${repo_root}/tools/exp15_validate.py" "${repo_root}/tools/exp19_analyze.py"; do
  [[ -e "${path}" ]] || fail_early "missing:${path}"
done
[[ "$(git -C "${repo_root}" branch --show-current)" == "exp/19-final-benchmark" ]] || fail_early unexpected_branch
[[ "$(sha256sum "${engine}" | awk '{print $1}')" == "${engine_sha}" ]] || fail_early engine_sha_mismatch
nvpmodel -q | grep -q 'NV Power Mode: 25W' || fail_early nvpmodel_not_25w
if [[ "${phase}" == "diagnostic" ]]; then
  python3 "${repo_root}/tools/exp12_clock_status.py" \
    --output "${output_dir}/clock_status.json" --require-locked \
    > "${output_dir}/clock_status_stdout.txt" 2>&1 || fail_early clocks_not_locked
else
  [[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" == "schedutil" ]] || fail_early cpu_not_dynamic
  [[ "$(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/governor)" == "nvhost_podgov" ]] || fail_early gpu_not_dynamic
fi
if [[ "${source_type}" == "file" ]]; then
  [[ "$(sha256sum "${video}" | awk '{print $1}')" == "${video_sha}" ]] || fail_early video_sha_mismatch
  source_args=(--source-type file --source "${video}" --max-frames "${frame_count}")
else
  source_args=(--source-type camera --sensor-id 0 --width 1920 --height 1080 --fps 30 --max-frames "${frame_count}")
fi

command=("${binary}" --engine "${engine}" "${source_args[@]}"
  --output-dir "${app_output}" --warmup "${warmup}" --confidence 0.25
  --nms-iou 0.70 --postprocess "${mode}")
{
  hostname; whoami; uname -a
  git -C "${repo_root}" branch --show-current
  git -C "${repo_root}" rev-parse HEAD
  nvpmodel -q
  printf 'phase=%s\nvariant=%s\nmode=%s\nsource_type=%s\nframes=%s\nwarmup=%s\nmonitor_interval=%s\n' \
    "${phase}" "${variant}" "${mode}" "${source_type}" "${frame_count}" "${warmup}" "${interval}"
  sha256sum "${binary}" "${engine}"
  [[ "${source_type}" == "file" ]] && sha256sum "${video}"
} > "${output_dir}/environment.txt" 2>&1
printf '%q ' "${command[@]}" > "${output_dir}/command.txt"; printf '\n' >> "${output_dir}/command.txt"

python3 "${repo_root}/tools/exp12_monitor.py" --output-dir "${output_dir}" \
  --interval-seconds "${interval}" --temperature-stop-c 90 -- "${command[@]}"
monitor_rc=$?
printf '%s\n' "${monitor_rc}" > "${output_dir}/monitor_return_code.txt"
if [[ ${monitor_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=monitor return_code=%s\n' "${monitor_rc}" > "${output_dir}/failure_summary.txt"
  printf '%s\n' "${monitor_rc}" > "${output_dir}/return_code.txt"
  exit "${monitor_rc}"
fi

validate=(python3 "${repo_root}/tools/exp15_validate.py" "${output_dir}"
  --expected-frames "${frame_count}" --mode "${mode}")
[[ "${source_type}" == "file" && "${frame_count}" == "150" ]] && validate+=(--require-file-digest)
"${validate[@]}" > "${output_dir}/validation.log" 2>&1
validation_rc=$?
printf '%s\n' "${validation_rc}" > "${output_dir}/validation_return_code.txt"
if [[ ${validation_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=validation return_code=%s\n' "${validation_rc}" > "${output_dir}/failure_summary.txt"
  printf '1\n' > "${output_dir}/return_code.txt"
  exit 1
fi

python3 "${repo_root}/tools/exp19_analyze.py" --run-dir "${output_dir}" \
  --phase "${phase}" --variant "${variant}" --source-type "${source_type}" \
  > "${output_dir}/analyze_stdout.txt" 2>&1
analyze_rc=$?
printf '%s\n' "${analyze_rc}" > "${output_dir}/analyze_return_code.txt"
printf '%s\n' "${analyze_rc}" > "${output_dir}/return_code.txt"
if [[ ${analyze_rc} -ne 0 ]]; then
  cp "${output_dir}/exp19_summary.txt" "${output_dir}/failure_summary.txt"
  exit "${analyze_rc}"
fi
sha256sum "${app_output}/summary.json" "${app_output}/frames.csv" \
  "${app_output}/detections.csv" "${output_dir}/validation.json" \
  "${output_dir}/exp19_summary.json" > "${output_dir}/sha256.txt"
printf 'result=PASS run_id=%s output_dir=%s\n' "${run_id}" "${output_dir}" \
  | tee "${output_dir}/runner_summary.txt"

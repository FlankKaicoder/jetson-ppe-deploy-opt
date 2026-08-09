#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 P0|P1|P2 frame_count" >&2
  exit 2
fi
path_name="$1"
frame_count="$2"
[[ "${frame_count}" =~ ^[1-9][0-9]*$ ]] || exit 2
case "${path_name}" in
  P0) mode="raw_pinned" ;;
  P1) mode="fixed" ;;
  P2) mode="cub" ;;
  *) echo "path must be P0, P1, or P2" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${repo_root}/build/postprocess_gain_gate/exp15_gpu_postprocess"
engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
video="/home/nvidia/imx219_test_sensor0.mp4"
engine_sha="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
video_sha="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="postprocess_gain_gate_${path_name}_${timestamp}"
output_dir="${repo_root}/results/gpu_postprocess/${run_id}"
app_output="${output_dir}/app_output"
mkdir -p "${app_output}"

for path in "${binary}" "${engine}" "${video}" \
  "${repo_root}/tools/exp15_validate.py" \
  "${repo_root}/tools/exp12_monitor.py"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing=%s\n' "${path}" \
      > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done
[[ "$(sha256sum "${engine}" | awk '{print $1}')" == "${engine_sha}" ]] || exit 1
[[ "$(sha256sum "${video}" | awk '{print $1}')" == "${video_sha}" ]] || exit 1

command=("${binary}" --engine "${engine}" --source-type file --source "${video}"
  --max-frames "${frame_count}" --output-dir "${app_output}" --warmup 2
  --confidence 0.25 --nms-iou 0.70 --postprocess "${mode}")
{
  hostname
  whoami
  uname -a
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  git -C "${repo_root}" status --short --untracked-files=no
  nvpmodel -q
  printf 'path=%s\nmode=%s\nframes=%s\n' \
    "${path_name}" "${mode}" "${frame_count}"
  sha256sum "${binary}" "${engine}" "${video}"
} > "${output_dir}/environment.txt" 2>&1
printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
python3 "${repo_root}/tools/exp12_monitor.py" \
  --output-dir "${output_dir}/monitor" --interval-seconds 0.5 \
  --temperature-stop-c 90 -- "${command[@]}" \
  > "${output_dir}/monitor.log" 2>&1
run_rc=$?
cp "${output_dir}/monitor/application.log" "${output_dir}/run.log"
printf '%s\n' "${run_rc}" > "${output_dir}/return_code.txt"
if [[ ${run_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=monitor_or_runtime return_code=%s\n' "${run_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${run_rc}"
fi

validate=(python3 "${repo_root}/tools/exp15_validate.py" "${output_dir}"
  --expected-frames "${frame_count}" --mode "${mode}")
[[ "${frame_count}" == "150" ]] && validate+=(--require-file-digest)
"${validate[@]}" > "${output_dir}/validation.log" 2>&1
validation_rc=$?
printf '%s\n' "${validation_rc}" > "${output_dir}/validation_return_code.txt"
if [[ ${validation_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=validation return_code=%s\n' "${validation_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${validation_rc}"
fi

sha256sum "${app_output}/summary.json" "${app_output}/frames.csv" \
  "${app_output}/detections.csv" "${output_dir}/validation.json" \
  "${output_dir}/monitor/monitor_summary.json" \
  "${output_dir}/monitor/tegrastats.log" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS path=%s output_dir=%s\n' "${path_name}" "${output_dir}" \
  | tee "${output_dir}/summary.txt"

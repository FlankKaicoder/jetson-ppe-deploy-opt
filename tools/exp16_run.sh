#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 control|plugin frame_count" >&2
  exit 2
fi
variant="$1"
frame_count="$2"
[[ "${variant}" == "control" || "${variant}" == "plugin" ]] || exit 2
[[ "${frame_count}" =~ ^[1-9][0-9]*$ ]] || exit 2

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
app_build="${PPE_EXP16_APP_BUILD:-${repo_root}/results/plugin/exp16_6_app_build_20260809_165659/build}"
control_engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
plugin_root="${repo_root}/results/plugin/exp16_8_rebuild_20260809_170133"
plugin_engine="${plugin_root}/yolo11n_exp16_plugin_fp16.engine"
plugin_library="${plugin_root}/build/libppe_yolo_decode_plugin.so"
video="/home/nvidia/imx219_test_sensor0.mp4"
control_engine_sha="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
plugin_engine_sha="c0b1cca81c18da176e04a38607f85d95f54720f7984bceeb51fa47d7f29c56cf"
plugin_library_sha="b5d402ffab879758c16289c7c385ddab7eeaa0c270a76f65042eb35a5e8477f1"
video_sha="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"

if [[ "${variant}" == "control" ]]; then
  binary="${app_build}/exp15_gpu_postprocess"
  engine="${control_engine}"
  mode="cub"
  extra_args=()
else
  binary="${app_build}/exp16_plugin_video_infer"
  engine="${plugin_engine}"
  mode="plugin"
  extra_args=(--plugin "${plugin_library}")
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="exp16_formal_${variant}_file_${timestamp}"
output_dir="${repo_root}/results/plugin/${run_id}"
app_output="${output_dir}/app_output"
mkdir -p "${app_output}"

for path in "${binary}" "${engine}" "${video}" "${repo_root}/tools/exp15_validate.py"; do
  if [[ ! -s "${path}" ]]; then
    printf 'result=FAIL missing=%s\n' "${path}" > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done
[[ "$(sha256sum "${control_engine}" | awk '{print $1}')" == "${control_engine_sha}" ]] || exit 1
[[ "$(sha256sum "${plugin_engine}" | awk '{print $1}')" == "${plugin_engine_sha}" ]] || exit 1
[[ "$(sha256sum "${plugin_library}" | awk '{print $1}')" == "${plugin_library_sha}" ]] || exit 1
[[ "$(sha256sum "${video}" | awk '{print $1}')" == "${video_sha}" ]] || exit 1

command=("${binary}" --engine "${engine}" --source-type file --source "${video}"
  --max-frames "${frame_count}" --output-dir "${app_output}" --warmup 2
  --confidence 0.25 --nms-iou 0.70 --postprocess "${mode}" "${extra_args[@]}")
{
  hostname
  whoami
  uname -a
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  nvpmodel -q
  printf 'variant=%s\nmode=%s\nframes=%s\nclock_policy=dynamic\n' \
    "${variant}" "${mode}" "${frame_count}"
  sha256sum "${binary}" "${engine}" "${video}"
  [[ "${variant}" == "plugin" ]] && sha256sum "${plugin_library}"
} > "${output_dir}/environment.txt" 2>&1
timeout 1s tegrastats --interval 200 > "${output_dir}/tegrastats_before.txt" 2>&1
printf '%s\n' "$?" > "${output_dir}/tegrastats_before_return_code.txt"
printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" 2>&1 | tee "${output_dir}/run.log"
run_rc=${PIPESTATUS[0]}
printf '%s\n' "${run_rc}" > "${output_dir}/return_code.txt"
timeout 1s tegrastats --interval 200 > "${output_dir}/tegrastats_after.txt" 2>&1
printf '%s\n' "$?" > "${output_dir}/tegrastats_after_return_code.txt"
if [[ ${run_rc} -ne 0 ]]; then
  printf 'result=FAIL return_code=%s\n' "${run_rc}" > "${output_dir}/failure_summary.txt"
  exit "${run_rc}"
fi

python3 "${repo_root}/tools/exp15_validate.py" "${output_dir}" \
  --expected-frames "${frame_count}" --mode "${mode}" \
  > "${output_dir}/validation.log" 2>&1
validation_rc=$?
printf '%s\n' "${validation_rc}" > "${output_dir}/validation_return_code.txt"
if [[ ${validation_rc} -ne 0 ]]; then
  printf 'result=FAIL validation_rc=%s\n' "${validation_rc}" > "${output_dir}/failure_summary.txt"
  exit 1
fi
sha256sum "${app_output}/summary.json" "${app_output}/frames.csv" \
  "${app_output}/detections.csv" "${output_dir}/validation.json" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS variant=%s output_dir=%s\n' "${variant}" "${output_dir}" \
  | tee "${output_dir}/summary.txt"

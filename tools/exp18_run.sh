#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 benchmark|profile normal|graph file|camera frame_count [candidate-trace]" >&2
  exit 2
fi

run_kind="$1"
variant="$2"
source_type="$3"
frame_count="$4"
candidate_trace="${5:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
engine="/home/nvidia/models/jetson-ppe/exp07/exp07_1_trt_fp32_fp16_formal_20260806_170501/yolo11n_baseline_exp07_b1_640_fp16.engine"
video="/home/nvidia/imx219_test_sensor0.mp4"
engine_sha="88dcc29d2c66b77bdf5b3ac90f327f793516365d5967b51155987a5b736c0a83"
video_sha="f00e116a9d8f1a9f2bdcb4fc08c8720019e6f61047934613b0940da961a66665"

[[ "${run_kind}" == "benchmark" || "${run_kind}" == "profile" ]] || exit 2
[[ "${variant}" == "normal" || "${variant}" == "graph" ]] || exit 2
[[ "${source_type}" == "file" || "${source_type}" == "camera" ]] || exit 2
[[ "${frame_count}" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -z "${candidate_trace}" || "${candidate_trace}" == "candidate-trace" ]] || exit 2

mode="cub"
[[ "${variant}" == "graph" ]] && mode="graph"
build_name="plain"
[[ "${run_kind}" == "profile" ]] && build_name="nvtx"
binary="${repo_root}/build/exp15_${build_name}/exp15_gpu_postprocess"
timestamp="$(date +%Y%m%d_%H%M%S)"
run_id="exp18_${run_kind}_${variant}_${source_type}_${timestamp}"
[[ -n "${candidate_trace}" ]] && run_id="exp18_trace_${variant}_${source_type}_${timestamp}"
output_dir="${repo_root}/results/profiling/${run_id}"
app_output="${output_dir}/app_output"
mkdir -p "${app_output}"

for path in "${binary}" "${engine}" "${repo_root}/tools/exp15_validate.py"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing=%s\n' "${path}" > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done
if [[ "$(sha256sum "${engine}" | awk '{print $1}')" != "${engine_sha}" ]]; then
  printf 'result=FAIL engine_sha256_mismatch\n' > "${output_dir}/failure_summary.txt"
  exit 1
fi
if [[ "${source_type}" == "file" ]]; then
  if [[ "$(sha256sum "${video}" | awk '{print $1}')" != "${video_sha}" ]]; then
    printf 'result=FAIL video_sha256_mismatch\n' > "${output_dir}/failure_summary.txt"
    exit 1
  fi
  source_args=(--source-type file --source "${video}" --max-frames "${frame_count}")
else
  source_args=(--source-type camera --sensor-id 0 --width 1920 --height 1080
    --fps 30 --max-frames "${frame_count}")
fi

app_command=("${binary}" --engine "${engine}" "${source_args[@]}"
  --output-dir "${app_output}" --warmup 2 --confidence 0.25 --nms-iou 0.70
  --postprocess "${mode}")
[[ -n "${candidate_trace}" ]] && app_command+=(--candidate-trace 1)
{
  hostname
  whoami
  uname -a
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  nvpmodel -q
  nsys --version
  printf 'run_kind=%s\nvariant=%s\nmode=%s\nsource_type=%s\nframes=%s\n' \
    "${run_kind}" "${variant}" "${mode}" "${source_type}" "${frame_count}"
  sha256sum "${binary}" "${engine}"
  [[ "${source_type}" == "file" ]] && sha256sum "${video}"
} > "${output_dir}/environment.txt" 2>&1
printf '%q ' "${app_command[@]}" > "${output_dir}/app_command.txt"
printf '\n' >> "${output_dir}/app_command.txt"

if [[ "${run_kind}" == "profile" ]]; then
  full_command=(nsys profile --trace=cuda,nvtx,osrt --cuda-graph-trace=node --sample=none
    --cpuctxsw=none --force-overwrite=true --output="${output_dir}/report"
    "${app_command[@]}")
else
  full_command=("${app_command[@]}")
fi
printf '%q ' "${full_command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${full_command[@]}" 2>&1 | tee "${output_dir}/run.log"
rc=${PIPESTATUS[0]}
printf '%s\n' "${rc}" > "${output_dir}/return_code.txt"
if [[ ${rc} -ne 0 ]]; then
  printf 'result=FAIL return_code=%s\n' "${rc}" > "${output_dir}/failure_summary.txt"
  exit "${rc}"
fi

validate=(python3 "${repo_root}/tools/exp15_validate.py" "${output_dir}"
  --expected-frames "${frame_count}" --mode "${mode}")
if [[ "${source_type}" == "file" && "${frame_count}" == "150" ]]; then
  validate+=(--require-file-digest)
fi
"${validate[@]}" > "${output_dir}/validation.log" 2>&1
validation_rc=$?
printf '%s\n' "${validation_rc}" > "${output_dir}/validation_return_code.txt"
if [[ ${validation_rc} -ne 0 ]]; then
  printf 'result=FAIL validation_rc=%s\n' "${validation_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit 1
fi

if [[ "${run_kind}" == "profile" ]]; then
  if [[ ! -s "${output_dir}/report.nsys-rep" ]]; then
    printf 'result=FAIL missing_report\n' > "${output_dir}/failure_summary.txt"
    exit 1
  fi
  nsys stats --force-export=true "${output_dir}/report.nsys-rep" \
    > "${output_dir}/nsys_stats.txt" 2>&1
  stats_rc=$?
  printf '%s\n' "${stats_rc}" > "${output_dir}/stats_return_code.txt"
  if [[ ${stats_rc} -ne 0 ]]; then
    printf 'result=FAIL stats_rc=%s\n' "${stats_rc}" \
      > "${output_dir}/failure_summary.txt"
    exit 1
  fi
fi

sha256sum "${app_output}/summary.json" "${app_output}/frames.csv" \
  "${app_output}/detections.csv" "${output_dir}/validation.json" \
  > "${output_dir}/sha256.txt"
if [[ -n "${candidate_trace}" ]]; then
  sha256sum "${app_output}/candidates.csv" >> "${output_dir}/sha256.txt"
fi
printf 'result=PASS run_id=%s output_dir=%s\n' "${run_id}" "${output_dir}" \
  | tee "${output_dir}/summary.txt"

#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/plugin/exp16_formal_compare_${timestamp}"
mkdir -p "${output_dir}"
registry="${output_dir}/run_registry.csv"
printf 'round,order_index,variant,run_dir,return_code\n' > "${registry}"
orders=("control plugin" "plugin control" "control plugin")

for round_index in 1 2 3; do
  order_index=0
  control_dir=""
  plugin_dir=""
  for variant in ${orders[$((round_index - 1))]}; do
    order_index=$((order_index + 1))
    stdout_file="${output_dir}/round_${round_index}_${order_index}_${variant}.log"
    "${repo_root}/tools/exp16_run.sh" "${variant}" 150 \
      > "${stdout_file}" 2>&1
    run_rc=$?
    run_dir="$(sed -n 's/.*output_dir=//p' "${stdout_file}" | tail -1)"
    printf '%s,%s,%s,%s,%s\n' "${round_index}" "${order_index}" \
      "${variant}" "${run_dir}" "${run_rc}" >> "${registry}"
    if [[ ${run_rc} -ne 0 || -z "${run_dir}" ]]; then
      printf 'result=FAIL round=%s variant=%s return_code=%s\n' \
        "${round_index}" "${variant}" "${run_rc}" \
        > "${output_dir}/failure_summary.txt"
      exit 1
    fi
    if [[ "${variant}" == "control" ]]; then
      control_dir="${run_dir}"
    else
      plugin_dir="${run_dir}"
    fi
  done

  semantic_dir="${output_dir}/round_${round_index}_semantic"
  python3 "${repo_root}/tools/exp16_compare_detections.py" \
    --reference "${control_dir}/app_output/detections.csv" \
    --candidate "${plugin_dir}/app_output/detections.csv" \
    --report-dir "${semantic_dir}" --expected-detections 151 \
    --box-max-abs 2.0 --confidence-max-abs 0.005 \
    > "${output_dir}/round_${round_index}_semantic.log" 2>&1
  semantic_rc=$?
  printf '%s\n' "${semantic_rc}" \
    > "${output_dir}/round_${round_index}_semantic_return_code.txt"
  if [[ ${semantic_rc} -ne 0 ]]; then
    printf 'result=FAIL round=%s stage=semantic return_code=%s\n' \
      "${round_index}" "${semantic_rc}" > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done

python3 "${repo_root}/tools/exp16_analyze_formal.py" "${output_dir}" \
  > "${output_dir}/analysis.log" 2>&1
analysis_rc=$?
printf '%s\n' "${analysis_rc}" > "${output_dir}/analysis_return_code.txt"
if [[ ${analysis_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=analysis return_code=%s\n' "${analysis_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${analysis_rc}"
fi
sha256sum "${registry}" "${output_dir}/analysis.json" \
  "${output_dir}/analysis.txt" > "${output_dir}/sha256.txt"
printf 'result=PASS output_dir=%s\n' "${output_dir}" \
  | tee "${output_dir}/summary.txt"

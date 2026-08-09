#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/gpu_postprocess/postprocess_gain_gate_formal_${timestamp}"
mkdir -p "${output_dir}"
registry="${output_dir}/run_registry.csv"
printf 'round,order_index,path,run_dir,return_code\n' > "${registry}"

orders=(
  "P0 P1 P2"
  "P2 P1 P0"
  "P1 P0 P2"
)
for round_index in 1 2 3; do
  order_index=0
  for path in ${orders[$((round_index - 1))]}; do
    order_index=$((order_index + 1))
    stdout_file="${output_dir}/round_${round_index}_${order_index}_${path}.log"
    "${repo_root}/tools/postprocess_gain_gate_run.sh" "${path}" 150 \
      > "${stdout_file}" 2>&1
    run_rc=$?
    run_dir="$(sed -n 's/.*output_dir=//p' "${stdout_file}" | tail -1)"
    printf '%s,%s,%s,%s,%s\n' "${round_index}" "${order_index}" \
      "${path}" "${run_dir}" "${run_rc}" >> "${registry}"
    if [[ ${run_rc} -ne 0 || -z "${run_dir}" ]]; then
      printf 'result=FAIL round=%s path=%s return_code=%s\n' \
        "${round_index}" "${path}" "${run_rc}" \
        > "${output_dir}/failure_summary.txt"
      exit 1
    fi
  done
done

python3 "${repo_root}/tools/postprocess_gain_gate_analyze_formal.py" \
  "${output_dir}" > "${output_dir}/analysis.log" 2>&1
analysis_rc=$?
printf '%s\n' "${analysis_rc}" > "${output_dir}/analysis_return_code.txt"
if [[ ${analysis_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=analysis return_code=%s\n' "${analysis_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${analysis_rc}"
fi

sha256sum "${registry}" "${output_dir}/runs.csv" \
  "${output_dir}/analysis.json" "${output_dir}/analysis.txt" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS output_dir=%s\n' "${output_dir}" \
  | tee "${output_dir}/summary.txt"

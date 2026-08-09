#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 || ( "$1" != "atomic" && "$1" != "cub" ) ]]; then
  echo "usage: $0 atomic|cub" >&2
  exit 2
fi
mode="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ncu="/opt/nvidia/nsight-compute/2024.3.1/ncu"
binary="${repo_root}/build/exp15_plain/cuda/exp15_gpu_postprocess_test"
fixture="${repo_root}/results/gpu_postprocess/exp15_raw_fixture_20260808_202730/raw_output_f32.bin"
fixture_sha="0e6aff4557d989ec62c26908988bcb5b15222de4a7d66f53ea73f36ce825abfe"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/gpu_postprocess/exp15_ncu_${mode}_${timestamp}"
mkdir -p "${output_dir}"

for path in "${ncu}" "${binary}" "${fixture}"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing=%s\n' "${path}" > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done
if [[ "$(sha256sum "${fixture}" | awk '{print $1}')" != "${fixture_sha}" ]]; then
  printf 'result=FAIL fixture_sha256_mismatch\n' > "${output_dir}/failure_summary.txt"
  exit 1
fi
{
  hostname; whoami; id; uname -a
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  "${ncu}" --version
  printf 'mode=%s\n' "${mode}"
  sha256sum "${binary}" "${fixture}"
} > "${output_dir}/environment.txt" 2>&1

command=("${ncu}" --set full --profile-from-start off --target-processes all
  --force-overwrite --export "${output_dir}/report"
  "${binary}" --profile-fixture "${fixture}" --mode "${mode}" --iterations 1)
printf '%q ' "${command[@]}" > "${output_dir}/command.txt"; printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" > "${output_dir}/run.log" 2>&1
rc=$?
printf '%s\n' "${rc}" > "${output_dir}/return_code.txt"
if [[ ${rc} -ne 0 || ! -s "${output_dir}/report.ncu-rep" ]]; then
  printf 'result=FAIL mode=%s return_code=%s report_exists=%s\n' \
    "${mode}" "${rc}" "$(test -s "${output_dir}/report.ncu-rep" && echo true || echo false)" \
    > "${output_dir}/failure_summary.txt"
  printf 'result=FAIL output_dir=%s\n' "${output_dir}"
  tail -n 30 "${output_dir}/run.log"
  exit 1
fi
"${ncu}" --import "${output_dir}/report.ncu-rep" --csv --page details \
  > "${output_dir}/details.csv" 2> "${output_dir}/import.log"
import_rc=$?
printf '%s\n' "${import_rc}" > "${output_dir}/import_return_code.txt"
if [[ ${import_rc} -ne 0 || ! -s "${output_dir}/details.csv" ]]; then
  printf 'result=FAIL import_return_code=%s\n' "${import_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit 1
fi
sha256sum "${output_dir}/report.ncu-rep" "${output_dir}/details.csv" \
  > "${output_dir}/sha256.txt"
printf 'result=PASS mode=%s output_dir=%s\n' "${mode}" "${output_dir}" \
  | tee "${output_dir}/summary.txt"

#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture="${repo_root}/results/gpu_postprocess/exp15_raw_fixture_20260808_202730/raw_output_f32.bin"
fixture_sha="0e6aff4557d989ec62c26908988bcb5b15222de4a7d66f53ea73f36ce825abfe"
binary="${repo_root}/build/postprocess_gain_gate/cuda/postprocess_gain_gate_benchmark"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/gpu_postprocess/postprocess_gain_gate_fixture_${timestamp}"
mkdir -p "${output_dir}"

for path in "${fixture}" "${binary}" \
  "${repo_root}/tools/postprocess_gain_gate_analyze_fixture.py"; do
  if [[ ! -e "${path}" ]]; then
    printf 'result=FAIL missing=%s\n' "${path}" \
      > "${output_dir}/failure_summary.txt"
    exit 1
  fi
done
if [[ "$(sha256sum "${fixture}" | awk '{print $1}')" != "${fixture_sha}" ]]; then
  printf 'result=FAIL fixture_sha256_mismatch\n' \
    > "${output_dir}/failure_summary.txt"
  exit 1
fi

command=("${binary}" --fixture "${fixture}" --output-dir "${output_dir}"
  --warmup 20 --iterations 1000)
{
  hostname
  whoami
  uname -a
  git -C "${repo_root}" rev-parse HEAD
  git -C "${repo_root}" branch --show-current
  git -C "${repo_root}" status --short --untracked-files=no
  nvpmodel -q
  sha256sum "${fixture}" "${binary}"
} > "${output_dir}/environment.txt" 2>&1
printf '%q ' "${command[@]}" > "${output_dir}/command.txt"
printf '\n' >> "${output_dir}/command.txt"
"${command[@]}" > "${output_dir}/run.log" 2>&1
run_rc=$?
printf '%s\n' "${run_rc}" > "${output_dir}/return_code.txt"
if [[ ${run_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=benchmark return_code=%s\n' "${run_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${run_rc}"
fi

python3 "${repo_root}/tools/postprocess_gain_gate_analyze_fixture.py" \
  "${output_dir}" > "${output_dir}/analysis.log" 2>&1
analysis_rc=$?
printf '%s\n' "${analysis_rc}" > "${output_dir}/analysis_return_code.txt"
if [[ ${analysis_rc} -ne 0 ]]; then
  printf 'result=FAIL stage=analysis return_code=%s\n' "${analysis_rc}" \
    > "${output_dir}/failure_summary.txt"
  exit "${analysis_rc}"
fi

sha256sum "${output_dir}/samples.csv" "${output_dir}/analysis.json" \
  "${output_dir}/analysis.txt" > "${output_dir}/sha256.txt"
printf 'result=PASS output_dir=%s\n' "${output_dir}" \
  | tee "${output_dir}/summary.txt"

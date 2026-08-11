#!/usr/bin/env bash
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${repo_root}/results/final_benchmark/exp19_fixed_diagnostic_${timestamp}"
state_file="/tmp/exp19_jetsonclocks_before_${timestamp}.conf"
mkdir -p "${output_dir}"
restore_needed=0

restore_clocks() {
  if [[ ${restore_needed} -eq 1 ]]; then
    sudo jetson_clocks --restore "${state_file}" \
      > "${output_dir}/restore.log" 2>&1
    restore_rc=$?
    printf '%s\n' "${restore_rc}" > "${output_dir}/restore_return_code.txt"
    sleep 2
    {
      cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
      cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/governor
    } > "${output_dir}/restored_governors.txt"
    if [[ ${restore_rc} -ne 0 ]] || ! grep -qx schedutil "${output_dir}/restored_governors.txt" || \
       ! grep -qx nvhost_podgov "${output_dir}/restored_governors.txt"; then
      printf 'result=FAIL reason=clock_restore_failed\n' > "${output_dir}/failure_summary.txt"
      return 1
    fi
  fi
}
trap 'restore_clocks' EXIT

[[ "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)" == "schedutil" ]] || exit 1
[[ "$(cat /sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/governor)" == "nvhost_podgov" ]] || exit 1
sudo jetson_clocks --store "${state_file}" > "${output_dir}/store.log" 2>&1
store_rc=$?
printf '%s\n' "${store_rc}" > "${output_dir}/store_return_code.txt"
[[ ${store_rc} -eq 0 && -s "${state_file}" ]] || exit 1
restore_needed=1
sudo jetson_clocks > "${output_dir}/lock.log" 2>&1
lock_rc=$?
printf '%s\n' "${lock_rc}" > "${output_dir}/lock_return_code.txt"
[[ ${lock_rc} -eq 0 ]] || exit 1
python3 "${repo_root}/tools/exp12_clock_status.py" \
  --output "${output_dir}/locked_clock_status.json" --require-locked \
  > "${output_dir}/locked_clock_status_stdout.txt" 2>&1 || exit 1

order=(v0 vfinal vfinal v0 v0 vfinal)
for variant in "${order[@]}"; do
  bash "${repo_root}/tools/exp19_run.sh" diagnostic "${variant}" file 150 || exit 1
done
restore_clocks || exit 1
restore_needed=0
printf 'result=PASS state_file=%s output_dir=%s\n' "${state_file}" "${output_dir}" \
  | tee "${output_dir}/summary.txt"

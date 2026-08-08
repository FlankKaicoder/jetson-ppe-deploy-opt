#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_int(path: Path) -> int:
    return int(path.read_text().strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-locked", action="store_true")
    args = parser.parse_args()

    cpus = []
    for cpu in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        cpufreq = cpu / "cpufreq"
        if not cpufreq.is_dir():
            continue
        cpus.append(
            {
                "cpu": cpu.name,
                "current_khz": read_int(cpufreq / "scaling_cur_freq"),
                "minimum_khz": read_int(cpufreq / "scaling_min_freq"),
                "maximum_khz": read_int(cpufreq / "scaling_max_freq"),
                "hardware_maximum_khz": read_int(cpufreq / "cpuinfo_max_freq"),
                "governor": (cpufreq / "scaling_governor").read_text().strip(),
            }
        )
    gpu = None
    for device in Path("/sys/class/devfreq").glob("*"):
        try:
            name = (device / "name").read_text().strip()
        except OSError:
            continue
        if "gpu" in name.lower():
            gpu = {
                "name": name,
                "current_hz": read_int(device / "cur_freq"),
                "minimum_hz": read_int(device / "min_freq"),
                "maximum_hz": read_int(device / "max_freq"),
                "governor": (device / "governor").read_text().strip(),
            }
            break
    nvpmodel = subprocess.run(
        ["nvpmodel", "-q"], check=False, capture_output=True, text=True
    )
    cpu_locked = bool(cpus) and all(
        item["minimum_khz"] == item["maximum_khz"] for item in cpus
    )
    gpu_locked = bool(gpu) and gpu["minimum_hz"] == gpu["maximum_hz"]
    locked = cpu_locked and gpu_locked
    result = {
        "result": "PASS" if nvpmodel.returncode == 0 and (locked or not args.require_locked) else "FAIL",
        "nvpmodel": nvpmodel.stdout.strip(),
        "cpu_locked": cpu_locked,
        "gpu_locked": gpu_locked,
        "jetson_clocks_inferred_locked": locked,
        "cpus": cpus,
        "gpu": gpu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

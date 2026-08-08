#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def read_status(pid: int) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if key in {"VmRSS", "VmSwap", "Threads"} and fields:
            values[key] = int(fields[0])
    return values


def read_cpu_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[13]) + int(fields[14])


def read_max_temperature_c() -> float:
    temperatures = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            temperatures.append(int((zone / "temp").read_text().strip()) / 1000.0)
        except (OSError, TypeError, ValueError):
            continue
    return max(temperatures) if temperatures else float("nan")


def stop_process(process: subprocess.Popen, timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--temperature-stop-c", type=float, default=90.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval_seconds <= 0:
        parser.error("--interval-seconds must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tegrastats_path = args.output_dir / "tegrastats.log"
    application_path = args.output_dir / "application.log"
    process_csv_path = args.output_dir / "process_samples.csv"
    monitor_summary_path = args.output_dir / "monitor_summary.json"
    interval_ms = max(100, int(round(args.interval_seconds * 1000.0)))
    started_at = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    safety_stop = False
    sample_count = 0
    max_observed_temperature = float("nan")
    app_return_code = None
    monitor_error = None

    tegrastats_file = tegrastats_path.open("w", encoding="utf-8")
    application_file = application_path.open("w", encoding="utf-8")
    process_csv_file = process_csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        process_csv_file,
        fieldnames=[
            "timestamp_utc", "elapsed_seconds", "rss_kb", "swap_kb", "threads",
            "process_cpu_percent", "host_cpu_count", "max_temperature_c",
        ],
    )
    writer.writeheader()
    process_csv_file.flush()

    tegrastats = None
    application = None
    previous_ticks = None
    previous_sample_time = None
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    try:
        tegrastats = subprocess.Popen(
            ["tegrastats", "--interval", str(interval_ms)],
            stdout=tegrastats_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        application = subprocess.Popen(
            command,
            stdout=application_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while application.poll() is None:
            sample_time = time.monotonic()
            try:
                status = read_status(application.pid)
                ticks = read_cpu_ticks(application.pid)
                temperature = read_max_temperature_c()
            except (FileNotFoundError, ProcessLookupError):
                break
            cpu_percent = 0.0
            if previous_ticks is not None and previous_sample_time is not None:
                delta_seconds = sample_time - previous_sample_time
                if delta_seconds > 0:
                    cpu_percent = (
                        (ticks - previous_ticks) / ticks_per_second / delta_seconds * 100.0
                    )
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": f"{sample_time - start_monotonic:.6f}",
                    "rss_kb": status.get("VmRSS", 0),
                    "swap_kb": status.get("VmSwap", 0),
                    "threads": status.get("Threads", 0),
                    "process_cpu_percent": f"{cpu_percent:.6f}",
                    "host_cpu_count": os.cpu_count() or 0,
                    "max_temperature_c": f"{temperature:.3f}",
                }
            )
            process_csv_file.flush()
            sample_count += 1
            if temperature == temperature:
                if max_observed_temperature != max_observed_temperature:
                    max_observed_temperature = temperature
                else:
                    max_observed_temperature = max(max_observed_temperature, temperature)
                if temperature >= args.temperature_stop_c:
                    safety_stop = True
                    application.send_signal(signal.SIGTERM)
                    break
            previous_ticks = ticks
            previous_sample_time = sample_time
            deadline = sample_time + args.interval_seconds
            while application.poll() is None and time.monotonic() < deadline:
                time.sleep(min(0.1, deadline - time.monotonic()))
        app_return_code = application.wait(timeout=15)
    except Exception as error:  # preserve the concrete monitor failure in JSON
        monitor_error = f"{type(error).__name__}: {error}"
        monitor_traceback = traceback.format_exc()
        if application is not None:
            stop_process(application)
            app_return_code = application.returncode
    finally:
        if tegrastats is not None:
            stop_process(tegrastats)
        process_csv_file.close()
        application_file.close()
        tegrastats_file.close()

    finished_at = datetime.now(timezone.utc)
    summary = {
        "result": "PASS" if app_return_code == 0 and not safety_stop and monitor_error is None else "FAIL",
        "command": command,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": time.monotonic() - start_monotonic,
        "sample_interval_seconds": args.interval_seconds,
        "sample_count": sample_count,
        "application_return_code": app_return_code,
        "safety_stop": safety_stop,
        "temperature_stop_c": args.temperature_stop_c,
        "max_observed_temperature_c": (
            max_observed_temperature
            if max_observed_temperature == max_observed_temperature
            else None
        ),
        "monitor_error": monitor_error,
        "monitor_traceback": locals().get("monitor_traceback"),
    }
    monitor_summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blank() -> np.ndarray:
    return np.zeros((7, 8400), dtype=np.float32)


def set_candidate(raw, index, cx, cy, width, height, scores):
    raw[0:4, index] = np.asarray([cx, cy, width, height], dtype=np.float32)
    raw[4:7, index] = np.asarray(scores, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    fixtures = {}
    fixtures["zero"] = blank()

    single = blank()
    set_candidate(single, 17, 320, 320, 100, 50, [0.1, 0.8, 0.2])
    fixtures["single"] = single

    all_valid = blank()
    all_valid[0, :] = 320
    all_valid[1, :] = 320
    all_valid[2, :] = 10
    all_valid[3, :] = 10
    all_valid[4, :] = 0.3
    fixtures["all_8400"] = all_valid

    boundary = blank()
    set_candidate(boundary, 0, 10, 10, 40, 40, [0.1, 0.2, 0.25])
    set_candidate(boundary, 1, 630, 630, 40, 40, [0.7, 0.7, 0.7])
    set_candidate(boundary, 2, 320, 320, 10, 10, [np.inf, 0.9, 0.1])
    set_candidate(boundary, 3, np.nan, 320, 10, 10, [0.9, 0.1, 0.1])
    set_candidate(boundary, 4, 320, 320, 0, 10, [0.9, 0.1, 0.1])
    set_candidate(boundary, 5, -20, 320, 10, 10, [0.9, 0.1, 0.1])
    set_candidate(boundary, 6, 320, 320, 10, 10, [0.8, np.nan, 0.1])
    fixtures["boundary_invalid"] = boundary

    manifest = {"status": "PASS", "fixtures": {}}
    for name, raw in fixtures.items():
        path = args.output_dir / f"{name}.bin"
        raw.tofile(path)
        manifest["fixtures"][name] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

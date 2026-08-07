#!/usr/bin/env python3
"""Create deterministic non-square Exp10 preprocessing fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.shape != (640, 640, 3):
        raise RuntimeError(f"expected 640x640 source, got {image.shape}")

    wide = image[140:500, :].copy()
    tall = image[:, 140:500].copy()
    fixtures = {
        "square": image,
        "wide": wide,
        "tall": tall,
        "hd_wide": cv2.resize(wide, (1280, 720), interpolation=cv2.INTER_LINEAR),
        "small_tall": cv2.resize(tall, (240, 480), interpolation=cv2.INTER_LINEAR),
    }
    records = []
    for name, fixture in fixtures.items():
        path = output_dir / f"{name}.png"
        if not cv2.imwrite(str(path), fixture, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"failed to write {path}")
        records.append(
            {
                "name": name,
                "path": str(path),
                "height": int(fixture.shape[0]),
                "width": int(fixture.shape[1]),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "result": "PASS",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "generation": {
            "wide": "source[140:500, :]",
            "tall": "source[:, 140:500]",
            "hd_wide": "resize(wide, 1280x720, INTER_LINEAR)",
            "small_tall": "resize(tall, 240x480, INTER_LINEAR)",
        },
        "fixtures": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"result=PASS fixtures={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

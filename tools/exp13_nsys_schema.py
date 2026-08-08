#!/usr/bin/env python3
"""Inspect an Nsight Systems SQLite export without assuming a tool version."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    database = args.sqlite.resolve()
    if not database.is_file():
        raise FileNotFoundError(database)

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    tables: dict[str, dict[str, object]] = {}
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    for name in names:
        escaped = name.replace('"', '""')
        columns = [
            {"index": row[0], "name": row[1], "type": row[2]}
            for row in connection.execute(f'PRAGMA table_info("{escaped}")')
        ]
        row_count = connection.execute(
            f'SELECT COUNT(*) FROM "{escaped}"'
        ).fetchone()[0]
        tables[name] = {"columns": columns, "row_count": row_count}

    result = {
        "result": "PASS",
        "sqlite": str(database),
        "tables": tables,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

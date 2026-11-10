# SPDX-License-Identifier: Apache-2.0
"""JSON-lines files, the format of every per-run record."""

from __future__ import annotations

import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    """Records in `path`; an empty list if the file doesn't exist."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path: Path, records: list[dict]) -> str:
    """Write records and return the file body."""
    body = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    path.write_text(body)
    return body

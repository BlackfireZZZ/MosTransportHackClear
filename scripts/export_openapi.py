#!/usr/bin/env python3
"""Export the FastAPI OpenAPI document deterministically or detect contract drift."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "contracts" / "openapi.json"


def render_openapi() -> str:
    """Load the application without starting a server and render a stable snapshot."""
    os.environ["APP_NAME"] = "TramFlow API"
    os.environ["API_V1_PREFIX"] = "/api/v1"
    sys.path.insert(0, str(BACKEND_ROOT))
    from app.main import app  # noqa: PLC0415

    return json.dumps(app.openapi(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="snapshot path (default: contracts/openapi.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when the committed snapshot differs from FastAPI",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    rendered = render_openapi()

    if args.check:
        if not output.exists():
            print(f"OpenAPI snapshot is missing: {output}", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != rendered:
            print(
                "OpenAPI snapshot is stale. Run `uv run --package tramflow-backend "
                "python scripts/export_openapi.py`, then regenerate frontend types.",
                file=sys.stderr,
            )
            return 1
        print(f"OpenAPI snapshot is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote OpenAPI snapshot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""`tramflow-kudago` — collect, normalise and publish a KudaGo delivery directory.

Fetching and normalising are separate subcommands over the same `raw/` tree. The
acceptance criterion that the same raw input normalises to byte-identical output
cannot be checked at all if the only way to run the normaliser is to re-fetch.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

from tramflow_ml import __version__

from .delivery import DeliveryError, build_quality_report, publish, utc_now
from .fetch import collect_months, live_client
from .normalize import NormalizeConfig, NormalizeError, normalize
from .records import RunCounts, RunScope, RunStatus

MOSCOW_MIDNIGHT = "T00:00:00+03:00"


def _month_bounds(month: str) -> tuple[str, str]:
    """The half-open ``[first, next_first)`` bounds of one ``YYYY-MM``."""
    year, index = (int(part) for part in month.split("-"))
    end_year, end_month = (year + 1, 1) if index == 12 else (year, index + 1)
    return (
        f"{year:04d}-{index:02d}-01{MOSCOW_MIDNIGHT}",
        f"{end_year:04d}-{end_month:02d}-01{MOSCOW_MIDNIGHT}",
    )


def _buffer_bounds(target_from: str, target_to: str) -> tuple[str, str]:
    """One day either side, so a session straddling a boundary is not cut in half."""
    start = datetime.fromisoformat(target_from) - timedelta(days=1)
    end = datetime.fromisoformat(target_to) + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def _periods(months: Sequence[str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """One target span and one buffered span per collected month, never one hull.

    Disjoint months must stay disjoint: spanning January to September would place
    eight unfetched months inside the window and in the coverage denominator.
    """
    targets = [_month_bounds(month) for month in sorted(months)]
    return targets, [_buffer_bounds(start, end) for start, end in targets]


def _run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    months: list[str] = list(args.month)
    targets, windows = _periods(months)

    summary = collect_months(out, months, client=live_client()) if args.fetch else None
    if summary is not None and args.fetch_only:
        print(json.dumps({"status": summary.status, "requests": summary.request_count}))
        return 0

    config = NormalizeConfig(input_dir=out, output_dir=out, windows=tuple(windows))
    report = normalize(config)

    fetch_view = (
        {
            "status": summary.status,
            "unresolved_request_ids": list(summary.unresolved_request_ids),
            "failures": [f.kind for f in summary.failures],
        }
        if summary is not None
        else {"status": "unknown", "unresolved_request_ids": [], "failures": []}
    )
    scope: RunScope = {
        "location": "msk",
        "movies_included": False,
        "category_filter": None,
        "months": months,
    }
    status: RunStatus = summary.status if summary is not None else "partial"

    counts: RunCounts = {
        "events": report["events"],
        "places": report["places"],
        "occurrences": report["occurrences"],
        "observations": report["observations"],
        "quarantine": report["quarantined"],
        "requests": report["requests_total"],
    }
    quality = build_quality_report(
        out, scope=scope, status=status, normalize_report=report, fetch_summary=fetch_view
    )
    manifest = publish(
        out,
        quality_report=quality,
        manifest_kwargs={
            "run_id": (summary.run_id if summary is not None else report["run_ids"][0]),
            "owner": args.owner,
            "created_at": utc_now(),
            "scope": scope,
            "status": status,
            "targets": [list(span) for span in targets],
            "windows": [list(span) for span in windows],
            "collector_version": __version__,
            "normalizer_version": report["normalizer_version"],
            "normalizer_config_sha256": report["normalizer_config_sha256"],
            "counts": counts,
            "unresolved_failed_request_ids": list(fetch_view["unresolved_request_ids"]),
        },
    )
    print(json.dumps({"status": manifest["status"], "counts": counts}, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tramflow-kudago", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="fetch, normalise and publish a delivery directory")
    collect.add_argument("--month", action="append", required=True, help="YYYY-MM, repeatable")
    collect.add_argument("--out", required=True)
    collect.add_argument("--owner", required=True, help="who is answerable for this run")
    collect.add_argument("--no-fetch", dest="fetch", action="store_false", default=True)
    collect.add_argument("--fetch-only", action="store_true")
    collect.set_defaults(handler=_run)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (DeliveryError, NormalizeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

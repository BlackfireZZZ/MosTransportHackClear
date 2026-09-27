import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from tramflow_ml import __version__
from tramflow_ml.evaluation import evaluate, load_cases
from tramflow_ml.ingestion import (
    ADAPTERS,
    DEFAULT_CHUNK_SIZE,
    IngestionConfig,
    IngestionError,
    ReconciliationError,
    ingest,
)
from tramflow_ml.intake import (
    IntakeError,
    IntakeRequest,
    InvariantError,
    SchemaError,
    load_profile,
    profile_sample,
    render,
)
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset


def _add_ingest_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    ingest_parser = subparsers.add_parser(
        "ingest", help="normalize fixture event streams in bounded, resumable chunks"
    )
    ingest_parser.add_argument("--input", type=Path, required=True)
    ingest_parser.add_argument("--output", type=Path, required=True)
    ingest_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ingest_parser.add_argument("--format", choices=("jsonl", "csv"), default="jsonl")
    ingest_parser.add_argument("--adapter", choices=sorted(ADAPTERS), default="synthetic")


def _run_ingest(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit 2 for usage/input problems, 1 when counts do not reconcile, 0 on success."""
    started = time.monotonic()
    try:
        config = IngestionConfig(
            input=args.input,
            output=args.output,
            chunk_size=args.chunk_size,
            source_format=args.format,
            adapter=ADAPTERS[args.adapter],
        )
        manifest = ingest(config)
    except ReconciliationError as error:
        print(f"reconciliation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except (ValueError, OSError, IngestionError) as error:
        parser.error(str(error))
    elapsed = time.monotonic() - started
    rows = sum(report["counts"]["input_rows"] for report in manifest["streams"].values())
    print(
        json.dumps(
            {
                "output_hash": manifest["output_hash"],
                "streams": {
                    name: report["counts"] for name, report in manifest["streams"].items()
                },
                "elapsed_seconds": round(elapsed, 3),
                "rows_per_second": round(rows / elapsed) if elapsed else None,
            },
            sort_keys=True,
        )
    )


def _add_intake_parser(subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    intake_parser = subparsers.add_parser(
        "intake", help="profile a data sample read-only; the sample is never modified"
    )
    intake_parser.add_argument("--input", type=Path, required=True)
    intake_parser.add_argument("--profile", type=Path)
    intake_parser.add_argument("--catalog", type=Path)
    intake_parser.add_argument("--output", type=Path)


def _run_intake(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit 2 for usage, input or schema problems, 1 for a failed invariant, 0 on success.

    A schema diagnosis skips ``parser.error`` so the argparse usage block cannot bury
    the checklist an operator has to act on; the exit code is the same 2.
    """
    try:
        _refuse_nested_output(args)
        profile = load_profile(args.profile) if args.profile else None
        report = profile_sample(
            IntakeRequest(source=args.input, profile=profile, catalog=args.catalog)
        )
    except InvariantError as error:
        print(f"intake invariant failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except SchemaError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
    except (IntakeError, OSError) as error:
        parser.error(str(error))
    serialized = render(report)
    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        except OSError as error:
            parser.error(f"cannot write --output {args.output}: {error}")
    print(serialized, end="")


def _refuse_nested_output(args: argparse.Namespace) -> None:
    """The sample directory stays byte-identical, so the report may not land inside it."""
    if args.output and args.output.resolve().is_relative_to(args.input.resolve()):
        raise IntakeError(
            f"--output {args.output} is inside --input {args.input}; intake never writes "
            "into the sample it profiles"
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="tramflow-ml")
    subparsers = parser.add_subparsers(dest="command")
    evaluate_parser = subparsers.add_parser("evaluate", help="run the golden forecast suite")
    evaluate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("ml/evals/golden_cases.json"),
    )
    evaluate_parser.add_argument("--output", type=Path)
    synthetic_parser = subparsers.add_parser(
        "generate-synthetic", help="generate versioned synthetic validation/telemetry fixtures"
    )
    synthetic_parser.add_argument("--output", type=Path, required=True)
    synthetic_parser.add_argument("--mode", choices=("tiny", "million"), default="tiny")
    synthetic_parser.add_argument("--events", type=int)
    synthetic_parser.add_argument("--seed", type=int, default=42)
    synthetic_parser.add_argument("--start-date", type=date.fromisoformat, default=date(2024, 1, 1))
    synthetic_parser.add_argument("--end-date", type=date.fromisoformat, default=date(2026, 1, 1))
    synthetic_parser.add_argument("--duplicate-every", type=int, default=17)
    synthetic_parser.add_argument("--late-every", type=int, default=11)
    synthetic_parser.add_argument("--telemetry-every", type=int, default=5)
    synthetic_parser.add_argument("--gap-every-days", type=int, default=13)
    _add_ingest_parser(subparsers)
    _add_intake_parser(subparsers)
    args = parser.parse_args()

    if args.command == "ingest":
        _run_ingest(parser, args)
        return

    if args.command == "intake":
        _run_intake(parser, args)
        return

    if args.command == "generate-synthetic":
        try:
            config = SyntheticConfig(
                start=args.start_date,
                end=args.end_date,
                events=args.events
                if args.events is not None
                else (1_000_000 if args.mode == "million" else 64),
                seed=args.seed,
                duplicate_every=args.duplicate_every,
                late_every=args.late_every,
                telemetry_every=args.telemetry_every,
                gap_every_days=args.gap_every_days,
            )
            result = generate_dataset(config, args.output)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {
                    "dataset_id": result["manifest"]["dataset_id"],
                    "source_hash": result["manifest"]["source_hash"],
                    "counts": result["generation"]["counts"],
                },
                sort_keys=True,
            )
        )
        return

    if args.command != "evaluate":
        print(f"TramFlow ML {__version__}: workspace is ready")
        return

    summary = evaluate(load_cases(args.dataset))
    serialized = summary.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if not summary.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

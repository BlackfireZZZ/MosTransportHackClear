import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .alignment import Anchor, DecodeConfig, DelayScenario, Pattern, decode
from .audit import audit, verify_run, write_json
from .catalog import catalog_inventory
from .dataset import export_dataset
from .partitions import prepare
from .pilot import pilot
from .real import RealConfig, reconstruct
from .real_export import export_real_dataset
from .real_stability import run_sensitivity
from .records import AuditConfig
from .simulation import evaluate_synthetic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline experimental boarding dataset; no publication"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("audit")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--config", type=Path)
    for name in ["detect", "export"]:
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        sub.add_argument("--out", type=Path, required=True)
    v = commands.add_parser("verify")
    v.add_argument("--run", type=Path, required=True)
    c = commands.add_parser("catalog")
    c.add_argument("--source", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    e = commands.add_parser("evaluate")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--seed", type=int, default=20260926)
    d = commands.add_parser("decode")
    d.add_argument("--input", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    prep = commands.add_parser("prepare-real")
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    real = commands.add_parser("reconstruct")
    for option in ["partitions", "histories", "graph", "timetables", "config", "out"]:
        real.add_argument("--" + option, type=Path, required=True)
    real.add_argument("--workers", type=int, default=1)
    real_export = commands.add_parser("export-real")
    real_export.add_argument("--run", type=Path, required=True)
    real_export.add_argument("--out", type=Path, required=True)
    real_export.add_argument("--service-policy", type=Path)
    sensitivity = commands.add_parser("sensitivity-real")
    for option in ["partitions", "histories", "graph", "timetables", "out"]:
        sensitivity.add_argument("--" + option, type=Path, required=True)
    sensitivity.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args(argv)
    try:
        result: Any
        if args.command == "audit":
            config = (
                AuditConfig.model_validate_json(args.config.read_text())
                if args.config
                else AuditConfig()
            )
            result = audit(args.source, args.out, config)
        elif args.command == "prepare-real":
            result = prepare(args.run, args.out)
        elif args.command == "reconstruct":
            result = reconstruct(
                args.partitions,
                args.histories,
                args.graph,
                args.timetables,
                args.out,
                RealConfig(**json.loads(args.config.read_text())),
                workers=args.workers,
            )
        elif args.command == "export-real":
            result = export_real_dataset(args.run, args.out, service_policy=args.service_policy)
        elif args.command == "sensitivity-real":
            if args.out.exists():
                raise ValueError("output already exists")
            result = run_sensitivity(
                args.partitions,
                args.histories,
                args.graph,
                args.timetables,
                args.out,
                seed=args.seed,
            )
        elif args.command == "verify":
            result = verify_run(args.run)
        elif args.command == "detect":
            result = pilot(args.run, args.out)
        elif args.command == "export":
            result = export_dataset(args.run, args.out)
        else:
            if args.out.exists():
                raise ValueError("output already exists")
            if args.command == "catalog":
                result = catalog_inventory(args.source)
            elif args.command == "evaluate":
                result = evaluate_synthetic(seed=args.seed)
            else:
                payload = json.loads(args.input.read_text())
                if set(payload) - {"timestamps", "patterns", "scenarios", "anchors", "config"}:
                    raise ValueError("unknown decode input fields")
                result = asdict(
                    decode(
                        payload["timestamps"],
                        tuple(Pattern(**p) for p in payload["patterns"]),
                        scenarios=tuple(DelayScenario(**s) for s in payload.get("scenarios", [{}])),
                        anchors=tuple(Anchor(**a) for a in payload.get("anchors", [])),
                        config=DecodeConfig(**payload.get("config", {})),
                    )
                )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, result)
        print(
            json.dumps(
                {
                    "command": args.command,
                    "status": "completed",
                    "real_stop_accuracy": "unverified",
                },
                sort_keys=True,
            )
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {"command": args.command, "status": "failed", "error_type": type(error).__name__}
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

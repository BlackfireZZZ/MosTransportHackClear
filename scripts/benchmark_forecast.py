"""Validate an isolated forecast stack, then measure a fixed HTTP workload."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "docs/benchmark/cases.json"
COMPOSE = ("docker", "compose", "-f", "compose.yaml", "-f", "compose.benchmark.yaml")
EXPECTED_CPUS = 2.0
EXPECTED_MEMORY = 4 * 1024**3


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture() -> dict:
    data = json.loads(CASES.read_text())
    assert data["schema_version"] == 1
    assert 1 <= data["requests"] <= 10000
    assert 1 <= data["warmup"] <= 100
    assert 1 <= data["concurrency"] <= 64
    ids = [case["id"] for case in data["cases"]]
    assert len(ids) == len(set(ids))
    assert {"get_day", "get_month", "post_approved_day", "post_approved_month", "post_approved_year", "post_stop_day", "post_stop_month", "post_stop_year", "post_stop_filter", "post_invalid_date"} <= set(ids)
    for case in data["cases"]:
        assert case["path"].startswith("/api/v1/") and "//" not in case["path"]
        assert case["method"] in {"GET", "POST"}
        assert case["expect"] in {"published", "planning", "error"}
        assert (case["method"] == "POST") == ("payload" in case)
    return data


def stack_conditions() -> dict:
    if os.environ.get("COMPOSE_PROJECT_NAME") != "tramflow-benchmark":
        raise RuntimeError("COMPOSE_PROJECT_NAME must be tramflow-benchmark")
    config = command(*COMPOSE, "config", "--format", "json")
    rendered = json.loads(config)
    if rendered["name"] != "tramflow-benchmark":
        raise RuntimeError("unexpected Compose project name")
    backend = rendered["services"]["backend"]
    if float(backend.get("cpus", 0)) != EXPECTED_CPUS or int(backend.get("mem_limit", 0)) != EXPECTED_MEMORY or int(backend.get("memswap_limit", -1)) != EXPECTED_MEMORY:
        raise RuntimeError("backend Compose CPU/memory/no-swap limits differ from 2 CPU / 4 GiB")
    published = {str(port["published"]) for service in rendered["services"].values() for port in service.get("ports", [])}
    if not {"15432", "18000", "18080"} <= published or {"5432", "8000", "8080"} & published:
        raise RuntimeError("benchmark ports are not isolated")
    cid = command(*COMPOSE, "ps", "-q", "backend")
    if not cid:
        raise RuntimeError("benchmark backend is not running")
    inspect = json.loads(command("docker", "inspect", cid))[0]
    host = inspect["HostConfig"]
    if host["NanoCpus"] != 2_000_000_000 or host["Memory"] != EXPECTED_MEMORY or host["MemorySwap"] != EXPECTED_MEMORY:
        raise RuntimeError("Docker runtime limits differ from benchmark profile")
    cgroup = command("docker", "exec", cid, "sh", "-c", "cat /sys/fs/cgroup/cpu.max; cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/memory.swap.max").splitlines()
    if len(cgroup) != 3 or cgroup[0].split()[0] == "max" or int(cgroup[0].split()[0]) / int(cgroup[0].split()[1]) != EXPECTED_CPUS or int(cgroup[1]) != EXPECTED_MEMORY or int(cgroup[2]) != 0:
        raise RuntimeError(f"actual cgroup limits differ: {cgroup}")
    image = json.loads(command("docker", "image", "inspect", inspect["Image"]))[0]
    dbid = command(*COMPOSE, "ps", "-q", "db")
    if not dbid:
        raise RuntimeError("benchmark PostgreSQL is not running")
    dbinspect = json.loads(command("docker", "inspect", dbid))[0]
    workers = sum("uvicorn" in line for line in command("docker", "top", cid).splitlines()[1:])
    if workers < 1:
        raise RuntimeError("backend worker count is zero")
    postgres_version = command("docker", "exec", dbid, "postgres", "--version")
    return {
        "backend_cpus": EXPECTED_CPUS,
        "backend_memory_bytes": EXPECTED_MEMORY,
        "backend_swap_bytes": 0,
        "cgroup_cpu_max": cgroup[0],
        "cgroup_memory_max": cgroup[1],
        "cgroup_memory_swap_max": cgroup[2],
        "workers": workers,
        "backend_image": image["Id"],
        "postgres_image": dbinspect["Image"],
        "postgres_version": postgres_version,
        "postgres_scope": "separate container; not included in backend CPU/memory limits",
        "postgres_runtime_limits": {"nano_cpus": dbinspect["HostConfig"]["NanoCpus"], "memory_bytes": dbinspect["HostConfig"]["Memory"]},
        "host_cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor()),
        "host_arch": platform.machine(),
        "host_platform": platform.platform(),
        "git_sha": command("git", "rev-parse", "HEAD"),
        "compose_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "artifact_sha256": digest(CASES),
        "forecast_manifest_sha256": digest(ROOT / "ml/competition_submissions/current.json"),
        "approved_config_sha256": digest(ROOT / "ml/competition_submissions/approved.json"),
    }


def check_response(case: dict, response: httpx.Response) -> None:
    if case["expect"] == "error":
        if response.status_code != case["status"] or not response.json().get("detail"):
            raise ValueError("wrong error status or missing detail")
        return
    if response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}, expected 200")
    body = response.json()
    if body.get("horizon") != case["horizon"] or not body.get("points") or not body.get("model_version"):
        raise ValueError("wrong horizon, missing points or model version")
    if case["expect"] == "published":
        if not body.get("run") or not body.get("generated_at") or not body.get("route"):
            raise ValueError("published provenance missing")
        query = parse_qs(urlsplit(case["path"]).query)
        if "start" in query and "end" in query:
            start = datetime.fromisoformat(query["start"][0])
            end = datetime.fromisoformat(query["end"][0])
            if any(not start <= datetime.fromisoformat(point["timestamp"]) < end for point in body["points"]):
                raise ValueError("published interval filter was not applied")
    else:
        if body.get("forecast_mode") != case["mode"] or body.get("timezone") != "Europe/Moscow":
            raise ValueError("planning mode or timezone mismatch")
        if body.get("qualitative") != (case["horizon"] == "year"):
            raise ValueError("qualitative year marker mismatch")
        if case["horizon"] == "year" and len(body["points"]) != 12:
            raise ValueError("qualitative year must contain 12 monthly points")
        if case["id"] == "post_stop_filter" and body.get("stop_id") != case["payload"]["stop_id"]:
            raise ValueError("stop filter was not echoed")
        if case["id"] == "post_stop_filter":
            expected = case["payload"]["stop_id"]
            direction = case["payload"]["direction"]
            if body.get("direction") != direction or any(
                spatial["stop_id"] != expected or spatial["direction"] != direction
                for point in body["points"] for spatial in point["spatial"]
            ):
                raise ValueError("stop/direction filter was not applied")


async def request(client: httpx.AsyncClient, base: str, case: dict) -> httpx.Response:
    return await client.request(case["method"], base + case["path"], json=case.get("payload"))


async def functional_preflight(base: str, cases: list[dict]) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        for case in cases:
            check_response(case, await request(client, base, case))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def memory_bytes(value: str) -> int:
    amount, unit = re.match(r"([\d.]+)([KMGT]?i?B)", value).groups()
    scale = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
             "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    return int(float(amount) * scale[unit])


def public_result(raw: dict, template: dict, expected_ids: set[str], source_sha256: str) -> dict:
    runs = raw.get("runs", [])
    labels = template.get("required_cases", [])
    if raw.get("schema_version") != 1 or raw.get("status") != "measured":
        raise ValueError("only a completed measured result can be published")
    if {run.get("case_id") for run in runs} != expected_ids or len(runs) != len(expected_ids):
        raise ValueError("result does not cover the fixed benchmark cases exactly")
    if {item.get("id") for item in labels} != expected_ids or len(labels) != len(expected_ids):
        raise ValueError("public case labels do not match the fixed fixture")
    if any(run.get("errors") != 0 or run.get("conditions", {}).get("artifact_sha256") != digest(CASES) for run in runs):
        raise ValueError("result contains errors or a different fixture")
    return {"schema_version": 1, "status": "measured", "updated_at": raw["updated_at"],
            "source_sha256": source_sha256, "required_cases": labels, "runs": runs}


async def sample_backend(cid: str, samples: list[dict], stop: asyncio.Event) -> None:
    while not stop.is_set():
        raw = await asyncio.to_thread(command, "docker", "stats", "--no-stream", "--format", "{{json .}}", cid)
        row = json.loads(raw)
        samples.append({"cpu_percent": float(row["CPUPerc"].rstrip("%")),
                        "memory_bytes": memory_bytes(row["MemUsage"].split("/")[0].strip())})
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
        except asyncio.TimeoutError:
            pass


async def benchmark_case(base: str, case: dict, count: int, concurrency: int, warmup: int) -> dict:
    durations: list[float] = []
    codes: Counter[int] = Counter()
    failures: Counter[str] = Counter()
    semaphore = asyncio.Semaphore(concurrency)
    cid = command(*COMPOSE, "ps", "-q", "backend")
    samples: list[dict] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(warmup):
            check_response(case, await request(client, base, case))
        stop = asyncio.Event()
        sampler = asyncio.create_task(sample_backend(cid, samples, stop))
        async def one() -> None:
            async with semaphore:
                started = time.perf_counter()
                try:
                    response = await request(client, base, case)
                    codes[response.status_code] += 1
                    check_response(case, response)
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                    failures[type(error).__name__] += 1
                durations.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(count)))
        elapsed = time.perf_counter() - started
        stop.set()
        await sampler
    return {"case_id": case["id"], "requests": count, "warmup": warmup, "concurrency": concurrency,
            "elapsed_seconds": elapsed, "rps": (count - sum(failures.values())) / elapsed,
            "errors": sum(failures.values()), "failure_types": dict(failures),
            "http_status_counts": {str(k): v for k, v in codes.items()},
            "backend_resource_samples": samples,
            "backend_cpu_percent_sample_max": max(sample["cpu_percent"] for sample in samples),
            "backend_memory_bytes_sample_max": max(sample["memory_bytes"] for sample in samples),
            "latency_ms": {"p50": statistics.median(durations), "p95": percentile(durations, .95), "p99": percentile(durations, .99)},
            "latency_scope": "HTTP and JSON validation; client semaphore wait excluded"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["dry-run", "preflight", "run", "publish"])
    parser.add_argument("--base-url", default="http://localhost:18000")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    cases = fixture()
    if args.mode == "publish":
        if args.input is None:
            parser.error("publish requires --input")
        destination = ROOT / "frontend/public/benchmark-results.json"
        raw = json.loads(args.input.read_text())
        template = json.loads(destination.read_text())
        result = public_result(raw, template, {case["id"] for case in cases["cases"]}, digest(args.input))
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"public result: {destination}")
        return
    if args.mode == "dry-run":
        print(json.dumps({"status": "configuration_valid", "cases": [case["id"] for case in cases["cases"]], "requests_per_case": cases["requests"], "performance_metrics": None}, indent=2))
        return
    conditions = stack_conditions()
    asyncio.run(functional_preflight(args.base_url, cases["cases"]))
    if args.mode == "preflight":
        print(json.dumps({"status": "preflight_passed", "conditions": conditions, "performance_metrics": None}, indent=2))
        return
    if args.output is None:
        parser.error("run requires --output")
    if command("git", "status", "--porcelain"):
        raise RuntimeError("benchmark run requires a clean committed worktree")
    started = datetime.now(timezone.utc).isoformat()
    runs = asyncio.run(run_all(args.base_url, cases))
    finished = datetime.now(timezone.utc).isoformat()
    result = {"schema_version": 1, "status": "measured" if all(run["errors"] == 0 for run in runs) else "failed", "updated_at": started, "finished_at": finished,
              "conditions": conditions, "runs": [{**run, "conditions": conditions} for run in runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"raw result: {args.output}")
    if result["status"] != "measured":
        raise SystemExit(1)


async def run_all(base: str, cases: dict) -> list[dict]:
    return [await benchmark_case(base, case, cases["requests"], cases["concurrency"], cases["warmup"]) for case in cases["cases"]]


if __name__ == "__main__":
    main()

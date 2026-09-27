"""Run verification only against a fresh, owned Compose project."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("sql", "migration", "stack", "scored-upgrade"))
    args = parser.parse_args()
    project = f"tramflow-test-{uuid.uuid4().hex}"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("COMPOSE_")
    }
    env.update(
        COMPOSE_PROJECT_NAME=project,
        POSTGRES_DB="tramflow_test",
        POSTGRES_USER="tramflow_test",
        POSTGRES_PASSWORD="isolated_test_only",
        POSTGRES_PORT="127.0.0.1:",
        BACKEND_PORT="127.0.0.1:",
        FRONTEND_PORT="127.0.0.1:",
        DATABASE_URL="postgresql+asyncpg://tramflow_test:isolated_test_only@db:5432/tramflow_test",
    )
    compose = [
        "docker",
        "compose",
        "--env-file",
        "/dev/null",
        "-f",
        str(ROOT / "compose.yaml"),
        "--project-name",
        project,
    ]

    def run(
        command: list[str], *, capture: bool = False
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )

    def port(service: str, target: int) -> str:
        address = run(
            [*compose, "port", service, str(target)], capture=True
        ).stdout.strip()
        host, value = address.rsplit(":", 1)
        if host != "127.0.0.1" or not value.isdigit():
            raise RuntimeError(f"Unexpected test binding: {address}")
        return value

    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    print(f"Owned test project: {project}", flush=True)
    status = 0
    try:
        run([*compose, "config", "--quiet"])
        if args.mode == "stack":
            run([*compose, "up", "--build", "--wait", "--wait-timeout", "180"])
            env.update(
                BASE_URL=f"http://127.0.0.1:{port('backend', 8000)}",
                FRONTEND_URL=f"http://127.0.0.1:{port('frontend', 80)}",
            )
            run(["./scripts/smoke.sh"])
        else:
            run([*compose, "up", "-d", "--wait", "--wait-timeout", "90", "db"])
            env["DATABASE_URL"] = (
                "postgresql+asyncpg://tramflow_test:isolated_test_only@"
                f"127.0.0.1:{port('db', 5432)}/tramflow_test"
            )
            uv = ["uv", "run", "--locked", "--package", "tramflow-backend", "--extra", "dev"]
            alembic = [*uv, "alembic", "-c", "backend/alembic.ini"]
            if args.mode == "scored-upgrade":
                run([*alembic, "upgrade", "20260927_0001"])
                run([*uv, "python", "scripts/verify-scored-upgrade.py", "prepare"])
            run([*alembic, "upgrade", "head"])
            run([*alembic, "check"])
            if args.mode == "scored-upgrade":
                run([*uv, "python", "scripts/verify-scored-upgrade.py", "verify"])
            if args.mode == "sql":
                env["TRAMFLOW_TEST_DATABASE_URL"] = env["DATABASE_URL"]
                env["TRAMFLOW_TEST_PROJECT"] = project
                run([*uv, "pytest", "backend/tests/integration", "-v"])
    except subprocess.CalledProcessError as error:
        status = error.returncode or 1
    except SystemExit as error:
        status = int(error.code or 1)
    except (OSError, RuntimeError) as error:
        print(f"Verification failed: {error}", flush=True)
        status = 1
    finally:
        # A second cancellation during diagnostics must not bypass resource cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if status:
            try:
                subprocess.run(
                    [*compose, "logs", "--no-color", "--tail=80"],
                    cwd=ROOT,
                    env=env,
                    check=False,
                    timeout=15,
                )
            except (OSError, subprocess.TimeoutExpired):
                print("Test diagnostics unavailable", flush=True)
        try:
            cleanup = subprocess.run(
                [*compose, "down", "--volumes", "--remove-orphans", "--timeout", "10"],
                cwd=ROOT,
                env=env,
                check=False,
                timeout=60,
            )
            cleanup_status = cleanup.returncode
        except (OSError, subprocess.TimeoutExpired):
            cleanup_status = 1
        if cleanup_status:
            print(f"Cleanup failed; retained project: {project}", flush=True)
            status = status or cleanup_status
    return status


if __name__ == "__main__":
    raise SystemExit(main())

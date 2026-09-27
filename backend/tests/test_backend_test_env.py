import importlib.util
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "backend-test-env.py"
    spec = importlib.util.spec_from_file_location("backend_test_env", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(path), "sql"])
    monkeypatch.setattr(module.signal, "signal", Mock())
    return module


@pytest.fixture
def processes(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[list[str], dict]]:
    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        options = dict(kwargs)
        options["env"] = dict(kwargs["env"])
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="127.0.0.1:49173\n")

    monkeypatch.setattr(runner.subprocess, "run", Mock(side_effect=run))
    return calls


def assert_owned_cleanup(calls: list[tuple[list[str], dict]]) -> None:
    cleanups = [command for command, _ in calls if "down" in command]
    assert len(cleanups) == 1
    assert cleanups[0][-5:] == ["down", "--volumes", "--remove-orphans", "--timeout", "10"]
    projects = {
        command[command.index("--project-name") + 1]
        for command, _ in calls
        if command[:2] == ["docker", "compose"]
    }
    assert len(projects) == 1
    assert next(iter(projects)).startswith("tramflow-test-")


def test_sql_success_migrates_before_tests_then_removes_owned_resources(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
) -> None:
    assert runner.main() == 0
    commands = [command for command, _ in processes]
    upgrade = next(i for i, command in enumerate(commands) if "upgrade" in command)
    check = next(i for i, command in enumerate(commands) if "check" in command)
    tests = next(i for i, command in enumerate(commands) if "pytest" in command)
    assert upgrade < check < tests < len(commands) - 1
    assert commands[tests][-3:] == ["pytest", "backend/tests/integration", "-v"]
    assert_owned_cleanup(processes)
    assert not any("logs" in command for command in commands)


@pytest.mark.parametrize("stage", ["config", "up", "port", "upgrade", "check", "pytest"])
def test_failed_stage_preserves_exit_status_and_cleans_up(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
    stage: str,
) -> None:
    successful_run = runner.subprocess.run.side_effect

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = successful_run(command, **kwargs)
        if stage in command:
            raise subprocess.CalledProcessError(37, command)
        return result

    runner.subprocess.run.side_effect = fail
    assert runner.main() == 37
    assert_owned_cleanup(processes)
    assert processes[-2][0][-3:] == ["logs", "--no-color", "--tail=80"]
    if stage != "pytest":
        assert not any("pytest" in command for command, _ in processes)


def test_sigterm_preserves_signal_status_and_cleans_up(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
) -> None:
    successful_run = runner.subprocess.run.side_effect

    def terminate(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = successful_run(command, **kwargs)
        if "up" in command:
            handler = next(
                call.args[1]
                for call in runner.signal.signal.call_args_list
                if call.args[0] == signal.SIGTERM
            )
            handler(signal.SIGTERM, None)
        return result

    runner.subprocess.run.side_effect = terminate
    assert runner.main() == 128 + signal.SIGTERM
    assert_owned_cleanup(processes)
    runner.signal.signal.assert_any_call(signal.SIGTERM, signal.SIG_IGN)
    runner.signal.signal.assert_any_call(signal.SIGINT, signal.SIG_IGN)


def test_inherited_database_and_compose_configuration_cannot_select_working_stack(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "DATABASE_URL": "postgresql+asyncpg://live:secret@production:5432/live",
        "TRAMFLOW_TEST_DATABASE_URL": "postgresql+asyncpg://live:secret@production:5432/live",
        "TRAMFLOW_TEST_PROJECT": "production",
        "COMPOSE_FILE": "/tmp/production-compose.yaml",
        "COMPOSE_PROJECT_NAME": "production",
        "COMPOSE_PROFILES": "production",
        "COMPOSE_ENV_FILES": "/tmp/production.env",
        "POSTGRES_DB": "live",
        "POSTGRES_USER": "live",
        "POSTGRES_PASSWORD": "secret",
        "POSTGRES_PORT": "5432",
        "BACKEND_PORT": "8000",
        "FRONTEND_PORT": "8080",
    }.items():
        monkeypatch.setenv(name, value)
    assert runner.main() == 0
    for command, options in processes:
        env = options["env"]
        assert "production" not in env["DATABASE_URL"]
        assert env["COMPOSE_PROJECT_NAME"].startswith("tramflow-test-")
        assert "COMPOSE_FILE" not in env
        assert "COMPOSE_PROFILES" not in env
        assert "COMPOSE_ENV_FILES" not in env
        assert env["POSTGRES_DB"] == env["POSTGRES_USER"] == "tramflow_test"
        assert env["POSTGRES_PASSWORD"] == "isolated_test_only"
        for port in ("POSTGRES_PORT", "BACKEND_PORT", "FRONTEND_PORT"):
            assert env[port] == "127.0.0.1:"
        if command[:2] == ["docker", "compose"]:
            assert command[command.index("-f") + 1] == str(runner.ROOT / "compose.yaml")
            assert command[command.index("--env-file") + 1] == "/dev/null"
        if "pytest" in command:
            assert env["TRAMFLOW_TEST_DATABASE_URL"] == env["DATABASE_URL"]
            assert "127.0.0.1:49173/tramflow_test" in env["DATABASE_URL"]
            assert env["TRAMFLOW_TEST_PROJECT"] == env["COMPOSE_PROJECT_NAME"]
    assert_owned_cleanup(processes)


def test_each_invocation_owns_a_unique_project(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
) -> None:
    assert runner.main() == 0
    assert runner.main() == 0
    projects = [
        options["env"]["COMPOSE_PROJECT_NAME"]
        for command, options in processes
        if "down" in command
    ]
    assert len(projects) == len(set(projects)) == 2


@pytest.mark.parametrize("failure", ["returncode", "timeout", "oserror"])
@pytest.mark.parametrize("test_status", [0, 37])
def test_cleanup_failure_is_nonzero_without_hiding_test_failure(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
    failure: str,
    test_status: int,
) -> None:
    successful_run = runner.subprocess.run.side_effect

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = successful_run(command, **kwargs)
        if "pytest" in command and test_status:
            raise subprocess.CalledProcessError(test_status, command)
        if "down" in command:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 60)
            if failure == "oserror":
                raise OSError("Docker unavailable")
            return subprocess.CompletedProcess(command, 19)
        return result

    runner.subprocess.run.side_effect = fail
    expected_cleanup_status = 19 if failure == "returncode" else 1
    assert runner.main() == (test_status or expected_cleanup_status)
    assert_owned_cleanup(processes)


@pytest.mark.parametrize("failure", ["timeout", "oserror"])
def test_failed_diagnostics_cannot_bypass_cleanup(
    runner: ModuleType,
    processes: list[tuple[list[str], dict]],
    failure: str,
) -> None:
    successful_run = runner.subprocess.run.side_effect

    def fail(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        result = successful_run(command, **kwargs)
        if "up" in command:
            raise subprocess.CalledProcessError(37, command)
        if "logs" in command:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 15)
            raise OSError("Docker unavailable")
        return result

    runner.subprocess.run.side_effect = fail
    assert runner.main() == 37
    assert_owned_cleanup(processes)

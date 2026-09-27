import json
from pathlib import Path
from typing import Any

import pytest

from tramflow_ml.boarding import real
from tramflow_ml.boarding.audit import write_json
from tramflow_ml.boarding.real import RealConfig, reconstruct


def interrupted_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ...]:
    partitions = tmp_path / "partitions"
    histories = tmp_path / "histories"
    partitions.mkdir()
    histories.mkdir()
    write_json(partitions / "manifest.json", {"fixture": "partition identity"})
    write_json(histories / "history-manifest.json", [])
    graph, timetables = tmp_path / "graph.json", tmp_path / "timetables.json"
    write_json(graph, {})
    write_json(timetables, {})
    config = RealConfig(("2025-01-01", "2025-01-02"))
    calls: list[str] = []
    state = {"interrupt": True}

    def decode_day(
        partitions: Path,
        histories: Path,
        graph: Path,
        timetables: Path,
        day: str,
        config: RealConfig,
        out: Path,
    ) -> dict[str, Any]:
        calls.append(day)
        if day == "2025-01-02" and state["interrupt"]:
            raise ValueError("simulated interruption")
        out.mkdir(parents=True, exist_ok=True)
        (out / "data.txt").write_text(day)
        report = {
            "day": day,
            "source_success": 2,
            "assigned_weak": 0,
            "ambiguous": 2,
            "unassigned": 0,
            "candidate_stop_rows": 2,
            "wall_seconds": 0,
        }
        write_json(out / "evaluation.json", report)
        return report

    monkeypatch.setattr(real, "reconstruct_day", decode_day)
    args = (partitions, histories, graph, timetables, tmp_path / "out", config)
    with pytest.raises(ValueError, match="simulated interruption"):
        reconstruct(*args)
    assert not (args[4] / "manifest.json").exists()
    assert (args[4] / "2025-01-01" / "receipt.json").exists()
    state["interrupt"] = False
    return args, calls


def test_resume_reuses_verified_day_and_refuses_completed_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, calls = interrupted_run(tmp_path, monkeypatch)
    result = reconstruct(*args)
    assert calls == ["2025-01-01", "2025-01-02", "2025-01-02"]
    assert result["source_success"] == result["ambiguous"] == 4
    assert result["complete"] is True
    assert len(result["day_receipts"]) == 2
    with pytest.raises(ValueError, match="complete"):
        reconstruct(*args)


@pytest.mark.parametrize("mutation", ["source", "config", "completed_day"])
def test_resume_rejects_changed_inputs_or_completed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    args, calls = interrupted_run(tmp_path, monkeypatch)
    if mutation == "source":
        args[2].write_text(json.dumps({"changed": True}))
    elif mutation == "config":
        args = (*args[:-1], RealConfig(args[-1].dates, seed=7))
    else:
        (args[4] / "2025-01-01" / "data.txt").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        reconstruct(*args)
    assert calls == ["2025-01-01", "2025-01-02"]
    assert not (args[4] / "manifest.json").exists()

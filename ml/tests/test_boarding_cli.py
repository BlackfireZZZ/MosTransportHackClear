import json
from pathlib import Path

from test_boarding_audit import archive

from tramflow_ml.boarding.__main__ import main


def test_vertical_cli(tmp_path: Path) -> None:
    src = archive(tmp_path / "input.zip")
    run = tmp_path / "run"
    assert main(["audit", "--source", str(src), "--out", str(run)]) == 0
    assert main(["verify", "--run", str(run)]) == 0
    assert main(["export", "--run", str(run), "--out", str(tmp_path / "data")]) == 0
    assert main(["audit", "--source", str(src), "--out", str(run)]) == 2
    assert main(["export", "--run", str(tmp_path / "missing"), "--out", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad").exists()


def test_decode_refuses_uncertified_absolute_labels(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "timestamps": [0, 30],
                "patterns": [
                    {
                        "pattern_id": "p",
                        "direction": "out",
                        "stop_ids": ["a", "b", "c"],
                        "travel_seconds": [30, 30],
                    }
                ],
            }
        )
    )
    out = tmp_path / "out.json"
    assert main(["decode", "--input", str(source), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["stop_ids"] == [None, None]

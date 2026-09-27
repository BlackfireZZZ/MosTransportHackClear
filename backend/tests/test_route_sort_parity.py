"""The backend and the extraction script must order route refs identically.

The exporter and the domain must give identical ordering to both decimal refs
and nondecimal route labels, including Unicode digit-like characters.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from app.domain.tram_graph import route_sort_key, sorted_refs

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_tram_graph.py"

REFS = [
    "1", "2", "4", "10", "16", "50",
    "1а", "39а", "47а",
    "А", "т1", "т2", "²",
]


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_tram_graph", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"cannot import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_and_domain_agree_on_order() -> None:
    script_order = list(load_script().sorted_routes(set(REFS)))
    assert script_order == list(sorted_refs(REFS))


def test_numeric_refs_are_not_sorted_as_strings() -> None:
    assert list(sorted_refs(["10", "2", "1"])) == ["1", "2", "10"]
    assert list(load_script().sorted_routes({"10", "2", "1"})) == ["1", "2", "10"]


def test_non_numeric_refs_sort_after_numeric_ones() -> None:
    order = list(sorted_refs(["т1", "5", "А", "1а"]))
    assert order[0] == "5"
    assert set(order[1:]) == {"1а", "А", "т1"}
    assert order == list(load_script().sorted_routes({"т1", "5", "А", "1а"}))


def test_sort_key_never_calls_int_on_a_letter() -> None:
    for ref in REFS:
        route_sort_key(ref)

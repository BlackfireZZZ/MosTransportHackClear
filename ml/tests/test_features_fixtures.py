import json
import re
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import pytest

from tramflow_ml.features import (
    DAY_HOUR,
    MONTH_DAY,
    MOSCOW,
    YEAR_MONTH,
    CoverageCalendar,
    FeatureRequest,
    aggregate,
    build_features,
    load_entity_keys,
    load_observations,
)
from tramflow_ml.ingestion import IngestionConfig, ingest
from tramflow_ml.synthetic import SyntheticConfig, generate_dataset

TARGET = "synthetic_boardings"
UNIT = "event_count"
SOURCE_ROOT = Path(__file__).parents[1] / "src"
CONTRACTS_IMPORT = re.compile(r"^\s*(?:from|import)\s+contracts\b", re.MULTILINE)


@pytest.fixture
def fixture_run(tmp_path):
    """The real path: generate a fixture, ingest it, then read the ingestion output."""
    source = tmp_path / "in"
    output = tmp_path / "out"
    generate_dataset(SyntheticConfig(), source)
    manifest = ingest(IngestionConfig(input=source, output=output))
    return source, output, manifest


def generation(source):
    return json.loads((source / "generation.json").read_bytes())


def coverage_of(report):
    config = report["config"]
    return CoverageCalendar.from_range(
        date.fromisoformat(config["start"]),
        date.fromisoformat(config["end"]),
        gaps=[date.fromisoformat(day) for day in report["gap_dates"]],
    )


def oracle_cells(report):
    """Totals from the generator's own report, never from the feature code path."""
    return Counter(
        {
            (
                cell["route_id"],
                cell["direction_id"],
                cell["stop_id"],
                cell["date"],
                cell["hour"],
            ): cell["count"]
            for cell in report["cell_totals"]
        }
    )


def produced_cells(index, key):
    return Counter({key(cell): cell.value for cell in index.cells()})


def hourly_key(cell):
    return (
        cell.entity.route_id,
        cell.entity.direction_id,
        cell.entity.stop_id,
        cell.bucket.start.date().isoformat(),
        cell.bucket.start.hour,
    )


def monthly_key(cell):
    return (
        cell.entity.route_id,
        cell.entity.direction_id,
        cell.entity.stop_id,
        cell.bucket.start.isoformat()[:7],
    )


def index_for(fixture_run, granularity):
    source, output, _ = fixture_run
    report = generation(source)
    observations = load_observations(output / "validations.jsonl")
    return aggregate(observations, granularity, coverage_of(report).complete(), TARGET, UNIT)


def test_hourly_cells_reconcile_with_the_generator_report(fixture_run):
    source, _, manifest = fixture_run
    report = generation(source)

    index = index_for(fixture_run, "hourly")

    assert produced_cells(index, hourly_key) == oracle_cells(report)
    assert index.total == report["counts"]["unique_validations"]
    assert index.total == manifest["streams"]["validations"]["counts"]["valid"]


@pytest.mark.parametrize("granularity", ["daily", "monthly"])
def test_coarser_granularities_conserve_the_same_total(fixture_run, granularity):
    source, _, _ = fixture_run
    report = generation(source)

    index = index_for(fixture_run, granularity)

    values = [cell.value for cell in index.cells()]

    assert index.total == sum(oracle_cells(report).values())
    assert [value for value in values if value is None] == []
    assert sum(value for value in values if value is not None) == (
        report["counts"]["unique_validations"]
    )


def test_monthly_totals_group_the_generator_dates_by_calendar_month(fixture_run):
    source, _, _ = fixture_run
    report = generation(source)
    expected: Counter = Counter()
    for cell in report["cell_totals"]:
        key = (cell["route_id"], cell["direction_id"], cell["stop_id"], cell["date"][:7])
        expected[key] += cell["count"]

    index = index_for(fixture_run, "monthly")

    assert produced_cells(index, monthly_key) == expected


def build_from(source, output, policy, origin):
    report = generation(source)
    return build_features(
        FeatureRequest(
            policy=policy,
            origin=datetime.fromisoformat(origin).replace(tzinfo=MOSCOW),
            entities=load_entity_keys(source / "entities.json"),
            observations=load_observations(output / "validations.jsonl"),
            coverage=coverage_of(report),
            target=TARGET,
            unit=UNIT,
        )
    )


def test_two_builds_from_one_fixture_have_the_same_digest(fixture_run):
    source, output, _ = fixture_run

    first = build_from(source, output, YEAR_MONTH, "2025-01-01T00:00")
    second = build_from(source, output, YEAR_MONTH, "2025-01-01T00:00")

    assert first.digest == second.digest
    assert first.feature_digest == second.feature_digest


def test_a_second_ingestion_run_yields_the_same_features(tmp_path):
    source = tmp_path / "in"
    generate_dataset(SyntheticConfig(), source)

    results = []
    for name in ("out-a", "out-b"):
        manifest = ingest(IngestionConfig(input=source, output=tmp_path / name))
        built = build_from(source, tmp_path / name, MONTH_DAY, "2025-06-01T00:00")
        results.append((manifest["output_hash"], built.digest))

    assert results[0] == results[1]


def test_the_fixture_has_entities_and_uncovered_gap_dates(fixture_run):
    source, _, _ = fixture_run
    report = generation(source)

    assert len(load_entity_keys(source / "entities.json")) > 1
    assert report["gap_dates"]
    assert date.fromisoformat(report["gap_dates"][0]) not in coverage_of(report).dates


def test_production_code_never_imports_the_contracts_package():
    offenders = [
        path.relative_to(SOURCE_ROOT).as_posix()
        for path in sorted(SOURCE_ROOT.rglob("*.py"))
        if CONTRACTS_IMPORT.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


# The documented determinism evidence is only evidence while something checks it.
# These are the values recorded in ml/README.md and the completed execution plan, so a
# change that moves one is a decision to take deliberately, not a document to refresh
# afterwards. The year/month value moved once already, when the *_units columns were
# clipped at the cutoff, and nothing failed.
PINNED = {
    "day/hour": (
        DAY_HOUR,
        "2025-06-10T00:00",
        288,
        25,
        "7e412bb01ce0072aa68d77bfde177581c185abdeaed8e95dbaf926b82735b999",
        "2f7067776b125aad6bc8451b350e3957a362f8f9808e74ff39f8fd7dd046ef30",
    ),
    "month/day": (
        MONTH_DAY,
        "2025-06-01T00:00",
        360,
        25,
        "b3ad914c0ca53de67cb57fe785c296d042d5cd614de43566f636807489cb4170",
        "0e84576dcd94e5fe6ee005a1b9270a4ec3b713f03d00750e3c201437ecf8f03a",
    ),
    "year/month": (
        YEAR_MONTH,
        "2025-01-01T00:00",
        144,
        29,
        "58e57f490b3315b195dc3b480bc5b14dea44c390f796256b1388b9680db30156",
        "fb7bf53b3bdd509eef2f795dc64c498a1e09d1829ce21e4b0d20cef3b858ccce",
    ),
}


@pytest.mark.parametrize("policy_name", sorted(PINNED))
def test_the_published_digest_still_describes_the_committed_fixture(fixture_run, policy_name):
    source, output, _ = fixture_run
    policy, origin, rows, columns, digest, feature_digest = PINNED[policy_name]

    table = build_from(source, output, policy, origin)

    assert (len(table.rows), len(table.feature_names)) == (rows, columns)
    assert table.digest == digest
    assert table.feature_digest == feature_digest

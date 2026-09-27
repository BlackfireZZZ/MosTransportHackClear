"""Exercise active-run backfill across the scored forecast schema upgrade."""

import argparse
import asyncio
from pathlib import Path

from sqlalchemy import text

from app.infrastructure import forecast_publication
from app.infrastructure.db.session import session_factory
from app.infrastructure.scored_bundle import load_scored_bundle

ROOT = Path(__file__).resolve().parents[1]


async def prepare() -> None:
    bundle = load_scored_bundle(ROOT / "ml/competition_submissions/current.json")

    async def before_pointer_schema(*_args: object) -> None:
        return None

    forecast_publication._activate = before_pointer_schema
    async with session_factory() as session:
        await forecast_publication.publish_scored_bundle(session, bundle)


async def verify() -> None:
    bundle = load_scored_bundle(ROOT / "ml/competition_submissions/current.json")
    async with session_factory() as session:
        pointers = await session.scalar(text("SELECT count(*) FROM forecast_active_runs"))
        assert pointers == 20, f"expected 20 backfilled route/horizon pointers, found {pointers}"
        for horizon, expected in (("day", 14640), ("month", 610)):
            run_id = f"scored-2025-{horizon}-{bundle.manifest_sha256}"
            point_count = await session.scalar(
                text("SELECT count(*) FROM forecast_points WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            active_count = await session.scalar(
                text("SELECT count(*) FROM forecast_active_runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            assert point_count == expected, f"{horizon} point count: {point_count}"
            assert active_count == 10, f"{horizon} active route count: {active_count}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify"))
    phase = parser.parse_args().phase
    asyncio.run(prepare() if phase == "prepare" else verify())


if __name__ == "__main__":
    main()

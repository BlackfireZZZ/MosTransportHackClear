from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from app.domain.forecast import ForecastHorizon, ForecastQueryError, ForecastSelection


def test_window_limit_uses_elapsed_time_for_zoneinfo_calls() -> None:
    zone = ZoneInfo("Europe/Berlin")
    selection = ForecastSelection(
        start=datetime(2026, 10, 25, tzinfo=zone), end=datetime(2026, 10, 26, tzinfo=zone)
    )
    with pytest.raises(ForecastQueryError, match="horizon limit"):
        selection.validate(ForecastHorizon.DAY)


@pytest.mark.parametrize(
    "target,unit",
    [
        ("synthetic_boardings", "event_count"),
        ("validation_count", "event_count"),
        ("boarding_count", "passengers"),
    ],
)
def test_disjoint_count_targets_are_additive(target: str, unit: str) -> None:
    from app.domain.forecast import require_additive_target

    require_additive_target(target, unit, 2)


def test_occupancy_is_only_readable_without_summing() -> None:
    from app.domain.forecast import ForecastDataConflict, require_additive_target

    require_additive_target("onboard_load", "passengers", 1)
    with pytest.raises(ForecastDataConflict, match="cannot be summed"):
        require_additive_target("onboard_load", "passengers", 2)

import html
import json

import pytest

from tramflow_ml.boarding.timetables import parse_timetable_html

URL = (
    "https://web.archive.org/web/20250729161007id_/"
    "https://transport.mos.ru/transport/schedule/route/141501932"
)


def source() -> bytes:
    geography = {
        "features": [
            {
                "id": 12,
                "geometry": {"type": "Point", "coordinates": [37.7, 55.8]},
                "properties": {"hintContent": "A & B"},
            },
            {"id": 99, "geometry": {"type": "LineString", "coordinates": [[37.7, 55.8]]}},
        ]
    }
    return (
        '<html><h1><i class="icon-tramway"></i>7</h1>'
        f'<div data-coords="{html.escape(json.dumps(geography))}"></div>'
        '<ul><li data-stop="2" data-date="2025-07-29" data-direction="0">'
        '<div class="a_dotted d-inline">A &amp; B</div>'
        '<div class="dt1"><strong>00:</strong></div><div class="div10">06</div>'
        '<div class="dt1"><strong>23:</strong></div><div class="div10">59</div>'
        "</li></ul></html>"
    ).encode()


def parse(raw: bytes) -> dict:
    return parse_timetable_html(
        raw,
        route_ref="7",
        direction=0,
        source_url=URL,
        capture_date="2025-07-29",
        status="historical_snapshot",
    )


def test_preserves_midnight_clocks_geography_and_source_provenance() -> None:
    result = parse(source())
    assert result["stops"][0]["departures_seconds"] == [360, 86340]
    assert result["stops"][0]["stop_id"] == "12"
    assert result["stops"][0]["stop_name"] == "A & B"
    assert result["trip_identity_available"] is False
    assert result["status"] == "historical_snapshot"
    assert len(result["source_sha256"]) == 64


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (b"</html>", b"", "incomplete HTML"),
        (b">7</h1>", b">12</h1>", "route heading"),
        (b"icon-tramway", b"ic-bus", "not a tram"),
        (b'data-direction="0"', b'data-direction="1"', "direction mismatch"),
        (b">59</div>", b">60</div>", "invalid departure"),
        (b"A &amp; B</div>", b"C</div>", "stop name/order"),
        (b"2025-07-29", b"2025-07-30", "historical snapshot"),
    ],
)
def test_rejects_wrong_or_incomplete_source(before: bytes, after: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse(source().replace(before, after))


def test_live_page_cannot_claim_historical_capture() -> None:
    with pytest.raises(ValueError, match="historical snapshot"):
        parse_timetable_html(
            source(),
            route_ref="7",
            direction=0,
            source_url="https://transport.mos.ru/transport/schedule/route/11",
            capture_date="2025-07-29",
            status="historical_snapshot",
        )


def test_current_proxy_retains_actual_schedule_date() -> None:
    result = parse_timetable_html(
        source(),
        route_ref="7",
        direction=0,
        source_url="https://transport.mos.ru/transport/schedule/route/11",
        capture_date="2026-09-26",
        status="current_proxy",
    )
    assert result["schedule_date"] == "2025-07-29"
    assert result["capture_date"] == "2026-09-26"
    assert result["stops"][0]["status"] == "current_proxy"

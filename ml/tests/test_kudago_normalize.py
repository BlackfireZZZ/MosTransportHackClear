"""Offline normalisation: no test here performs network I/O.

The fixtures mirror the three shapes the live API returns for one entity -- an
inventory projection, an object inlined by `expand`, and a detail response -- because
a suite that only ever fed detail responses passed while the collector mangled its
first real page.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from tramflow_ml.external.kudago import normalize as normalize_module
from tramflow_ml.external.kudago.normalize import (
    FLAG_COORDINATES_MISSING,
    FLAG_COORDINATES_OUT_OF_RANGE,
    FLAG_COORDINATES_OUTSIDE_MOSCOW,
    FLAG_COORDINATES_POSSIBLY_TRANSPOSED,
    FLAG_HISTORICAL_SNAPSHOT_UNAVAILABLE,
    FLAG_PLACE_DETAIL_MISSING,
    FLAG_PLACE_MISSING,
    REASON_BODY_HASH_MISMATCH,
    REASON_MISSING_SOURCE_URL,
    REASON_MISSING_TITLE,
    REASON_RAW_BODY_MISSING,
    NormalizeConfig,
    normalize,
)
from tramflow_ml.external.kudago.records import (
    EVENTS_NAME,
    MOSCOW,
    OBSERVATIONS_NAME,
    OCCURRENCES_NAME,
    PLACES_NAME,
    QUARANTINE_NAME,
    REQUESTS_NAME,
)
from tramflow_ml.external.kudago.timerules import (
    FLAG_OUTSIDE_WINDOW,
    REASON_NEGATIVE_INTERVAL,
)

ROOT = "https://kudago.com/public-api/v1.4"
INVENTORY_URL = f"{ROOT}/events/?fields=id%2Cdates%2Cplace%2Csite_url&location=msk&page_size=5"
EVENT_URL = ROOT + "/events/{}/?expand=dates%2Cplace&lang=ru"
PLACE_URL = ROOT + "/places/{}/?lang=ru"
FETCHED_AT = "2026-09-27T10:00:00+03:00"
CITY_CENTRE = {"lat": 55.7522, "lon": 37.6156}


def unix(text):
    return int(datetime.fromisoformat(text).replace(tzinfo=MOSCOW).timestamp())


def date_record(day, start_time, end_time):
    return {
        "start_date": day,
        "start_time": start_time,
        "start": unix(f"{day}T{start_time}"),
        "end_date": day,
        "end_time": end_time,
        "end": unix(f"{day}T{end_time}"),
        "is_continuous": False,
        "is_startless": False,
        "is_endless": False,
        "use_place_schedule": False,
        "schedules": [],
    }


def inlined_place(place_id=42, coords=None, **overrides):
    """The venue as `expand=place` inlines it: eleven keys, no catalogue fields."""
    card = {
        "id": place_id,
        "title": "Театр",
        "slug": "teatr",
        "address": "Тверская, 1",
        "location": "msk",
        "phone": "+7 495 000-00-00",
        "site_url": "https://kudago.com/msk/place/teatr/",
        "subway": "Пушкинская",
        "coords": {"lat": 55.7601, "lon": 37.6089} if coords is None else coords,
        "is_closed": False,
        "is_stub": False,
    }
    card.update(overrides)
    return card


def place_card(place_id=42, coords=None, **overrides):
    """The venue as `places/{id}/` returns it: the inlined fields plus the rest."""
    card = inlined_place(place_id, coords)
    card.update(
        {
            "timetable": "ежедневно с 10:00 до 22:00",
            "description": "<p>Описание площадки</p>",
            "body_text": "<p>Текст площадки</p>",
            "categories": ["theatre"],
            "favorites_count": 12,
            "comments_count": 3,
            "has_parking_lot": False,
            "age_restriction": 0,
            "foreign_url": "",
        }
    )
    card.update(overrides)
    return card


def event_card(event_id=3606, dates=None, place=None, **overrides):
    card = {
        "id": event_id,
        "title": "Концерт",
        "slug": "kontsert",
        "site_url": "https://kudago.com/msk/event/kontsert/",
        "publication_date": unix("2026-04-16T12:00:00"),
        "description": "<p>Описание</p>",
        "body_text": "<p>Текст</p>",
        "price": "от 500 рублей",
        "age_restriction": 6,
        "is_free": False,
        "location": {"slug": "msk", "coords": CITY_CENTRE},
        "categories": ["concert"],
        "tags": ["музыка"],
        "dates": [date_record("2025-01-15", "19:00:00", "21:00:00")] if dates is None else dates,
        "place": inlined_place() if place is None else place,
    }
    card.update(overrides)
    return card


def inventory_row(card):
    """One row of `fields=id,dates,place,site_url`: no title exists in this shape."""
    place = card.get("place")
    return {
        "id": card["id"],
        "dates": card["dates"],
        "place": {"id": place["id"]} if isinstance(place, dict) else None,
        "site_url": card["site_url"],
    }


def page(*objects):
    return {"count": len(objects), "next": None, "previous": None, "results": list(objects)}


def card_request(card, request_id="req-1", fetched_at=FETCHED_AT):
    return (request_id, EVENT_URL.format(card["id"]), fetched_at, card)


def venue_request(place, request_id=None, fetched_at=FETCHED_AT):
    return (
        request_id or f"req-place-{place['id']}",
        PLACE_URL.format(place["id"]),
        fetched_at,
        place,
    )


def inventory_request(*cards, request_id="req-list", fetched_at=FETCHED_AT):
    return (request_id, INVENTORY_URL, fetched_at, page(*[inventory_row(card) for card in cards]))


def build_run(tmp_path, requests, dictionaries=None, name="in"):
    """Write the layout the fetcher leaves behind: raw bodies plus a request journal."""
    root = tmp_path / name
    (root / "raw").mkdir(parents=True, exist_ok=True)
    lines = []
    for request_id, url, fetched_at, payload in requests:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        (root / "raw" / f"{request_id}.json").write_bytes(body)
        lines.append(
            {
                "request_id": request_id,
                "run_id": "run-1",
                "url": url,
                "requested_at": fetched_at,
                "fetched_at": fetched_at,
                "http_status": 200,
                "attempt": 1,
                "raw_path": f"raw/{request_id}.json",
                "body_sha256": hashlib.sha256(body).hexdigest(),
                "error": None,
                "count": payload.get("count") if isinstance(payload, dict) else None,
                "next_url": None,
            }
        )
    journal = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    (root / REQUESTS_NAME).write_text(journal, encoding="utf-8")
    if dictionaries:
        (root / "dictionaries").mkdir(exist_ok=True)
        for stem, entries in dictionaries.items():
            (root / "dictionaries" / f"{stem}.json").write_text(
                json.dumps(entries, ensure_ascii=False), encoding="utf-8"
            )
    return root


def configure(source, output):
    return NormalizeConfig(
        input_dir=source, output_dir=output, windows=(("2025-01-01", "2025-02-01"),)
    )


def run_normalize(tmp_path, requests, out="out", dictionaries=None):
    source = build_run(tmp_path, requests, dictionaries)
    return normalize(configure(source, tmp_path / out)), tmp_path / out


def venues_of(cards):
    """One venue detail per distinct inlined place, in first-seen order."""
    venues = {}
    for card in cards:
        place = card.get("place")
        if isinstance(place, dict) and "id" in place:
            venues.setdefault(place["id"], place_card(place["id"], place.get("coords")))
    return list(venues.values())


def collect(tmp_path, *cards, venues=True, inventory=False, out="out"):
    requests = [inventory_request(*cards)] if inventory else []
    requests += [
        card_request(card, request_id=f"req-{index}") for index, card in enumerate(cards, start=1)
    ]
    if venues:
        requests += [venue_request(place) for place in venues_of(cards)]
    return run_normalize(tmp_path, requests, out=out)


def rows(directory, name):
    text = (directory / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


# The three shapes of one entity.


def test_all_three_shapes_of_one_entity_collapse_into_one_row_each(tmp_path):
    card = event_card()
    inlined = card["place"]
    detail = place_card()
    assert len(inventory_row(card)) < len(inlined) < len(detail)

    report, out = run_normalize(
        tmp_path,
        [
            inventory_request(card),
            card_request(card, request_id="req-1"),
            venue_request(detail),
        ],
    )

    events = rows(out, EVENTS_NAME)
    places = rows(out, PLACES_NAME)
    assert [event["event_id"] for event in events] == ["3606"]
    assert [place["place_id"] for place in places] == ["42"]
    assert (places[0]["lat"], places[0]["lon"]) == (55.7601, 37.6089)
    assert places[0]["payload"] == detail
    assert rows(out, QUARANTINE_NAME) == []
    assert report["event_projections"] == 1
    assert report["place_projections"] == 2
    assert report["events_without_detail"] == 0
    assert report["places_without_detail"] == 0


def test_a_card_seen_only_as_an_inventory_projection_is_not_quarantined(tmp_path):
    card = event_card()

    report, out = run_normalize(tmp_path, [inventory_request(card)])

    assert rows(out, EVENTS_NAME) == []
    assert rows(out, PLACES_NAME) == []
    assert rows(out, OCCURRENCES_NAME) == []
    assert rows(out, OBSERVATIONS_NAME) == []
    assert rows(out, QUARANTINE_NAME) == []
    assert report["events_without_detail"] == 1
    assert report["places_without_detail"] == 1
    assert report["date_records"] == 0


def test_an_inlined_venue_never_becomes_a_venue_row(tmp_path):
    report, out = collect(tmp_path, event_card(), venues=False)

    assert rows(out, PLACES_NAME) == []
    assert [row["object_type"] for row in rows(out, OBSERVATIONS_NAME)] == ["event"]
    assert report["places_without_detail"] == 1


def test_a_card_whose_venue_was_never_detailed_is_flagged_not_dangling(tmp_path):
    _, out = collect(tmp_path, event_card(), venues=False)

    event = rows(out, EVENTS_NAME)[0]
    occurrence = rows(out, OCCURRENCES_NAME)[0]
    assert FLAG_PLACE_DETAIL_MISSING in event["quality_flags"]
    assert event["place_id"] == "42"
    assert occurrence["place_id"] == "42"
    assert occurrence["place_snapshot_id"] is None


def test_the_occurrence_points_at_the_delivered_venue_row(tmp_path):
    _, out = collect(tmp_path, event_card())

    place = rows(out, PLACES_NAME)[0]
    event = rows(out, EVENTS_NAME)[0]
    occurrence = rows(out, OCCURRENCES_NAME)[0]
    assert occurrence["place_snapshot_id"] == place["snapshot_id"]
    assert FLAG_PLACE_DETAIL_MISSING not in event["quality_flags"]


def test_a_reference_shaped_venue_is_counted_and_never_published(tmp_path):
    bare = {"id": 2033}

    report, out = run_normalize(
        tmp_path, [inventory_request(event_card(place=bare, event_id=2033))]
    )

    assert rows(out, PLACES_NAME) == []
    assert report["place_projections"] == 1
    assert report["places_without_detail"] == 1


# Delivery records.


def test_a_detail_response_becomes_an_event_a_venue_and_an_occurrence(tmp_path):
    report, out = collect(tmp_path, event_card())

    occurrences = rows(out, OCCURRENCES_NAME)
    observations = rows(out, OBSERVATIONS_NAME)
    assert len(occurrences) == 1
    assert occurrences[0]["time_precision"] == "exact_interval"
    assert occurrences[0]["start_at"] == "2025-01-15T19:00:00+03:00"
    assert {row["object_type"] for row in observations} == {"event", "place"}
    assert report["reconciled"] is True
    assert rows(out, QUARANTINE_NAME) == []


def test_a_card_going_a_b_a_leaves_three_observations_over_two_snapshots(tmp_path):
    first = event_card()
    postponed = event_card(dates=[date_record("2025-01-22", "19:00:00", "21:00:00")])
    reverted = event_card()

    report, out = run_normalize(
        tmp_path,
        [
            card_request(first, "req-1", "2026-09-25T10:00:00+03:00"),
            card_request(postponed, "req-2", "2026-09-26T10:00:00+03:00"),
            card_request(reverted, "req-3", "2026-09-27T10:00:00+03:00"),
        ],
    )

    observations = [row for row in rows(out, OBSERVATIONS_NAME) if row["object_type"] == "event"]
    assert [row["request_id"] for row in observations] == ["req-1", "req-2", "req-3"]
    assert len({row["snapshot_id"] for row in observations}) == 2
    assert observations[0]["snapshot_id"] == observations[2]["snapshot_id"]
    assert len(rows(out, EVENTS_NAME)) == 2
    assert report["observations"] == 3


def test_three_shapes_of_one_card_are_not_three_states_of_it(tmp_path):
    card = event_card()

    report, out = run_normalize(
        tmp_path,
        [
            inventory_request(card),
            card_request(card, request_id="req-1"),
            venue_request(place_card()),
        ],
    )

    observations = rows(out, OBSERVATIONS_NAME)
    assert [row["object_type"] for row in observations] == ["event", "place"]
    assert report["observations"] == 2
    assert len(rows(out, EVENTS_NAME)) == 1


def test_availability_is_the_fetch_and_publication_date_never_moves_it_earlier(tmp_path):
    report, out = collect(tmp_path, event_card())

    event = rows(out, EVENTS_NAME)[0]
    assert event["available_at"] == FETCHED_AT
    assert event["available_at"] == event["fetched_at"]
    assert event["availability_basis"] == "observed_snapshot"
    assert event["published_at"] == "2026-04-16T12:00:00+03:00"
    assert event["published_at"] < event["available_at"]
    assert FLAG_HISTORICAL_SNAPSHOT_UNAVAILABLE in event["quality_flags"]
    assert rows(out, OCCURRENCES_NAME)[0]["available_at"] == event["available_at"]
    assert report["first_fetched_at"] == FETCHED_AT


def test_card_presence_is_not_evidence_that_the_event_happened(tmp_path):
    _, out = collect(tmp_path, event_card())

    event = rows(out, EVENTS_NAME)[0]
    assert event["status"] == "unknown"
    assert event["status_evidence"] is None


def test_identifiers_keep_the_form_the_source_wrote(tmp_path):
    _, out = collect(tmp_path, event_card(event_id="00123", place=inlined_place(place_id=7)))

    event = rows(out, EVENTS_NAME)[0]
    assert event["event_id"] == "00123"
    assert event["place_id"] == "7"


def test_a_card_without_a_place_is_flagged_not_quarantined(tmp_path):
    card = event_card()
    card["place"] = None

    report, out = collect(tmp_path, card)

    event = rows(out, EVENTS_NAME)[0]
    assert FLAG_PLACE_MISSING in event["quality_flags"]
    assert event["place_id"] is None
    assert rows(out, PLACES_NAME) == []
    assert report["quarantined"] == 0
    assert rows(out, OCCURRENCES_NAME)[0]["place_snapshot_id"] is None


# Coordinates.


def test_valid_venue_coordinates_are_kept_with_their_evidence(tmp_path):
    _, out = collect(tmp_path, event_card())

    place = rows(out, PLACES_NAME)[0]
    assert (place["lat"], place["lon"]) == (55.7601, 37.6089)
    assert place["coordinate_source"] == "place_api"
    assert place["coordinate_precision"] == "venue_area"
    assert place["coordinate_evidence"] == "raw/req-place-42.json#/coords"
    assert place["timetable_text"] == "ежедневно с 10:00 до 22:00"


def test_the_city_centre_in_location_coords_is_never_used_as_a_venue(tmp_path):
    venue = place_card()
    del venue["coords"]

    _, out = run_normalize(
        tmp_path, [card_request(event_card()), venue_request(venue)]
    )

    place = rows(out, PLACES_NAME)[0]
    assert place["lat"] is None and place["lon"] is None
    assert place["coordinate_source"] == "missing"
    assert FLAG_COORDINATES_MISSING in place["quality_flags"]
    assert place["payload"].get("coords") is None
    assert CITY_CENTRE["lat"] not in (place["lat"], place["lon"])


def test_coordinates_outside_the_valid_range_are_rejected(tmp_path):
    venue = place_card(coords={"lat": 91.0, "lon": 37.6})

    _, out = run_normalize(tmp_path, [venue_request(venue)])

    place = rows(out, PLACES_NAME)[0]
    assert (place["lat"], place["lon"]) == (None, None)
    assert FLAG_COORDINATES_OUT_OF_RANGE in place["quality_flags"]
    assert place["payload"]["coords"] == {"lat": 91.0, "lon": 37.6}


def test_transposed_coordinates_are_flagged_and_never_swapped(tmp_path):
    venue = place_card(coords={"lat": 37.6089, "lon": 55.7601})

    _, out = run_normalize(tmp_path, [venue_request(venue)])

    place = rows(out, PLACES_NAME)[0]
    assert (place["lat"], place["lon"]) == (None, None)
    assert FLAG_COORDINATES_OUTSIDE_MOSCOW in place["quality_flags"]
    assert FLAG_COORDINATES_POSSIBLY_TRANSPOSED in place["quality_flags"]
    assert place["payload"]["coords"] == {"lat": 37.6089, "lon": 55.7601}


def test_a_venue_detail_stands_on_its_own(tmp_path):
    report, out = run_normalize(tmp_path, [venue_request(place_card())])

    place = rows(out, PLACES_NAME)[0]
    assert report["events"] == 0 and report["places"] == 1
    assert place["place_id"] == "42"
    assert place["coordinate_evidence"] == "raw/req-place-42.json#/coords"
    assert rows(out, OBSERVATIONS_NAME)[0]["object_type"] == "place"


# Required fields and quarantine.


def test_a_card_with_an_empty_title_is_quarantined_with_its_date_records(tmp_path):
    dates = [
        date_record("2025-01-15", "19:00:00", "21:00:00"),
        date_record("2025-01-16", "19:00:00", "21:00:00"),
    ]

    report, out = collect(tmp_path, event_card(title="", dates=dates))

    quarantine = rows(out, QUARANTINE_NAME)
    assert rows(out, EVENTS_NAME) == []
    assert rows(out, OCCURRENCES_NAME) == []
    assert len(quarantine) == 1
    assert quarantine[0]["reason"] == REASON_MISSING_TITLE
    assert quarantine[0]["event_id"] == "3606"
    assert quarantine[0]["raw_ref"] == "raw/req-1.json#/title"
    assert "2 date records" in quarantine[0]["detail"]
    assert report["date_records"] == 2
    assert report["date_records_quarantined"] == 2
    assert report["date_records_resolved"] == 0
    assert report["reconciled"] is True


def test_a_card_without_a_site_url_is_quarantined_and_leaves_no_observation(tmp_path):
    card = event_card()
    del card["site_url"]

    report, out = collect(tmp_path, card, venues=False)

    assert rows(out, EVENTS_NAME) == []
    assert [row["reason"] for row in rows(out, QUARANTINE_NAME)] == [REASON_MISSING_SOURCE_URL]
    assert rows(out, OBSERVATIONS_NAME) == []
    assert report["events"] == 0


def test_a_repeated_broken_card_is_quarantined_once_per_snapshot(tmp_path):
    broken = event_card(title="")

    report, _ = run_normalize(
        tmp_path,
        [
            card_request(broken, "req-1", "2026-09-25T10:00:00+03:00"),
            card_request(broken, "req-2", "2026-09-26T10:00:00+03:00"),
        ],
    )

    assert report["quarantined"] == 1
    assert report["date_records"] == 1


def test_a_venue_without_a_title_keeps_a_flag_rather_than_being_dropped(tmp_path):
    nameless = place_card()
    nameless["title"] = ""

    report, out = run_normalize(tmp_path, [card_request(event_card()), venue_request(nameless)])

    place = rows(out, PLACES_NAME)[0]
    assert "title_missing" in place["quality_flags"]
    assert (place["lat"], place["lon"]) == (55.7601, 37.6089)
    assert report["events"] == 1
    assert rows(out, OCCURRENCES_NAME)[0]["place_snapshot_id"] == place["snapshot_id"]
    assert report["quarantined"] == 0


def test_every_date_record_yields_an_occurrence_or_a_quarantine_row(tmp_path):
    negative = date_record("2025-01-15", "20:00:00", "18:00:00")
    negative["end"] = None
    negative["start"] = None
    dates = [date_record("2025-01-15", "19:00:00", "21:00:00"), negative, "не дата"]

    report, out = collect(tmp_path, event_card(dates=dates))

    quarantine = rows(out, QUARANTINE_NAME)
    assert report["date_records"] == 3
    assert report["date_records_resolved"] == 1
    assert report["date_records_quarantined"] == 2
    assert report["reconciled"] is True
    assert {row["reason"] for row in quarantine} == {
        REASON_NEGATIVE_INTERVAL,
        "invalid_dates_field",
    }
    assert all(row["raw_ref"].startswith("raw/req-1.json#/dates/") for row in quarantine)
    occurrences = rows(out, OCCURRENCES_NAME)
    assert len(occurrences) == 1
    covered = {row["date_index"] for row in quarantine} | {row["date_index"] for row in occurrences}
    assert covered == set(range(len(dates)))


def test_a_date_outside_the_window_is_kept_as_one_unresolved_row(tmp_path):
    _, out = collect(
        tmp_path, event_card(dates=[date_record("2024-06-01", "19:00:00", "21:00:00")])
    )

    occurrences = rows(out, OCCURRENCES_NAME)
    assert len(occurrences) == 1
    assert occurrences[0]["time_precision"] == "unresolved"
    assert FLAG_OUTSIDE_WINDOW in occurrences[0]["quality_flags"]
    assert occurrences[0]["start_at"] is None
    assert occurrences[0]["start_date"] == "2024-06-01"
    assert rows(out, QUARANTINE_NAME) == []


def test_the_occurrence_carries_the_raw_date_record_it_came_from(tmp_path):
    raw = date_record("2025-01-15", "19:00:00", "21:00:00")

    _, out = collect(tmp_path, event_card(dates=[raw]))

    occurrence = rows(out, OCCURRENCES_NAME)[0]
    assert occurrence["raw_date"] == raw
    assert occurrence["date_index"] == 0
    assert occurrence["expansion_index"] == 0
    assert occurrence["schedule_basis"] == "explicit"


def test_a_missing_raw_body_is_quarantined_with_a_reference(tmp_path):
    source = build_run(tmp_path, [card_request(event_card())])
    (source / "raw" / "req-1.json").unlink()

    report = normalize(configure(source, tmp_path / "out"))

    quarantine = rows(tmp_path / "out", QUARANTINE_NAME)
    assert report["events"] == 0
    assert [row["reason"] for row in quarantine] == [REASON_RAW_BODY_MISSING]
    assert quarantine[0]["raw_ref"] == "raw/req-1.json#"


def test_a_body_that_does_not_match_the_journal_hash_is_quarantined(tmp_path):
    source = build_run(tmp_path, [card_request(event_card())])
    (source / "raw" / "req-1.json").write_text("{}", encoding="utf-8")

    report = normalize(configure(source, tmp_path / "out"))

    assert report["events"] == 0
    assert [row["reason"] for row in rows(tmp_path / "out", QUARANTINE_NAME)] == [
        REASON_BODY_HASH_MISMATCH
    ]


def test_unknown_dictionary_slugs_are_flagged_when_the_dictionary_is_present(tmp_path):
    source = build_run(
        tmp_path,
        [card_request(event_card())],
        dictionaries={
            "event-categories": [{"slug": "theater"}],
            "locations": [{"slug": "msk"}],
        },
    )

    normalize(configure(source, tmp_path / "out"))

    event = rows(tmp_path / "out", EVENTS_NAME)[0]
    assert "unknown_category" in event["quality_flags"]
    assert "unknown_location" not in event["quality_flags"]


# Determinism and encoding.


def test_normalising_the_same_raw_twice_is_byte_identical(tmp_path):
    second = event_card(
        event_id=7,
        dates=[
            date_record("2025-01-20", "10:00:00", "12:00:00"),
            {"start_date": "2025-01-21", "schedules": [], "is_endless": True},
        ],
        place=inlined_place(place_id=9, coords={"lat": 0.0, "lon": 0.0}),
    )
    requests = [
        inventory_request(event_card(), second),
        card_request(event_card(), "req-1", "2026-09-25T10:00:00+03:00"),
        card_request(second, "req-2", "2026-09-26T10:00:00+03:00"),
        venue_request(place_card()),
        venue_request(place_card(9, {"lat": 0.0, "lon": 0.0})),
    ]
    source = build_run(tmp_path, requests)
    first_report = normalize(configure(source, tmp_path / "a"))
    second_report = normalize(configure(source, tmp_path / "b"))
    third_report = normalize(configure(source, tmp_path / "a"))

    names = (EVENTS_NAME, PLACES_NAME, OCCURRENCES_NAME, OBSERVATIONS_NAME, QUARANTINE_NAME)
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert first_report == second_report == third_report
    assert first_report["occurrences"] > 0


def test_run_metadata_comes_from_the_journal_not_from_the_clock(tmp_path):
    report, _ = run_normalize(
        tmp_path,
        [
            card_request(event_card(event_id=2), "req-2", "2026-09-26T10:00:00+03:00"),
            card_request(event_card(event_id=1), "req-1", "2026-09-25T10:00:00+03:00"),
        ],
    )

    assert report["first_fetched_at"] == "2026-09-25T10:00:00+03:00"
    assert report["last_fetched_at"] == "2026-09-26T10:00:00+03:00"
    assert report["run_ids"] == ["run-1"]
    assert report["normalizer_config_sha256"] == NormalizeConfig(
        tmp_path / "elsewhere", tmp_path / "other", (("2025-01-01", "2025-02-01"),)
    ).fingerprint()


def test_output_lines_are_one_utf8_json_object_each_without_a_bom(tmp_path):
    _, out = collect(tmp_path, event_card())

    for name in (EVENTS_NAME, PLACES_NAME, OCCURRENCES_NAME, OBSERVATIONS_NAME):
        payload = (out / name).read_bytes()
        assert not payload.startswith(b"\xef\xbb\xbf")
        assert payload.endswith(b"\n")
        for line in payload.decode("utf-8").splitlines():
            assert isinstance(json.loads(line), dict)
            assert "NaN" not in line and "Infinity" not in line
    assert "Концерт".encode() in (out / EVENTS_NAME).read_bytes()


def test_the_normaliser_reads_files_and_never_the_network(tmp_path):
    source = Path(normalize_module.__file__).read_text(encoding="utf-8")

    assert "urllib" not in source
    assert "socket" not in source
    assert "kudago.http" not in source
    assert "kudago.fetch" not in source

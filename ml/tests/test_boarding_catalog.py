import io
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

from tramflow_ml.boarding.catalog import MEMBER, catalog_inventory


def reference_archive(path: Path, *, duplicate_sequence: bool = False,
                      missing_stop: bool = False, bad_header: bool = False) -> Path:
    pattern_headers = ['route_id', 'route_short_name', 'trip_id', 'direction_id',
                       'start_date', 'end_date', 'stop_sequence', 'stop_id', 'actual_date']
    patterns = [
        ['r', '1', 'pattern', '0', '2025-12-20', '', '3', 'a', '2026-01-01'],
        ['r', '1', 'pattern', '0', '2025-12-20', '', '1', 'a', '2026-01-01'],
        ['r', '1', 'pattern', '0', '2025-12-20', '',
         '1' if duplicate_sequence else '2', 'absent' if missing_stop else 'b', '2026-01-01'],
    ]
    sheets = [
        ('Маршруты GTFS_ROUTES', ['route_id', 'route_short_name'], [['r', '1']], True),
        ('Остановки GTFS_STOPS', ['stop_id', 'stop_name', 'stop_lat', 'stop_lon'],
         [['a', 'Stop A', '55.1', '37.1'], ['b', 'Stop B', '55.2', '37.2']], True),
        ('Порядок_остановок GTFS_TRIPS_ST', pattern_headers, patterns, True),
        ('Порядок_с_координатами', pattern_headers + ['stop_lat', 'stop_lon'],
         [row + ['55.1', '37.1'] for row in patterns], False),
        ('Наряд', ['date', 'vehicle_id'], [['20260208', 'PRIVATE_VEHICLE']], True),
        ('Расписание', ['trip_id', 'stop_sequence', 'arrival_time', 'departure_time'],
         [['pattern', '1', '12:00', '12:01']], True),
    ]
    if bad_header:
        sheets[2][1][2] = 'unrecognized'
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as workbook:
        workbook.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets>' +
                          ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="r{i}"/>'
                                  for i, (name, _, _, _) in enumerate(sheets, 1)) +
                          '</sheets></workbook>')
        workbook.writestr('xl/_rels/workbook.xml.rels', '<Relationships>' +
                          ''.join(f'<Relationship Id="r{i}" Target="worksheets/sheet{i}.xml"/>'
                                  for i in range(1, len(sheets) + 1)) + '</Relationships>')
        for i, (_, headers, data, note) in enumerate(sheets, 1):
            rows = ([['Description']] if note else []) + [headers] + data
            body = []
            for n, row in enumerate(rows, 1):
                cells = ''.join(f'<c r="{chr(65 + j)}{n}" t="inlineStr"><is><t>'
                                f'{escape(value)}</t></is></c>' for j, value in enumerate(row))
                body.append(f'<row r="{n}">{cells}</row>')
            workbook.writestr(f'xl/worksheets/sheet{i}.xml',
                              f'<worksheet xmlns="{ns}"><sheetData>{"".join(body)}'
                              '</sheetData></worksheet>')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(MEMBER, stream.getvalue())
    return path


def test_catalog_preserves_ordered_repeated_visits_and_unknown_history(tmp_path: Path) -> None:
    source = reference_archive(tmp_path / 'source.zip')
    before = source.read_bytes()
    result = catalog_inventory(source)
    assert source.read_bytes() == before
    assert result['historical_verified'] is False
    assert result['known_at'] is None
    assert result['schedule_observed'] is False
    assert result['independent_anchors'] == 0
    assert len(result['source_member_sha256']) == 64
    assert result['pattern_count'] == 1
    pattern = result['patterns'][0]
    assert pattern['historical_verified'] is False
    assert pattern['known_at'] is None
    assert pattern['declared_valid_from'] == '2025-12-20'
    assert pattern['declared_valid_to'] is None
    assert [visit['sequence'] for visit in pattern['visits']] == [1, 2, 3]
    assert [visit['stop_id'] for visit in pattern['visits']] == ['a', 'b', 'a']
    assert result['sheets']['Порядок_с_координатами']['header_row'] == 1
    assert result['sheets']['Порядок_остановок GTFS_TRIPS_ST']['header_row'] == 2
    assert result['integrity'] == {'missing_stop_references': 0,
                                   'missing_route_references': 0,
                                   'coordinate_order_key_sets_equal': True}
    encoded = json.dumps(result, allow_nan=False)
    assert 'PRIVATE_VEHICLE' not in encoded
    assert result == catalog_inventory(source)


def test_duplicate_pattern_sequences_are_not_silently_merged(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='duplicate or negative'):
        catalog_inventory(reference_archive(tmp_path / 'source.zip', duplicate_sequence=True))


def test_missing_stop_is_explicit_and_coordinates_stay_unknown(tmp_path: Path) -> None:
    result = catalog_inventory(reference_archive(tmp_path / 'source.zip', missing_stop=True))
    assert result['integrity']['missing_stop_references'] == 1
    visit = result['patterns'][0]['visits'][1]
    assert visit['latitude'] is None
    assert visit['longitude'] is None
    assert result['historical_verified'] is False


def test_missing_header_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='missing reference headers'):
        catalog_inventory(reference_archive(tmp_path / 'source.zip', bad_header=True))

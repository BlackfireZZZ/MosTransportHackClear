"""Read reference patterns as unverified inventory, never historical boarding labels."""

import hashlib
import io
import math
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

MEMBER = 'spravochniki/Хакатон_справочники_трамвай_10_маршрутов.xlsx'
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RELATIONSHIP = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
SHEETS = {
    'Маршруты GTFS_ROUTES': {'route_id', 'route_short_name'},
    'Остановки GTFS_STOPS': {'stop_id', 'stop_lat', 'stop_lon'},
    'Порядок_остановок GTFS_TRIPS_ST': {
        'route_id', 'route_short_name', 'trip_id', 'direction_id', 'stop_sequence', 'stop_id',
        'start_date', 'end_date', 'actual_date',
    },
    'Порядок_с_координатами': {'trip_id', 'stop_sequence', 'stop_id', 'stop_lat', 'stop_lon'},
    'Наряд': {'date', 'vehicle_id'},
    'Расписание': {'trip_id', 'stop_sequence', 'arrival_time', 'departure_time'},
}
DATE_FIELDS = ('date', 'start_date', 'end_date', 'actual_date',
               'route_date_start', 'route_date_end')


def _cells(row: ET.Element, strings: list[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for cell in row.findall('s:c', NS):
        coordinate = cell.get('r', '')
        match = re.fullmatch(r'([A-Z]+)[1-9][0-9]*', coordinate)
        if match is None:
            raise ValueError('invalid spreadsheet cell reference')
        value = cell.find('s:v', NS)
        text = value.text if value is not None else None
        kind = cell.get('t')
        if kind == 's' and text is not None:
            index = int(text)
            if index < 0 or index >= len(strings):
                raise ValueError('invalid shared string reference')
            text = strings[index]
        elif kind == 'inlineStr':
            text = ''.join(node.text or '' for node in cell.findall('.//s:t', NS))
        elif kind == 'e':
            raise ValueError('spreadsheet error cell')
        if cell.find('s:f', NS) is not None:
            raise ValueError('reference inventory must contain literal values')
        result[match[1]] = text
    return result


def _date_summary(rows: list[dict[str, str | None]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in DATE_FIELDS:
        values = [value for row in rows if (value := row.get(field))]
        if values:
            result[field] = {'min': min(values), 'max': max(values),
                             'count': len(values), 'unique_count': len(set(values)),
                             'semantics': 'source_declared; not proof of historical availability'}
    return result


def _workbook(data: bytes) -> tuple[dict[str, list[dict[str, str | None]]], dict[str, Any]]:
    tables: dict[str, list[dict[str, str | None]]] = {}
    metadata: dict[str, Any] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as workbook:
        if len(workbook.namelist()) != len(set(workbook.namelist())):
            raise ValueError('duplicate workbook members')
        strings = []
        if 'xl/sharedStrings.xml' in workbook.namelist():
            strings = [''.join(t.text or '' for t in node.findall('.//s:t', NS))
                       for node in ET.fromstring(workbook.read('xl/sharedStrings.xml'))]
        relationships = {
            node.get('Id'): node.get('Target')
            for node in ET.fromstring(workbook.read('xl/_rels/workbook.xml.rels'))
            if node.get('TargetMode') != 'External'
        }
        root = ET.fromstring(workbook.read('xl/workbook.xml'))
        for sheet in root.findall('s:sheets/s:sheet', NS):
            name = sheet.get('name', '')
            if name not in SHEETS:
                continue
            target = relationships.get(sheet.get(RELATIONSHIP))
            if target is None:
                raise ValueError('missing internal sheet relationship')
            path = (target.lstrip('/') if target.startswith('/') else
                    posixpath.normpath(posixpath.join('xl', target)))
            if not path.startswith('xl/'):
                raise ValueError('sheet relationship escapes workbook')
            root_sheet = ET.fromstring(workbook.read(path))
            rows = root_sheet.findall('s:sheetData/s:row', NS)
            header_index = next((index for index, row in enumerate(rows[:10])
                                 if SHEETS[name] <= set(_cells(row, strings).values())), None)
            if header_index is None:
                raise ValueError(f'missing reference headers in {name}')
            header = {column: value for column, value in _cells(rows[header_index], strings).items()
                      if value}
            if len(header.values()) != len(set(header.values())):
                raise ValueError('duplicate reference column names')
            records = []
            for row in rows[header_index + 1:]:
                cells = _cells(row, strings)
                record = {field: cells.get(column) for column, field in header.items()}
                if any(value is not None and value != '' for value in record.values()):
                    records.append(record)
            tables[name] = records
            metadata[name] = {'header_row': int(rows[header_index].get('r', '0')),
                              'headers': header, 'data_rows': len(records),
                              'date_provenance': _date_summary(records)}
    missing = SHEETS.keys() - tables.keys()
    if missing:
        raise ValueError(f'missing reference sheets: {sorted(missing)}')
    return tables, metadata


def _required(row: dict[str, str | None], field: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f'missing reference value: {field}')
    return value


def _coordinate(value: str | None) -> float | None:
    if value is None or value == '':
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('nonfinite stop coordinate')
    return number


def catalog_inventory(source: Path) -> dict[str, Any]:
    """Return source-declared route visits without certifying dates, trips or anchors."""
    with zipfile.ZipFile(source) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate archive member names')
        data = archive.read(MEMBER)
    tables, metadata = _workbook(data)
    stops = {_required(row, 'stop_id'): row for row in tables['Остановки GTFS_STOPS']}
    routes = {_required(row, 'route_id'): row for row in tables['Маршруты GTFS_ROUTES']}
    if len(stops) != len(tables['Остановки GTFS_STOPS']) or len(routes) != len(
            tables['Маршруты GTFS_ROUTES']):
        raise ValueError('duplicate reference stop or route identity')
    grouped: dict[tuple[str, str, str], list[dict[str, str | None]]] = defaultdict(list)
    rows = tables['Порядок_остановок GTFS_TRIPS_ST']
    for row in rows:
        grouped[(_required(row, 'route_short_name'), _required(row, 'trip_id'),
                 _required(row, 'direction_id'))].append(row)
    patterns = []
    for (route, trip, direction), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(_required(row, 'stop_sequence')))
        sequences = [int(_required(row, 'stop_sequence')) for row in ordered]
        if len(sequences) != len(set(sequences)) or min(sequences) < 0:
            raise ValueError('duplicate or negative pattern visit sequence')
        declared_start = {_required(row, 'start_date') for row in ordered}
        declared_end = {row.get('end_date') or None for row in ordered}
        route_ids = {_required(row, 'route_id') for row in ordered}
        if len(declared_start) != 1 or len(declared_end) != 1 or len(route_ids) != 1:
            raise ValueError('conflicting pattern version metadata')
        start, end = next(iter(declared_start)), next(iter(declared_end))
        if end is not None and date.fromisoformat(end) < date.fromisoformat(start):
            raise ValueError('reversed declared pattern validity')
        date.fromisoformat(start)
        visits = []
        for row, sequence in zip(ordered, sequences, strict=True):
            stop_id = _required(row, 'stop_id')
            stop = stops.get(stop_id, {})
            visits.append({'sequence': sequence, 'stop_id': stop_id,
                           'stop_name': stop.get('stop_name'),
                           'latitude': _coordinate(stop.get('stop_lat')),
                           'longitude': _coordinate(stop.get('stop_lon')),
                           'actual_date': row.get('actual_date'),
                           'stop_mode': row.get('stop_mode'),
                           'is_addpoint': row.get('is_addpoint')})
        patterns.append({'pattern_id': trip, 'route': route, 'direction': direction,
                         'declared_valid_from': start, 'declared_valid_to': end,
                         'validity_boundaries': 'inclusivity_unknown',
                         'known_at': None, 'historical_verified': False,
                         'status': 'pattern_version_unverified', 'visits': visits,
                         'actual_dates': sorted({_required(row, 'actual_date')
                                                 for row in ordered})})
    def keys(items: list[dict[str, str | None]]) -> set[tuple[str | None, ...]]:
        return {tuple(row.get(field) for field in ('trip_id', 'direction_id',
                                                  'stop_sequence', 'stop_id')) for row in items}

    return {'schema_version': 'boarding-catalog-inventory.v1', 'source_member': MEMBER,
            'source_member_sha256': hashlib.sha256(data).hexdigest(),
            'historical_verified': False, 'known_at': None, 'independent_anchors': 0,
            'schedule_observed': False, 'schedule_trip_binding_validated': False,
            'historical_status': 'unverified; source dates do not establish known_at',
            'sheets': metadata, 'pattern_count': len(patterns), 'patterns': patterns,
            'integrity': {'missing_stop_references': sum(row.get('stop_id') not in stops
                                                        for row in rows),
                          'missing_route_references': sum(row.get('route_id') not in routes
                                                         for row in rows),
                          'coordinate_order_key_sets_equal': keys(rows) == keys(
                              tables['Порядок_с_координатами'])}}

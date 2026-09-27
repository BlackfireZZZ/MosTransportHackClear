"""Reconcile inferred allocation against the original source-hour ledger and v1."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from tramflow_ml.boarding.audit import digest, write_json


def audit(run: Path, baseline: Path, source_ledger: Path, out: Path) -> None:
    manifest = json.loads((run / 'manifest.json').read_text())
    prior_manifest = json.loads((baseline / 'manifest.json').read_text())
    assert manifest['complete'] and prior_manifest['complete']
    assert manifest['source_manifest_sha256'] == prior_manifest['source_manifest_sha256']
    original = pd.read_csv(source_ledger, dtype={'route': str})
    original['day'] = original.event_hour.str[:10]
    original['hour'] = original.event_hour.str[11:13].astype(int)
    original = original.groupby(['day', 'route', 'hour']).source_success.sum()
    rows, methods, basis, anchors = [], Counter(), Counter(), Counter()
    assignments = 0
    checks = 0
    for day in manifest['dates']:
        for name in ('soft.csv.gz', 'ledger.csv.gz', 'trips.csv.gz'):
            assert digest(run / 'days' / day / name) == manifest['daily_receipts'][day][name]
        assert (digest(baseline / 'days' / day / 'ledger.csv.gz')
                == prior_manifest['daily_receipts'][day]['ledger.csv.gz'])
        counts = pd.read_csv(run / 'days' / day / 'soft.csv.gz', dtype={'route': str})
        ledger = pd.read_csv(run / 'days' / day / 'ledger.csv.gz', dtype={'route': str})
        old = pd.read_csv(baseline / 'days' / day / 'ledger.csv.gz', dtype={'route': str})
        trips = pd.read_csv(run / 'days' / day / 'trips.csv.gz')
        assert not counts.duplicated(['day', 'route', 'hour', 'stop_id', 'direction']).any()
        assert counts.expected_count.ge(0).all() and counts.expected_count.mod(1).eq(0).all()
        assert counts.event_hour.str.startswith(day).all()
        actual = counts.groupby(['route', 'hour']).expected_count.sum()
        ledger_totals = ledger.groupby(['route', 'hour'])['count'].sum()
        assert actual.equals(ledger_totals)
        known = counts.loc[~counts.stop_id.str.startswith('unallocated:')]
        known_totals = known.groupby(['route', 'hour']).expected_count.sum()
        mapped = ledger.loc[ledger.status.eq('assigned')]
        mapped_totals = mapped.groupby(['route', 'hour'])['count'].sum()
        assert known_totals.equals(mapped_totals)
        expected = original.loc[day]
        keys = actual.index.union(expected.index)
        assert actual.reindex(keys, fill_value=0).equals(expected.reindex(keys, fill_value=0))
        checks += len(keys)
        for route, part in ledger.groupby('route'):
            previous = old.loc[old.route.eq(route)]
            rows.append(dict(day=day, route=route, total=int(part['count'].sum()),
                             old_assigned=int(previous.loc[previous.status.eq('assigned'),
                                                           'count'].sum()),
                             new_assigned=int(part.loc[part.status.eq('assigned'), 'count'].sum())))
        for key, count in ledger.groupby('status')['count'].sum().items():
            methods[str(key)] += int(count)
        assigned = ledger.loc[ledger.status.eq('assigned')]
        for key, count in assigned.groupby('schedule_basis')['count'].sum().items():
            basis[str(key)] += int(count)
        for key, count in assigned.groupby('allocation_basis')['count'].sum().items():
            anchors[str(key)] += int(count)
        assignments += int(trips.assigned.sum())
    frame = pd.DataFrame(rows)
    total = int(frame.total.sum())
    assigned = int(frame.new_assigned.sum())
    old_assigned = int(frame.old_assigned.sum())
    assert total == manifest['source_success']
    assert assigned == assignments == manifest['source_decoded']
    frame['month'] = frame.day.str[:7]
    splits = {}
    for dimension in ('route', 'month'):
        table = frame.groupby(dimension)[['total', 'old_assigned', 'new_assigned']].sum()
        table['old_coverage'] = table.old_assigned / table.total
        table['new_coverage'] = table.new_assigned / table.total
        splits[dimension] = table.reset_index().to_dict(orient='records')
    result = dict(
        schema='anchored-stop-mapping-audit.v1', method=manifest['method'],
        source_success=total, old_assigned=old_assigned, new_assigned=assigned,
        additional_assigned=assigned-old_assigned, old_coverage=old_assigned/total,
        new_coverage=assigned/total, remaining=total-assigned, status_counts=dict(methods),
        assigned_schedule_basis=dict(basis), assigned_allocation_basis=dict(anchors),
        original_route_hour_keys_checked=checks, daily_slices=frame.to_dict(orient='records'),
        slices=splits, source_ledger_sha256=digest(source_ledger),
        baseline_manifest_sha256=digest(baseline/'manifest.json'),
        new_manifest_sha256=digest(run/'manifest.json'), audit_code_sha256=digest(Path(__file__)),
        observed_stop_accuracy=None, first_stop_is_user_assumption=True,
        calendar_transfer_explicitly_authorized=True,
        comparison='historical validation allocation; not forecast known-mass percentage',
    )
    write_json(out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('daily_slices', 'slices')},
                     indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'baseline', 'source-ledger', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.baseline, args.source_ledger, args.out)

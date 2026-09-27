"""Finalize aggregate coverage and receipts for a completed duty mapping run."""
import argparse
import json
import hashlib
import subprocess
from pathlib import Path

import pandas as pd

from tramflow_ml.boarding.audit import digest, write_json


def finalize(run: Path) -> None:
    manifest = json.loads((run / 'manifest.json').read_text())
    if not manifest.get('complete'):
        raise ValueError('complete mapper output required')
    rows, receipts = [], {}
    for day in manifest['dates']:
        folder = run / 'days' / day
        receipts[day] = {name: digest(folder / name)
                         for name in ('soft.csv.gz', 'ledger.csv.gz', 'trips.csv.gz')}
        ledger = pd.read_csv(folder / 'ledger.csv.gz')
        counts = pd.read_csv(folder / 'soft.csv.gz')
        source = ledger.groupby(['route', 'hour'])['count'].sum()
        allocated = counts.groupby(['route', 'hour']).expected_count.sum()
        if not source.equals(allocated):
            raise ValueError('route-hour mass mismatch')
        grouped = ledger.groupby(['day', 'route', 'status'])['count'].sum().reset_index()
        rows.append(grouped)
    coverage = pd.concat(rows, ignore_index=True)
    coverage.to_csv(run / 'coverage-by-day-route.csv', index=False)
    coverage.groupby(['route', 'status'])['count'].sum().reset_index().to_csv(
        run / 'coverage-by-route.csv', index=False)
    assigned = int(coverage.loc[coverage.status.eq('assigned'), 'count'].sum())
    if assigned != manifest['source_decoded']:
        raise ValueError('assigned mass mismatch')
    manifest['daily_receipts'] = receipts
    manifest['coverage_sha256'] = {name: digest(run / name) for name in
                                  ('coverage-by-day-route.csv', 'coverage-by-route.csv')}
    manifest['finalizer_sha256'] = digest(Path(__file__))
    manifest['mass_contract'] = 'soft files include explicitly unallocated route mass; no renormalization'
    manifest['source_coverage'] = 'all successful source events on every listed date'
    manifest['stop_rule'] = (
        'mandatory first-stop anchor; stable adaptive onsets correct subsequent schedule; '
        'previous-stop half-open allocation within bounded trip windows'
        if manifest['method'].startswith('duty-calendar-clock.first-stop-local-anchors') else
        'nearest scheduled arrival within 90s stable under +/-15s, unique trip only')
    manifest['stop_rule_calibrated'] = False
    manifest['service_notice_policy_applied'] = False
    manifest['historical_operation_confirmed'] = False
    manifest['reference'] = 'https://gtfs.org/documentation/schedule/reference/'
    write_json(run / 'manifest.json', manifest)


def provenance(run: Path, base: str) -> None:
    """Supplement an existing run without rewriting its consumed manifest."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((run / 'manifest.json').read_text())
    initial = digest(run / 'manifest.json')
    snapshot = run / 'implementation.py'
    if digest(snapshot) != manifest['implementation_sha256']:
        raise ValueError('executed mapper snapshot differs')
    output = run / 'provenance.json'
    if output.exists():
        raise ValueError('provenance receipt already exists')
    revision = subprocess.check_output(
        ['git', 'rev-parse', base], cwd=root, text=True).strip()
    sources = {}
    folder = run / 'implementation-sources'
    folder.mkdir(exist_ok=False)
    paths = sorted((root / 'ml/src/tramflow_ml/boarding').glob('*.py')) + [root / 'uv.lock']
    for path in paths:
        relative = str(path.relative_to(root))
        if path.name == 'duty_mapping.py':
            content = snapshot.read_bytes()
        else:
            content = subprocess.check_output(['git', 'show', revision + ':' + relative], cwd=root)
            if content != path.read_bytes():
                raise ValueError('dependency differs from recorded base: ' + relative)
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        sources[relative] = hashlib.sha256(content).hexdigest()
    finalizer = subprocess.check_output(
        ['git', 'show', '54eeb378e0c2432ea4c685b3c281b1eb060b5bab:scripts/rebuild_duty_mapping.py'],
        cwd=root)
    if hashlib.sha256(finalizer).hexdigest() != manifest['finalizer_sha256']:
        raise ValueError('original finalizer differs from run receipt')
    (folder / 'original-finalizer.py').write_bytes(finalizer)
    write_json(output, {
        'schema': 'duty-mapping-implementation-provenance.v1',
        'manifest_sha256': initial, 'dependency_base_revision': revision,
        'dependency_bytes_match_base': True, 'source_sha256': sources,
        'original_finalizer_sha256': manifest['finalizer_sha256'],
        'receipt_writer_sha256': digest(Path(__file__)),
        'executed_mapper': 'implementation.py matches original manifest; not current wrapper',
    })
    if digest(run / 'manifest.json') != initial:
        raise ValueError('dataset manifest changed during provenance audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--provenance-base')
    args = parser.parse_args()
    if args.provenance_base:
        provenance(args.run, args.provenance_base)
    else:
        finalize(args.run)

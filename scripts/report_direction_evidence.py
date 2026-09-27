"""Aggregate evidence and timetable hypotheses without passenger or vehicle identifiers."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tramflow_ml.boarding.audit import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new report output required')
    manifest = json.loads((a.run / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(a.run / name) != expected:
            raise ValueError('run hash mismatch')
    summary = json.loads((a.run / 'summary.json').read_text())
    rows = json.loads((a.run / 'rows.json').read_text())
    a.out.mkdir(parents=True)
    write_json(a.out / 'summary.json', summary)
    write_json(a.out / 'manifest.json', manifest)
    examples = []
    for route in ('11', '12', '17'):
        for period in ('AM', 'PM'):
            selected = [r for r in rows if r['route'] == route and r['period'] == period
                        and r['stable'] and r['fits']['duty']['winner'] is not None]
            for row in sorted(selected, key=lambda r: r['window_id']):
                fit = row['fits']['duty']['fits'][row['fits']['duty']['winner']]
                if fit['continuation'] is None:
                    continue
                examples.append({
                    'selection': 'first date/rank/hour-ordered prior-stable window with feasible suffix per route/period',
                    'window_id': row['window_id'], 'direction': row['fits']['duty']['winner'],
                    'destination': fit['destination'], 'onset_clocks': row['onsets'],
                    'stop_names': fit['stop_names'], 'stop_ids': fit['stop_ids'],
                    'initial_offset_seconds': fit['prefix']['initial_offset'],
                    'suffix_mae_seconds': fit['suffix_mae'],
                    'prefix_size': 4, 'observed_truth': False,
                })
                break
    write_json(a.out / 'examples.json', examples)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    titles = {
        '11': '№11: 0 → Восточное Измайлово\n1 → Останкино',
        '12': '№12: 0 → МЦК Дубровка\n1 → Восточное Измайлово',
        '17': '№17: 0 → Останкино\n1 → Медведково',
    }
    for ax, route in zip(axes, ('11', '12', '17')):
        values = []
        for period in ('AM', 'PM'):
            counts = Counter()
            for row in summary['enrollment']:
                if row['route'] == route:
                    counts.update(row['clock']['periods'][period])
            for prefix in ('', 'stable300_'):
                d0 = 100*counts[prefix+'0']/counts['all_events']
                d1 = 100*counts[prefix+'1']/counts['all_events']
                values.append((d0, d1, 100-d0-d1))
        for i, (d0, d1, missing) in enumerate(values):
            for left, width, color, label in [(0,d0,'#246b88','0'),
                                             (d0,d1,'#bd582d','1'),
                                             (d0+d1,missing,'#d8dad7','?')]:
                ax.barh(i, width, left=left, color=color, height=.65)
                if width >= 12:
                    ax.text(left+width/2, i, f'{label}: {width:.0f}%', ha='center', va='center',
                            color='#ffffff' if label != '?' else '#1f2529', fontsize=10)
        ax.set_yticks(range(4), ['Утро: часы', 'Утро: 0, ±5 мин', 'Вечер: часы', 'Вечер: 0, ±5 мин'])
        ax.invert_yaxis()
        ax.set_xlim(0,100)
        ax.set_title(titles[route], fontsize=11, pad=14)
        ax.set_xlabel('% всех успешных оплат', fontsize=10)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=10)
    fig.suptitle('Направление по коду выхода и расписанию — гипотеза, не GPS', fontsize=14)
    fig.text(.5,.035,'12 дат мая–октября 2025 · утро 07–09, вечер 17–19 · ? — нет однозначного назначения\n'
             '0, ±5 мин: направление совпадает на трёх проверенных сдвигах времени',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0,.13,1,.9), w_pad=2)
    fig.savefig(a.out / 'direction-periods.png', dpi=150)
    plt.close(fig)
    write_json(a.out / 'report-manifest.json', {
        'builder_sha256': digest(Path(__file__)),
        'files': {name: digest(a.out / name) for name in
                  ('summary.json','manifest.json','examples.json','direction-periods.png')},
    })


if __name__ == '__main__':
    main()

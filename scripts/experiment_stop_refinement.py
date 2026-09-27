"""Run the preregistered bounded stop and true next-day comparison."""
import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from tramflow_ml.competition import feature_frame
from tramflow_ml.competition_hybrid import hybrid_predict
from tramflow_ml.stop_models import aggregate, calendar_stop_profile, evaluate
from tramflow_ml.stop_refinement import CONFIG, VERSION, load_verified, predict_refined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--route-export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_verified(args.source, args.route_export)
    results = []
    labels = aggregate(data.values, data.identities, 0).rename(columns={'prediction': 'boardings'})
    for days, origins in ((61, (120,181,243)), (1, (120,127,134,181,188,195,243,250,257))):
        for origin in origins:
            for name in ('stop_calendar', 'stop_bounded', 'route_hybrid'):
                if days == 61 and name == 'route_hybrid':
                    continue
                start = date(2025,1,1) + timedelta(days=origin)
                if name == 'route_hybrid':
                    past = labels.loc[labels.date < start]
                    future = feature_frame(past, start, 1)
                    pred = hybrid_predict(past, future, 'catboost_daily_recency_ensemble_50')
                    observed = labels.loc[labels.date == start].boardings.to_numpy()
                    error = float(np.abs(observed-np.rint(pred)).sum())
                    result = {'route_score': max(0,1-error/observed.sum()),
                              'absolute_error': error, 'actual':float(observed.sum())}
                else:
                    pred = (calendar_stop_profile(data.train,origin,days,blend=True)
                            if name == 'stop_calendar' else predict_refined(data,origin,days))
                    result = evaluate(data,pred,origin)
                results.append({'model': name, 'origin':str(start), 'days':days, **result})
                (args.output/'scores.json').write_text(json.dumps(results,indent=2))
                print(name, start, days, result['route_score'], flush=True)
    (args.output/'manifest.json').write_text(json.dumps({
        'version':VERSION,'config':CONFIG,'source':data.audit,'results':results,
        'module_sha256':hashlib.sha256(Path('ml/src/tramflow_ml/stop_refinement.py').read_bytes()).hexdigest(),
        'timezone':'Europe/Moscow','diagnostic_blind':False,'incumbent_changed':False,
    },indent=2))


if __name__ == '__main__':
    main()

from tramflow_ml.boarding.dense_profiles import (
    fit_dense_profiles,
    lookup_profile,
    summarize_transitions,
    validate_dense_holdout,
)


def row(**changes):
    result = dict(date='2025-01-01', route='1', session_id='s', vehicle_key='v1',
                  from_stop='A', to_stop='B', direction=0, from_state=0, to_state=1,
                  steps=1, observed_seconds=120., expected_seconds=150., residual_seconds=-30.,
                  posterior=.8, agreement=1., from_events=4, to_events=4,
                  identity_conflict=False, applicable_pattern=True, pattern_key='1/version1',
                  skipped_stops=[])
    result.update(changes)
    return result


def training():
    return [row(date=f'2025-01-0{1 + i % 2}', vehicle_key=f'v{1 + i % 2}',
                observed_seconds=value) for i, value in enumerate([118, 120, 120, 122, 120, 121])]


def test_fit_chronology_and_profile_version():
    profile = fit_dense_profiles(training() + [row(date='2025-02-01', observed_seconds=999)],
                                 '2025-02-01')
    assert profile['edge_count'] == 1
    assert profile['rejected_rows']['outside_fit_dates'] == 1
    assert lookup_profile(profile, '2025-01-02', '1/version1', 0) is None
    assert lookup_profile(profile, '2025-02-01', '1/version1', 0) == 120.
    assert lookup_profile(profile, '2025-02-01', '1/version2', 0) is None
    assert lookup_profile(profile, '2025-02-01', '1/version1', 1) is None


def test_robust_outlier_does_not_move_median_or_supply_support():
    profile = fit_dense_profiles(training() + [row(observed_seconds=220)], '2025-02-01')
    edge = profile['edges']['1/version1']['0']
    assert edge['median_seconds'] == 120
    assert edge['support'] == 6
    assert profile['rejected_rows']['robust_outlier'] == 1


def test_requires_observations_dates_and_vehicles():
    assert fit_dense_profiles(training()[:4], '2025-02-01')['edge_count'] == 0
    same_day = [dict(item, date='2025-01-01') for item in training()]
    assert fit_dense_profiles(same_day, '2025-02-01')['edge_count'] == 0
    same_vehicle = [dict(item, vehicle_key='v1') for item in training()]
    assert fit_dense_profiles(same_vehicle, '2025-02-01')['edge_count'] == 0


def test_weak_or_skipping_rows_never_fit():
    rejected = [row(steps=2), row(from_events=2), row(posterior=.49), row(agreement=.5),
                row(identity_conflict=True), row(applicable_pattern=False),
                row(observed_seconds=1000), row(vehicle_key=''), row(observed_seconds=float('nan'))]
    profile = fit_dense_profiles(rejected, '2025-02-01')
    assert profile['edge_count'] == 0
    assert sum(profile['rejected_rows'].values()) == len(rejected)


def test_holdout_excludes_fit_vehicles_and_keeps_bad_predictions():
    profile = fit_dense_profiles(training(), '2025-02-01')
    evaluation = validate_dense_holdout([
        row(date='2025-02-01', vehicle_key='v1'),
        row(date='2025-02-01', vehicle_key='v3', observed_seconds=300),
        row(date='2025-02-02', vehicle_key='v3', observed_seconds=120),
        row(date='2025-01-02', vehicle_key='v3'),
    ], profile)
    assert evaluation['evaluated_rows'] == 2
    assert evaluation['rejected_rows'] == {'before_holdout': 1, 'fit_vehicle': 1}
    assert evaluation['profile_mae_seconds'] == 90.
    assert evaluation['baseline_mae_seconds'] == 90.


def test_empty_holdout_has_null_metrics():
    profile = fit_dense_profiles(training(), '2025-02-01')
    evaluation = validate_dense_holdout([row(date='2025-02-01')], profile)
    assert evaluation['profile_mae_seconds'] is None
    assert evaluation['coverage_of_independent_rows'] is None


def test_circular_skips_use_advances_not_state_subtraction():
    summary = summarize_transitions([
        row(from_state=8, to_state=1, steps=3, skipped_stops=['X', 'Y']),
        row(from_state=8, to_state=1, steps=3, skipped_stops=['X', 'Y'], vehicle_key='v2'),
        row(steps=-1), row(steps=0), row(from_events=1),
    ])
    assert summary['inferred_edge_advances'] == 7
    assert summary['unobserved_intermediate_visits'] == 4
    assert summary['unobserved_visit_fraction'] == 4 / 7
    assert summary['maximum_skipped_chain'] == 2
    assert summary['resets'] == summary['same_visit_transitions'] == 1
    assert summary['dense']['transitions'] == 4
    assert summary['sparse']['transitions'] == 1
    assert summary['repeated_skipped_stop_hotspots'][0]['inferred_unobserved_visits'] == 2
    assert summary['repeated_skipped_stop_hotspots'][0]['vehicles'] == 2


def test_empty_summary_has_null_support():
    summary = summarize_transitions([])
    assert summary['unobserved_visit_fraction'] is None
    assert summary['residual_seconds']['p50'] is None
    assert summary['per_edge'] == []


def test_outlier_vehicle_is_not_independent_of_fit_selection():
    profile = fit_dense_profiles(training() + [row(vehicle_key='outlier', observed_seconds=220)],
                                 '2025-02-01')
    result = validate_dense_holdout([row(date='2025-02-01', vehicle_key='outlier')], profile)
    assert result['evaluated_rows'] == 0
    assert result['rejected_rows']['fit_vehicle'] == 1


def test_ineligible_anchor_cannot_fit_but_holdout_keeps_residual_outlier():
    profile = fit_dense_profiles(training() + [row(anchor_eligible=False)], '2025-02-01')
    assert profile['edges']['1/version1']['0']['support'] == 6
    assert profile['rejected_rows']['ineligible_anchor'] == 1
    result = validate_dense_holdout([row(date='2025-02-01', vehicle_key='new',
                                        anchor_eligible=False, observed_seconds=1000)], profile)
    assert result['evaluated_rows'] == 1
    assert result['profile_mae_seconds'] == 880.

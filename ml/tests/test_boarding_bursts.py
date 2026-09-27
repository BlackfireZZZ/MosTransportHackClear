from tramflow_ml.boarding.bursts import Payment, detect


def test_span_singletons_ties_and_midnight() -> None:
    events = [
        Payment(str(i), 86390 + t, "d", route="1")
        for i, t in enumerate([0, 0, 20, 40, 60, 80, 100, 500])
    ]
    bursts = detect(events, gap=30, max_span=90)
    assert [len(b.keys) for b in bursts] == [6, 1, 1]
    assert sum(b.successful for b in bursts) == len(events)
    assert bursts == detect(reversed(events), gap=30, max_span=90)
    assert bursts[0].start < 86400 < bursts[0].end


def test_device_conflict_missing_and_rejections() -> None:
    events = [
        Payment("a", 1, "d", "v1"),
        Payment("b", 1, "d", "v2"),
        Payment("c", 1, ""),
        Payment("d", 1, ""),
        Payment("e", 2, "d", "v1", success=False),
    ]
    bursts = detect(events)
    assert len(bursts) == 4
    assert sum(b.successful for b in bursts) == 4
    assert sum(b.rejected for b in bursts) == 1
    assert len({k for b in bursts for k in b.keys}) == 5


def test_future_mutation_does_not_change_history() -> None:
    past = [Payment("a", 1, "d"), Payment("b", 10, "d")]
    assert detect(past, cutoff=20) == detect(past + [Payment("c", 21, "d")], cutoff=20)

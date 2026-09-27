# Golden forecast evaluation

Набор содержит небольшие проверяемые срезы для трёх горизонтов и нетипичных
событий. Это не benchmark качества будущей production-модели, а исполняемый
контракт evaluation pipeline.

Gate проходит, если кандидат не хуже наивного baseline по WAPE на каждом
горизонте и покрытие интервалов не ниже 80%. Новая модель должна добавлять сюда
реальные временные holdout-срезы с versioned metadata, не заменяя их synthetic
примерами.

```bash
uv run --package tramflow-ml tramflow-ml evaluate
```

## Input and gate policy

`evaluate` requires all three horizons (`day`, `month`, `year`); unknown horizons,
missing horizons and globally duplicate case IDs raise `ValueError`. Multiple
cases per horizon are allowed when their IDs differ. This is a complete-suite
gate, not a partial-horizon promotion decision.

Every series must be non-empty, aligned, finite and non-negative; predictions
must lie within their inclusive bounds. NaN and either infinity are rejected,
including after a caller mutates a case. The coverage threshold must be finite
and in `[0, 1]`, inclusive. Overflow of aggregates or metrics raises `ValueError`
instead of producing a non-finite report.

Both overall and each horizon must have WAPE no worse than baseline and interval
coverage at least the threshold (default `0.8`). Coverage counts observations,
including zeros, with both interval endpoints inclusive; horizons do not receive
equal weight in overall metrics. Failure of either metric returns `passed=false`.

WAPE is undefined when a slice has zero total actual passenger demand. Evaluation
raises `ValueError` naming that horizon (or `overall`); it does not invent a zero
score, omit the slice or substitute an epsilon. Individual zero observations and
zero-demand cases remain included when their horizon has positive total demand.
Invalid input produces no `EvaluationSummary`; the existing CLI exits nonzero.

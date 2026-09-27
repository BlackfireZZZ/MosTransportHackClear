"""coherent_forecast_runs

Revision ID: 602cc47fd0b9
Revises: 20260919_0001
Create Date: 2026-09-25 01:18:46.751172
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "602cc47fd0b9"
down_revision: str | None = "20260919_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "forecast_runs",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("horizon", sa.String(length=16), nullable=False),
        sa.Column("state", sa.String(length=16), server_default="draft", nullable=False),
        sa.Column("model_version", sa.String(length=128), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("forecast_origin", sa.DateTime(timezone=True), nullable=False),
        sa.Column("data_cutoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_id", sa.String(length=128), nullable=False),
        sa.Column("source_version", sa.String(length=128), nullable=False),
        sa.Column("feature_version", sa.String(length=128), nullable=False),
        sa.Column("entity_version", sa.String(length=128), nullable=False),
        sa.Column("calendar_version", sa.String(length=128), nullable=False),
        sa.Column("graph_version", sa.String(length=128), nullable=True),
        sa.Column("target", sa.String(length=32), nullable=False),
        sa.Column("unit", sa.String(length=32), nullable=False),
        sa.Column("synthetic", sa.Boolean(), nullable=False),
        sa.Column("interval_level", sa.Float(), nullable=True),
        sa.Column("interval_method", sa.String(length=128), nullable=True),
        sa.CheckConstraint(
            "(target IN ('synthetic_boardings', 'validation_count') AND unit = "
            "'event_count') OR (target IN ('boarding_count', 'onboard_load') AND unit = "
            "'passengers')",
            name=op.f("ck_forecast_runs_target_unit"),
        ),
        sa.CheckConstraint(
            "horizon IN ('day', 'month', 'year')", name=op.f("ck_forecast_runs_horizon")
        ),
        sa.CheckConstraint(
            "state IN ('draft', 'published', 'failed')", name=op.f("ck_forecast_runs_state")
        ),
        sa.CheckConstraint(
            "synthetic = (target = 'synthetic_boardings')",
            name=op.f("ck_forecast_runs_synthetic_target"),
        ),
        sa.CheckConstraint(
            "(interval_level IS NULL AND interval_method IS NULL) OR (interval_level IS "
            "NOT NULL AND interval_method IS NOT NULL AND interval_level > 0 AND "
            "interval_level < 1)",
            name=op.f("ck_forecast_runs_interval"),
        ),
        sa.CheckConstraint(
            "data_cutoff <= forecast_origin AND forecast_origin <= generated_at",
            name=op.f("ck_forecast_runs_time_order"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecast_runs")),
        sa.UniqueConstraint(
            "id", "horizon", "model_version", "generated_at", name="uq_forecast_run_identity"
        ),
    )
    op.add_column("forecast_points", sa.Column("run_id", sa.String(length=128), nullable=True))
    op.add_column(
        "forecast_points",
        sa.Column(
            "direction_id",
            sa.String(length=128),
            server_default="legacy-unspecified",
            nullable=False,
        ),
    )
    op.execute("""
        DO $$ BEGIN
          IF EXISTS (SELECT horizon FROM forecast_points GROUP BY horizon
                     HAVING count(DISTINCT (model_version, generated_at)) > 1) THEN
            RAISE EXCEPTION 'Legacy horizon mixes metadata; reconcile before migration';
          END IF;
        END $$
    """)
    op.execute("""
        INSERT INTO forecast_runs
          (id, horizon, state, model_version, generated_at, forecast_origin,
           data_cutoff, dataset_id, source_version, feature_version, entity_version,
           calendar_version, target, unit, synthetic)
        SELECT 'legacy-' || horizon, horizon, 'published', model_version,
          generated_at, generated_at, generated_at, 'legacy-demo', 'legacy-demo',
          'legacy-demo', 'legacy-seed-ids', 'legacy-demo-calendar',
          'synthetic_boardings', 'event_count', true
        FROM forecast_points GROUP BY horizon, model_version, generated_at
    """)
    op.execute("SELECT setval(pg_get_serial_sequence('forecast_points', 'id'), "
               "COALESCE((SELECT max(id) FROM forecast_points), 1))")
    op.execute("UPDATE forecast_points SET run_id = 'legacy-' || horizon")
    op.alter_column("forecast_points", "run_id", nullable=False)
    op.alter_column(
        "forecast_points",
        "lower_bound",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=True,
    )
    op.alter_column(
        "forecast_points",
        "upper_bound",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=True,
    )
    op.alter_column(
        "forecast_points",
        "capacity",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=True,
    )
    op.alter_column(
        "forecast_points",
        "model_version",
        existing_type=sa.VARCHAR(length=64),
        type_=sa.String(length=128),
        existing_nullable=False,
    )
    op.drop_constraint(op.f("uq_forecast_points_route_id"), "forecast_points", type_="unique")
    op.create_index(
        "ix_forecast_run_route_time",
        "forecast_points",
        ["run_id", "route_id", "bucket_start"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_forecast_point_run_entity_time",
        "forecast_points",
        ["run_id", "route_id", "stop_id", "direction_id", "bucket_start"],
        postgresql_nulls_not_distinct=True,
    )
    op.create_foreign_key(
        "fk_forecast_point_run_identity",
        "forecast_points",
        "forecast_runs",
        ["run_id", "horizon", "model_version", "generated_at"],
        ["id", "horizon", "model_version", "generated_at"],
    )
    op.create_check_constraint(
        op.f("ck_forecast_points_prediction"),
        "forecast_points",
        "predicted_passengers >= 0 AND predicted_passengers < 'Infinity'::float8",
    )
    op.create_check_constraint(
        op.f("ck_forecast_points_bounds"),
        "forecast_points",
        "(lower_bound IS NULL AND upper_bound IS NULL) OR (lower_bound IS NOT NULL "
        "AND upper_bound IS NOT NULL AND lower_bound >= 0 AND lower_bound <= "
        "predicted_passengers AND predicted_passengers <= upper_bound AND "
        "upper_bound < 'Infinity'::float8)",
    )
    op.create_check_constraint(
        op.f("ck_forecast_points_capacity"),
        "forecast_points",
        "capacity IS NULL OR (capacity >= 0 AND capacity < 'Infinity'::float8)",
    )
    op.execute("""
        CREATE FUNCTION immutable_forecast_run() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF (to_jsonb(NEW) - 'state') IS DISTINCT FROM (to_jsonb(OLD) - 'state') THEN
            RAISE EXCEPTION 'Forecast run metadata is immutable';
          END IF;
          RETURN NEW;
        END $$
    """)
    op.execute("""CREATE TRIGGER immutable_forecast_run BEFORE UPDATE ON forecast_runs
                  FOR EACH ROW EXECUTE FUNCTION immutable_forecast_run()""")


def downgrade() -> None:
    op.execute("""
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM forecast_runs WHERE id NOT IN
                     ('legacy-day', 'legacy-month', 'legacy-year') OR state <> 'published')
             OR EXISTS (SELECT 1 FROM forecast_points WHERE direction_id <> 'legacy-unspecified'
                        OR capacity IS NULL OR lower_bound IS NULL OR upper_bound IS NULL
                        OR length(model_version) > 64)
             OR EXISTS (SELECT route_id, stop_id, horizon, bucket_start FROM forecast_points
                        GROUP BY route_id, stop_id, horizon, bucket_start HAVING count(*) > 1)
          THEN RAISE EXCEPTION 'Downgrade would lose run semantics; restore backup or forward-fix';
          END IF;
        END $$
    """)
    op.execute("DROP TRIGGER immutable_forecast_run ON forecast_runs")
    op.execute("DROP FUNCTION immutable_forecast_run()")
    for name in ("prediction", "bounds", "capacity"):
        op.drop_constraint(op.f("ck_forecast_points_" + name), "forecast_points", type_="check")
    op.drop_constraint("fk_forecast_point_run_identity", "forecast_points", type_="foreignkey")
    op.drop_constraint("uq_forecast_point_run_entity_time", "forecast_points", type_="unique")
    op.drop_index("ix_forecast_run_route_time", table_name="forecast_points")
    op.create_unique_constraint(
        op.f("uq_forecast_points_route_id"),
        "forecast_points",
        ["route_id", "stop_id", "horizon", "bucket_start"],
        postgresql_nulls_not_distinct=False,
    )
    op.alter_column(
        "forecast_points",
        "model_version",
        existing_type=sa.String(length=128),
        type_=sa.VARCHAR(length=64),
        existing_nullable=False,
    )
    op.alter_column(
        "forecast_points",
        "capacity",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=False,
    )
    op.alter_column(
        "forecast_points",
        "upper_bound",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=False,
    )
    op.alter_column(
        "forecast_points",
        "lower_bound",
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        nullable=False,
    )
    op.drop_column("forecast_points", "direction_id")
    op.drop_column("forecast_points", "run_id")
    op.drop_table("forecast_runs")

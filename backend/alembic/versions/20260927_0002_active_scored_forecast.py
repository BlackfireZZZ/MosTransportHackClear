"""Select scored route-hour publications through an explicit active pointer.

Revision ID: 20260927_0002
Revises: 20260927_0001
"""

import sqlalchemy as sa

from alembic import op

revision = "20260927_0002"
down_revision = "20260927_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "forecast_active_runs",
        sa.Column("route_id", sa.Integer(), nullable=False),
        sa.Column("horizon", sa.String(length=16), nullable=False),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.ForeignKeyConstraint(["route_id"], ["routes.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["run_id"], ["forecast_runs.id"]),
        sa.PrimaryKeyConstraint("route_id", "horizon"),
        sa.CheckConstraint("horizon IN ('day', 'month')", name="active_horizon"),
    )
    op.execute("""
        CREATE FUNCTION complete_scored_route_run(candidate_id text) RETURNS boolean
        LANGUAGE plpgsql STABLE AS $$
        DECLARE run_row forecast_runs%ROWTYPE;
        DECLARE expected_per_route integer;
        DECLARE expected_total integer;
        DECLARE actual_total bigint;
        DECLARE invalid_rows bigint;
        DECLARE complete_routes bigint;
        BEGIN
          SELECT * INTO run_row FROM forecast_runs WHERE id = candidate_id;
          IF NOT FOUND OR run_row.id NOT LIKE 'scored-2025-%'
             OR run_row.state <> 'published' OR run_row.target <> 'validation_count'
             OR run_row.unit <> 'event_count' OR run_row.synthetic
             OR run_row.horizon NOT IN ('day', 'month')
             OR run_row.forecast_origin <> '2025-10-31 21:00:00+00'::timestamptz THEN
            RETURN FALSE;
          END IF;
          expected_per_route := CASE WHEN run_row.horizon = 'day' THEN 1464 ELSE 61 END;
          expected_total := expected_per_route * 10;
          SELECT count(*), count(*) FILTER (WHERE
            r.number IS NULL OR r.number NOT IN ('1','5','7','11','12','17','25','26','28','50')
            OR p.stop_id IS NOT NULL OR p.direction_id <> 'legacy-unspecified'
            OR p.bucket_start < '2025-10-31 21:00:00+00'::timestamptz
            OR p.bucket_start >= '2025-12-31 21:00:00+00'::timestamptz
            OR date_trunc(
              CASE WHEN run_row.horizon = 'day' THEN 'hour' ELSE 'day' END,
              p.bucket_start AT TIME ZONE 'Europe/Moscow'
            ) <> p.bucket_start AT TIME ZONE 'Europe/Moscow'
          ) INTO actual_total, invalid_rows
          FROM forecast_points p LEFT JOIN routes r ON r.id = p.route_id
          WHERE p.run_id = candidate_id;
          SELECT count(*) INTO complete_routes FROM (
            SELECT route_id FROM forecast_points WHERE run_id = candidate_id
            GROUP BY route_id HAVING count(*) = expected_per_route
          ) complete;
          RETURN actual_total = expected_total AND invalid_rows = 0
            AND complete_routes = 10;
        END $$
    """)
    op.execute("""
        CREATE FUNCTION validate_active_scored_run() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NOT complete_scored_route_run(NEW.run_id) OR NOT EXISTS (
            SELECT 1 FROM forecast_runs run
            JOIN forecast_points point ON point.run_id = run.id
            WHERE run.id = NEW.run_id AND run.horizon = NEW.horizon
              AND point.route_id = NEW.route_id
          ) THEN
            RAISE EXCEPTION 'Active scored run must be a complete published route run';
          END IF;
          RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER validate_active_scored_run
        BEFORE INSERT OR UPDATE ON forecast_active_runs
        FOR EACH ROW EXECUTE FUNCTION validate_active_scored_run()
    """)
    op.execute("""
        INSERT INTO forecast_active_runs (route_id, horizon, run_id)
        SELECT DISTINCT ON (p.route_id, r.horizon)
          p.route_id, r.horizon, r.id
        FROM forecast_runs r
        JOIN (SELECT DISTINCT run_id, route_id FROM forecast_points) p ON p.run_id = r.id
        WHERE r.id LIKE 'scored-2025-%' AND r.state = 'published'
          AND r.target = 'validation_count' AND r.unit = 'event_count'
          AND NOT r.synthetic AND r.horizon IN ('day', 'month')
          AND complete_scored_route_run(r.id)
        ORDER BY p.route_id, r.horizon, r.generated_at DESC, r.id DESC
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER validate_active_scored_run ON forecast_active_runs")
    op.execute("DROP FUNCTION validate_active_scored_run()")
    op.execute("DROP FUNCTION complete_scored_route_run(text)")
    op.drop_table("forecast_active_runs")

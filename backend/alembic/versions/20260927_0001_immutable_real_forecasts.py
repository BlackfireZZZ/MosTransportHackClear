"""Protect published validation-count points from in-place changes.

Revision ID: 20260927_0001
Revises: 602cc47fd0b9
"""

from alembic import op

revision = "20260927_0001"
down_revision = "602cc47fd0b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE FUNCTION validate_scored_route_grid() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE expected_per_route integer;
        DECLARE expected_total integer;
        DECLARE invalid_rows bigint;
        BEGIN
          IF NEW.state <> 'published' OR OLD.state = 'published'
             OR NEW.id NOT LIKE 'scored-2025-%' THEN
            RETURN NEW;
          END IF;
          IF NEW.target <> 'validation_count' OR NEW.unit <> 'event_count'
             OR NEW.synthetic OR NEW.horizon NOT IN ('day', 'month') THEN
            RAISE EXCEPTION 'Scored run metadata differs from the route-hour contract';
          END IF;
          expected_per_route := CASE WHEN NEW.horizon = 'day' THEN 1464 ELSE 61 END;
          expected_total := expected_per_route * 10;
          SELECT count(*) INTO invalid_rows
          FROM forecast_points p JOIN routes r ON r.id = p.route_id
          WHERE p.run_id = NEW.id AND (
            r.number NOT IN ('1','5','7','11','12','17','25','26','28','50')
            OR p.stop_id IS NOT NULL OR p.direction_id <> 'legacy-unspecified'
            OR p.bucket_start < '2025-10-31 21:00:00+00'::timestamptz
            OR p.bucket_start >= '2025-12-31 21:00:00+00'::timestamptz
            OR date_trunc(
              CASE WHEN NEW.horizon = 'day' THEN 'hour' ELSE 'day' END,
              p.bucket_start AT TIME ZONE 'Europe/Moscow'
            ) <> p.bucket_start AT TIME ZONE 'Europe/Moscow'
          );
          IF invalid_rows <> 0 OR (
            SELECT count(*) FROM forecast_points WHERE run_id = NEW.id
          ) <> expected_total OR (
            SELECT count(*) FROM (
              SELECT route_id FROM forecast_points WHERE run_id = NEW.id
              GROUP BY route_id HAVING count(*) = expected_per_route
            ) complete_routes
          ) <> 10 THEN
            RAISE EXCEPTION 'Scored run cannot publish an incomplete route-hour grid';
          END IF;
          RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER validate_scored_route_grid
        BEFORE UPDATE ON forecast_runs
        FOR EACH ROW EXECUTE FUNCTION validate_scored_route_grid()
    """)
    op.execute("""
        CREATE FUNCTION immutable_published_validation_run_state() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.target = 'validation_count' AND OLD.state = 'published'
             AND NEW.state <> 'published' THEN
            RAISE EXCEPTION 'Published validation forecast runs cannot be demoted';
          END IF;
          RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER immutable_published_validation_run_state
        BEFORE UPDATE ON forecast_runs
        FOR EACH ROW EXECUTE FUNCTION immutable_published_validation_run_state()
    """)
    op.execute("""
        CREATE FUNCTION immutable_published_validation_point() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE point_run_id text;
        DECLARE previous_run_id text;
        BEGIN
          IF TG_OP = 'DELETE' THEN
            point_run_id := OLD.run_id;
          ELSE
            point_run_id := NEW.run_id;
          END IF;
          IF TG_OP = 'UPDATE' THEN
            previous_run_id := OLD.run_id;
          END IF;
          IF EXISTS (
            SELECT 1 FROM forecast_runs
            WHERE id IN (point_run_id, previous_run_id)
              AND state = 'published' AND target = 'validation_count'
          ) THEN
            RAISE EXCEPTION 'Published validation forecast points are immutable';
          END IF;
          IF TG_OP = 'DELETE' THEN
            RETURN OLD;
          END IF;
          RETURN NEW;
        END $$
    """)
    op.execute("""
        CREATE TRIGGER immutable_published_validation_point
        BEFORE INSERT OR UPDATE OR DELETE ON forecast_points
        FOR EACH ROW EXECUTE FUNCTION immutable_published_validation_point()
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER validate_scored_route_grid ON forecast_runs")
    op.execute("DROP FUNCTION validate_scored_route_grid()")
    op.execute("DROP TRIGGER immutable_published_validation_run_state ON forecast_runs")
    op.execute("DROP FUNCTION immutable_published_validation_run_state()")
    op.execute("DROP TRIGGER immutable_published_validation_point ON forecast_points")
    op.execute("DROP FUNCTION immutable_published_validation_point()")

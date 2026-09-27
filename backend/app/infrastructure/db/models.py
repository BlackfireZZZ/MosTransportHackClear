from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.db.base import Base


class RouteModel(Base):
    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    number: Mapped[str] = mapped_column(String(16), unique=True)
    name: Mapped[str] = mapped_column(String(160))
    color: Mapped[str] = mapped_column(String(7), default="#d9342b")


class StopModel(Base):
    __tablename__ = "stops"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), unique=True)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(24), default="tram")


class RouteStopModel(Base):
    __tablename__ = "route_stops"
    __table_args__ = (UniqueConstraint("route_id", "sequence"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id", ondelete="CASCADE"))
    stop_id: Mapped[int] = mapped_column(ForeignKey("stops.id", ondelete="CASCADE"))
    sequence: Mapped[int] = mapped_column(Integer)


class NetworkEdgeModel(Base):
    __tablename__ = "network_edges"
    __table_args__ = (UniqueConstraint("source_stop_id", "target_stop_id", "mode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_stop_id: Mapped[int] = mapped_column(ForeignKey("stops.id", ondelete="CASCADE"))
    target_stop_id: Mapped[int] = mapped_column(ForeignKey("stops.id", ondelete="CASCADE"))
    mode: Mapped[str] = mapped_column(String(24))
    travel_time_minutes: Mapped[float] = mapped_column(Float)
    capacity_per_hour: Mapped[float] = mapped_column(Float)


class ForecastRunModel(Base):
    __tablename__ = "forecast_runs"
    __table_args__ = (
        UniqueConstraint(
            "id", "horizon", "model_version", "generated_at", name="uq_forecast_run_identity"
        ),
        CheckConstraint("state IN ('draft', 'published', 'failed')", name="state"),
        CheckConstraint("horizon IN ('day', 'month', 'year')", name="horizon"),
        CheckConstraint("synthetic = (target = 'synthetic_boardings')", name="synthetic_target"),
        CheckConstraint(
            "(target IN ('synthetic_boardings', 'validation_count') AND unit = "
            "'event_count') OR (target IN ('boarding_count', 'onboard_load') AND unit = "
            "'passengers')",
            name="target_unit",
        ),
        CheckConstraint(
            "data_cutoff <= forecast_origin AND forecast_origin <= generated_at", name="time_order"
        ),
        CheckConstraint(
            "(interval_level IS NULL AND interval_method IS NULL) OR (interval_level IS "
            "NOT NULL AND interval_method IS NOT NULL AND interval_level > 0 AND "
            "interval_level < 1)",
            name="interval",
        ),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    horizon: Mapped[str] = mapped_column(String(16))
    state: Mapped[str] = mapped_column(String(16), server_default="draft")
    model_version: Mapped[str] = mapped_column(String(128))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    forecast_origin: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    data_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dataset_id: Mapped[str] = mapped_column(String(128))
    source_version: Mapped[str] = mapped_column(String(128))
    feature_version: Mapped[str] = mapped_column(String(128))
    entity_version: Mapped[str] = mapped_column(String(128))
    calendar_version: Mapped[str] = mapped_column(String(128))
    graph_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target: Mapped[str] = mapped_column(String(32))
    unit: Mapped[str] = mapped_column(String(32))
    synthetic: Mapped[bool] = mapped_column(Boolean)
    interval_level: Mapped[float | None] = mapped_column(Float, nullable=True)
    interval_method: Mapped[str | None] = mapped_column(String(128), nullable=True)


class ForecastActiveRunModel(Base):
    __tablename__ = "forecast_active_runs"
    __table_args__ = (
        CheckConstraint("horizon IN ('day', 'month')", name="active_horizon"),
    )

    route_id: Mapped[int] = mapped_column(
        ForeignKey("routes.id", ondelete="CASCADE"), primary_key=True
    )
    horizon: Mapped[str] = mapped_column(String(16), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("forecast_runs.id"))


class ForecastPointModel(Base):
    __tablename__ = "forecast_points"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "route_id",
            "stop_id",
            "direction_id",
            "bucket_start",
            name="uq_forecast_point_run_entity_time",
            postgresql_nulls_not_distinct=True,
        ),
        ForeignKeyConstraint(
            ["run_id", "horizon", "model_version", "generated_at"],
            [
                "forecast_runs.id",
                "forecast_runs.horizon",
                "forecast_runs.model_version",
                "forecast_runs.generated_at",
            ],
            name="fk_forecast_point_run_identity",
        ),
        CheckConstraint(
            "predicted_passengers >= 0 AND predicted_passengers < 'Infinity'::float8",
            name="prediction",
        ),
        CheckConstraint(
            "(lower_bound IS NULL AND upper_bound IS NULL) OR (lower_bound IS NOT NULL "
            "AND upper_bound IS NOT NULL AND lower_bound >= 0 AND lower_bound <= "
            "predicted_passengers AND predicted_passengers <= upper_bound AND "
            "upper_bound < 'Infinity'::float8)",
            name="bounds",
        ),
        CheckConstraint(
            "capacity IS NULL OR (capacity >= 0 AND capacity < 'Infinity'::float8)", name="capacity"
        ),
        Index("ix_forecast_run_route_time", "run_id", "route_id", "bucket_start"),
        Index("ix_forecast_route_horizon_time", "route_id", "horizon", "bucket_start"),
    )

    run_id: Mapped[str] = mapped_column(String(128))
    direction_id: Mapped[str] = mapped_column(String(128), server_default="legacy-unspecified")
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id", ondelete="CASCADE"))
    stop_id: Mapped[int | None] = mapped_column(
        ForeignKey("stops.id", ondelete="CASCADE"), nullable=True
    )
    horizon: Mapped[str] = mapped_column(String(16))
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    predicted_passengers: Mapped[float] = mapped_column(Float)
    lower_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[float | None] = mapped_column(Float, nullable=True)
    capacity: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

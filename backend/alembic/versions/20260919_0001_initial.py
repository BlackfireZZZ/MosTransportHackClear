"""Create network graph and seed a demonstrator forecast.

Revision ID: 20260919_0001
Revises:
Create Date: 2026-09-19
"""

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import sqlalchemy as sa

from alembic import op

revision = "20260919_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    routes = op.create_table(
        "routes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("number", sa.String(length=16), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_routes")),
        sa.UniqueConstraint("number", name=op.f("uq_routes_number")),
    )
    stops = op.create_table(
        "stops",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_stops")),
        sa.UniqueConstraint("name", name=op.f("uq_stops_name")),
    )
    route_stops = op.create_table(
        "route_stops",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("route_id", sa.Integer(), nullable=False),
        sa.Column("stop_id", sa.Integer(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["route_id"],
            ["routes.id"],
            name=op.f("fk_route_stops_route_id_routes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stop_id"], ["stops.id"], name=op.f("fk_route_stops_stop_id_stops"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_route_stops")),
        sa.UniqueConstraint("route_id", "sequence", name=op.f("uq_route_stops_route_id")),
    )
    network_edges = op.create_table(
        "network_edges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_stop_id", sa.Integer(), nullable=False),
        sa.Column("target_stop_id", sa.Integer(), nullable=False),
        sa.Column("mode", sa.String(length=24), nullable=False),
        sa.Column("travel_time_minutes", sa.Float(), nullable=False),
        sa.Column("capacity_per_hour", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(
            ["source_stop_id"],
            ["stops.id"],
            name=op.f("fk_network_edges_source_stop_id_stops"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_stop_id"],
            ["stops.id"],
            name=op.f("fk_network_edges_target_stop_id_stops"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_network_edges")),
        sa.UniqueConstraint(
            "source_stop_id", "target_stop_id", "mode", name=op.f("uq_network_edges_source_stop_id")
        ),
    )
    forecast_points = op.create_table(
        "forecast_points",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("route_id", sa.Integer(), nullable=False),
        sa.Column("stop_id", sa.Integer(), nullable=True),
        sa.Column("horizon", sa.String(length=16), nullable=False),
        sa.Column("bucket_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_passengers", sa.Float(), nullable=False),
        sa.Column("lower_bound", sa.Float(), nullable=False),
        sa.Column("upper_bound", sa.Float(), nullable=False),
        sa.Column("capacity", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["route_id"],
            ["routes.id"],
            name=op.f("fk_forecast_points_route_id_routes"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["stop_id"],
            ["stops.id"],
            name=op.f("fk_forecast_points_stop_id_stops"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_forecast_points")),
        sa.UniqueConstraint(
            "route_id",
            "stop_id",
            "horizon",
            "bucket_start",
            name=op.f("uq_forecast_points_route_id"),
        ),
    )
    op.create_index(
        "ix_forecast_route_horizon_time",
        "forecast_points",
        ["route_id", "horizon", "bucket_start"],
        unique=False,
    )

    op.bulk_insert(
        routes,
        [
            {
                "id": 1,
                "number": "Т1",
                "name": "Метро Сокольники — МЦД Каланчёвская",
                "color": "#d9342b",
            },
            {"id": 2, "number": "Т3", "name": "Черёмушки — Метро Чистые пруды", "color": "#246b88"},
        ],
    )
    stop_rows = [
        {
            "id": 1,
            "name": "Метро Сокольники",
            "latitude": 55.7892,
            "longitude": 37.6796,
            "mode": "interchange",
        },
        {
            "id": 2,
            "name": "Матросская Тишина",
            "latitude": 55.7874,
            "longitude": 37.6938,
            "mode": "tram",
        },
        {
            "id": 3,
            "name": "Бакунинская улица",
            "latitude": 55.7775,
            "longitude": 37.6862,
            "mode": "tram",
        },
        {
            "id": 4,
            "name": "Бауманская",
            "latitude": 55.7724,
            "longitude": 37.6786,
            "mode": "interchange",
        },
        {
            "id": 5,
            "name": "Красносельская",
            "latitude": 55.7808,
            "longitude": 37.6662,
            "mode": "interchange",
        },
        {
            "id": 6,
            "name": "МЦД Каланчёвская",
            "latitude": 55.7765,
            "longitude": 37.6514,
            "mode": "interchange",
        },
        {"id": 7, "name": "Черёмушки", "latitude": 55.6763, "longitude": 37.5681, "mode": "tram"},
        {
            "id": 8,
            "name": "Даниловский рынок",
            "latitude": 55.7112,
            "longitude": 37.6206,
            "mode": "tram",
        },
        {
            "id": 9,
            "name": "Павелецкая",
            "latitude": 55.7290,
            "longitude": 37.6388,
            "mode": "interchange",
        },
        {
            "id": 10,
            "name": "Чистые пруды",
            "latitude": 55.7647,
            "longitude": 37.6387,
            "mode": "interchange",
        },
    ]
    op.bulk_insert(stops, stop_rows)

    route_stop_rows: list[dict[str, Any]] = []
    for sequence, stop_id in enumerate(range(1, 7), 1):
        route_stop_rows.append(
            {"id": sequence, "route_id": 1, "stop_id": stop_id, "sequence": sequence}
        )
    for offset, stop_id in enumerate(range(7, 11), 1):
        route_stop_rows.append(
            {"id": 6 + offset, "route_id": 2, "stop_id": stop_id, "sequence": offset}
        )
    op.bulk_insert(route_stops, route_stop_rows)

    edge_rows: list[dict[str, Any]] = []
    edge_id = 1
    for stop_ids in (range(1, 7), range(7, 11)):
        ids = list(stop_ids)
        for source, target in zip(ids, ids[1:], strict=False):
            for start, end in ((source, target), (target, source)):
                edge_rows.append(
                    {
                        "id": edge_id,
                        "source_stop_id": start,
                        "target_stop_id": end,
                        "mode": "tram",
                        "travel_time_minutes": 5.0,
                        "capacity_per_hour": 720.0,
                    }
                )
                edge_id += 1
    op.bulk_insert(network_edges, edge_rows)

    generated_at = datetime(2026, 9, 19, 9, tzinfo=UTC)
    forecast_rows: list[dict[str, Any]] = []
    forecast_id = 1
    horizon_specs = {
        "day": (24, timedelta(hours=1), datetime(2026, 9, 20, tzinfo=UTC)),
        "month": (30, timedelta(days=1), datetime(2026, 10, 1, tzinfo=UTC)),
        "year": (12, timedelta(days=30), datetime(2026, 10, 1, tzinfo=UTC)),
    }
    for route_id in (1, 2):
        route_factor = 1.0 if route_id == 1 else 0.82
        for horizon, (count, step, start) in horizon_specs.items():
            for index in range(count):
                wave = 46 * math.sin(index / max(count - 1, 1) * math.pi) ** 2
                rush = 54 if horizon == "day" and index in {7, 8, 9, 17, 18, 19} else 0
                predicted = round((72 + wave + rush + (index % 4) * 4) * route_factor, 1)
                forecast_rows.append(
                    {
                        "id": forecast_id,
                        "route_id": route_id,
                        "stop_id": None,
                        "horizon": horizon,
                        "bucket_start": start + step * index,
                        "predicted_passengers": predicted,
                        "lower_bound": round(predicted * 0.86, 1),
                        "upper_bound": round(predicted * 1.14, 1),
                        "capacity": 180.0,
                        "model_version": "graph-baseline-v1",
                        "generated_at": generated_at,
                    }
                )
                forecast_id += 1

            route_stop_ids = range(1, 7) if route_id == 1 else range(7, 11)
            for sequence, stop_id in enumerate(route_stop_ids, 1):
                predicted = round((74 + sequence * 13) * route_factor, 1)
                forecast_rows.append(
                    {
                        "id": forecast_id,
                        "route_id": route_id,
                        "stop_id": stop_id,
                        "horizon": horizon,
                        "bucket_start": start,
                        "predicted_passengers": predicted,
                        "lower_bound": round(predicted * 0.86, 1),
                        "upper_bound": round(predicted * 1.14, 1),
                        "capacity": 180.0,
                        "model_version": "graph-baseline-v1",
                        "generated_at": generated_at,
                    }
                )
                forecast_id += 1
    op.bulk_insert(forecast_points, forecast_rows)


def downgrade() -> None:
    op.drop_index("ix_forecast_route_horizon_time", table_name="forecast_points")
    op.drop_table("forecast_points")
    op.drop_table("network_edges")
    op.drop_table("route_stops")
    op.drop_table("stops")
    op.drop_table("routes")

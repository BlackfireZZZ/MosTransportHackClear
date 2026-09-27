from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["boarding-audit.v1"] = "boarding-audit.v1"
    start: date = date(2025, 1, 1)
    end: date = date(2025, 11, 1)
    chunk_rows: int = Field(default=250000, ge=1, le=1000000)
    pattern_routes: tuple[str, ...] = ("1", "5", "7", "11", "12")
    timezone: Literal["Europe/Moscow"] = "Europe/Moscow"

    @model_validator(mode="after")
    def valid_range(self) -> "AuditConfig":
        if self.end <= self.start:
            raise ValueError("empty source range")
        return self

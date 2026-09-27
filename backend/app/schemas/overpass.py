from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OverpassQueryRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "Overpass QL. The length ceiling is a runtime setting, not part of this schema."
        ),
        examples=["[out:json][timeout:60];node(1);out;"],
    )


class OverpassQueryResponse(BaseModel):
    result: dict[str, Any] = Field(description="The Overpass JSON document, unmodified")
    remark: str | None = Field(
        default=None,
        description=(
            "Set when Overpass answered 200 with a PARTIAL result because the query hit "
            "its own [timeout:]. Lifted out of `result` so it cannot be missed; treating "
            "such a response as complete is the easiest way to publish wrong numbers."
        ),
    )


class OverpassStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    url: str
    reachable: bool
    detail: str | None = Field(
        default=None, description="Why the probe failed; null when the instance answered"
    )

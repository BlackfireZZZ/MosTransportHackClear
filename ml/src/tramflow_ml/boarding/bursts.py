"""Device-level payment groups; boundaries are observations, not stop visits."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Payment:
    event_key: str
    second: float
    device: str
    vehicle: str = ""
    route: str = ""
    exit: str = ""
    success: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.second) or not self.event_key:
            raise ValueError("finite timestamp and event key required")


@dataclass(frozen=True)
class Burst:
    keys: tuple[str, ...]
    start: float
    end: float
    successful: int
    rejected: int
    max_gap: float
    device: str
    route: str
    identity_status: str


def detect(
    events: Iterable[Payment], gap: float = 30, max_span: float = 90, cutoff: float | None = None
) -> tuple[Burst, ...]:
    """Groups only payments from one declared device/vehicle/route/exit tuple.

    Missing device events stay singletons. No clock correction or midnight split.
    The cutoff is exclusive on payment time; source availability remains unknown.
    """
    if not isfinite(gap) or not isfinite(max_span) or gap <= 0 or max_span < gap:
        raise ValueError("require finite 0 < gap <= max_span")
    if cutoff is not None and not isfinite(cutoff):
        raise ValueError("finite cutoff required")
    ordered = sorted(
        (e for e in events if cutoff is None or e.second < cutoff),
        key=lambda e: (e.device, e.vehicle, e.route, e.exit, e.second, e.event_key),
    )
    if len({e.event_key for e in ordered}) != len(ordered):
        raise ValueError("duplicate event key")
    result: list[Burst] = []
    group: list[Payment] = []

    def flush() -> None:
        if group:
            result.append(
                Burst(
                    tuple(e.event_key for e in group),
                    group[0].second,
                    group[-1].second,
                    sum(e.success for e in group),
                    sum(not e.success for e in group),
                    max(
                        (b.second - a.second for a, b in zip(group, group[1:], strict=False)),
                        default=0,
                    ),
                    group[0].device,
                    group[0].route,
                    "device_only_unverified" if group[0].device else "identity_missing",
                )
            )

    for e in ordered:
        if group:
            last = group[-1]
            if (
                not e.device
                or (e.device, e.vehicle, e.route, e.exit)
                != (last.device, last.vehicle, last.route, last.exit)
                or e.second - last.second > gap
                or e.second - group[0].second > max_span
            ):
                flush()
                group = []
        group.append(e)
    flush()
    return tuple(sorted(result, key=lambda b: (b.start, b.keys)))

import gzip
import hashlib
import json
import math
import zlib
from datetime import date, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any

from app.application.services.planning import PlanningUnavailable


class FilePlanningRepository:
    def __init__(self, directory: Path):
        self.directory = directory

    @cached_property
    def data(self) -> dict[str, Any]:
        content = (self.directory / "forecast.json.gz").read_bytes()
        manifest = json.loads((self.directory / "manifest.json").read_text())
        if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("Planning artifact checksum mismatch")
        result: dict[str, Any] = json.loads(gzip.decompress(content))
        rows = result["route_rows"]
        routes = {"1", "5", "7", "11", "12", "17", "25", "26", "28", "50"}
        keys = [(row["route"], row["date"], int(row["hour"])) for row in rows]
        expected = {
            (route, (date(2025, 11, 1) + timedelta(days=day)).isoformat(), hour)
            for route in routes
            for day in range(61)
            for hour in range(24)
        }
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Planning route grid mismatch")
        if any(
            not math.isfinite(float(row["prediction"])) or float(row["prediction"]) < 0
            for row in rows
        ):
            raise ValueError("Invalid route prediction")
        for cells in result["shares"].values():
            if not isinstance(cells, list) or any(
                not isinstance(cell, list)
                or len(cell) != 3
                or not isinstance(cell[0], str)
                or not isinstance(cell[1], str)
                for cell in cells
            ):
                raise ValueError("Invalid spatial cell structure")
            if any(not math.isfinite(cell[2]) or cell[2] < 0 for cell in cells):
                raise ValueError("Invalid spatial share")
            if sum(cell[2] for cell in cells) > 1 + 1e-9:
                raise ValueError("Spatial shares exceed route total")
        for stop in result["stops"]:
            if not -90 <= float(stop["lat"]) <= 90 or not -180 <= float(stop["lon"]) <= 180:
                raise ValueError("Invalid stop coordinates")
        result["provenance"]["artifact_sha256"] = manifest["sha256"]
        return result

    def load(self) -> dict[str, Any]:
        try:
            return self.data
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            IndexError,
            zlib.error,
        ) as error:
            raise PlanningUnavailable("Planning artifact unavailable or invalid") from error

    @cached_property
    def experiment(self) -> dict[str, Any]:
        path = self.directory.parent / "external-traffic/model-variants.json.gz"
        content = path.read_bytes()
        manifest = json.loads((path.parent / "model-variants-manifest.json").read_text())
        if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("Experimental bundle checksum mismatch")
        data: dict[str, Any] = json.loads(gzip.decompress(content))
        if data["cutoff"] != "2025-10-31":
            raise ValueError("Experimental bundle cutoff mismatch")
        if not all(
            isinstance(data.get(field), str) and data[field]
            for field in ("generated_at", "source_version")
        ):
            raise ValueError("Missing experimental provenance")
        if not isinstance(data.get("validation"), dict) or not data["validation"]:
            raise ValueError("Missing experimental evaluation")
        if data["schema"] != "external-forecast-variants.v1":
            raise ValueError("Unsupported experimental model bundle")
        expected = {
            (int(row["route"]), row["date"], int(row["hour"])) for row in self.load()["route_rows"]
        }
        actual = [tuple(key) for key in data["keys"]]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Experimental bundle grid mismatch")
        for branch in ("base", "calendar", "weather", "traffic", "news"):
            values = data["branches"][branch]
            if len(values) != len(expected) or any(
                not math.isfinite(value) or value < 0 for value in values
            ):
                raise ValueError("Invalid experimental predictions")
        data["artifact_sha256"] = hashlib.sha256(content).hexdigest()
        return data

    def experimental(self) -> dict[str, Any]:
        try:
            return self.experiment
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            IndexError,
            zlib.error,
        ) as error:
            raise PlanningUnavailable("Experimental artifact unavailable or invalid") from error

    @cached_property
    def approved_source_data(self) -> dict[str, Any]:
        path = self.directory / "approved-source-variants.json.gz"
        content = path.read_bytes()
        manifest_path = self.directory / "approved-source-variants-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(content).hexdigest()
        if manifest.get("schema") != "approved-source-variants-manifest.v1" or (
            digest != manifest.get("sha256") or manifest.get("rows") != 14640
        ):
            raise ValueError("Approved source artifact checksum or manifest mismatch")
        data: dict[str, Any] = json.loads(gzip.decompress(content))
        if (data.get("schema") != "approved-source-variants.v1"
                or data.get("cutoff") != "2025-10-31"
                or not isinstance(data.get("generated_at"), str)
                or not data["generated_at"]
                or data.get("base_csv_sha256") != self.load()["provenance"]["csv_sha256"]):
            raise ValueError("Approved source artifact does not bind the published forecast")
        expected = {
            (int(row["route"]), row["date"], int(row["hour"]))
            for row in self.load()["route_rows"]
        }
        keys = [tuple(key) for key in data["keys"]]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Approved source grid mismatch")
        names = {"calendar", "weather", "traffic", "events"}
        if set(data["branches"]) != names or set(data["weights"]) != names:
            raise ValueError("Approved source names mismatch")
        if any(
            not isinstance(weight, (int, float)) or isinstance(weight, bool)
            or not math.isfinite(weight) or weight <= 0
            for weight in data["weights"].values()
        ) or sum(data["weights"].values()) > 1:
            raise ValueError("Invalid approved source weights")
        for values in data["branches"].values():
            if len(values) != len(keys) or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 for value in values
            ):
                raise ValueError("Invalid approved source predictions")
        if not all(isinstance(data["availability"].get(name), str)
                   and data["availability"][name] for name in names):
            raise ValueError("Missing approved source availability")
        data["artifact_sha256"] = digest
        return data

    def approved_sources(self) -> dict[str, Any]:
        try:
            return self.approved_source_data
        except (OSError, ValueError, KeyError, TypeError, EOFError, IndexError,
                zlib.error) as error:
            raise PlanningUnavailable("Approved source artifact unavailable or invalid") from error

    @cached_property
    def independent_stops(self) -> dict[str, Any]:
        content = (self.directory / "stop-model.json.gz").read_bytes()
        manifest = json.loads((self.directory / "stop-model-manifest.json").read_text())
        if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("Independent stop artifact checksum mismatch")
        result: dict[str, Any] = json.loads(gzip.decompress(content))
        if (
            result["schema"] != "independent-stop-forecast.v1"
            or result["start_date"] != "2025-11-01"
            or result["days"] != 61
            or result["hours_per_day"] != 24
        ):
            raise ValueError("Unsupported independent stop grid")
        planning = self.load()
        provenance = result["provenance"]
        if not all(
            isinstance(provenance.get(field), str) and provenance[field]
            for field in (
                "model_version",
                "generated_at",
                "stop_predictions_sha256",
                "stop_run_sha256",
                "stop_dataset_audit_sha256",
            )
        ):
            raise ValueError("Missing independent stop provenance")
        if (
            provenance["catalog_sha256"] != planning["provenance"]["catalog_sha256"]
            or provenance["data_cutoff"] != "2025-10-31T23:59:59+03:00"
        ):
            raise ValueError("Independent stop provenance mismatch")
        keys = [(row["route"], row["stop_id"], row["direction"]) for row in result["identities"]]
        expected = {(row["route"], row["stop_id"], row["direction"]) for row in planning["stops"]}
        routes = {row["route"] for row in planning["route_rows"]}
        expected.update((route, f"unallocated:{route}", "-1") for route in routes)
        if (
            len(keys) != len(set(keys))
            or set(keys) != expected
            or result["rows"] != 1464 * len(keys)
        ):
            raise ValueError("Incomplete independent stop catalogue")
        for row in result["identities"]:
            if len(row["values"]) != 1464 or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in row["values"]
            ):
                raise ValueError("Invalid independent stop values")
        provenance["artifact_sha256"] = manifest["sha256"]
        return result

    def stop_model(self) -> dict[str, Any]:
        try:
            return self.independent_stops
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            EOFError,
            IndexError,
            zlib.error,
        ) as error:
            raise PlanningUnavailable("Independent stop artifact unavailable or invalid") from error

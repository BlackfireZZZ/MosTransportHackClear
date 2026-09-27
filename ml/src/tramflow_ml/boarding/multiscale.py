"""Offline adaptive pulse candidates; agreement is not a measured door event."""

from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN  # type: ignore[import-untyped]
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

SCALES = np.array([2, 3, 5, 10, 20, 40, 80])
FEATURES = [f"excess_{w}s" for w in SCALES] + [
    "rise_3s",
    "rise_10s",
    "rise_40s",
    "previous_gap",
    "next_gap",
    "tail_fraction",
]


def validate(times: Any, devices: Any) -> tuple[Any, Any]:
    t, d = np.asarray(times, dtype=float), np.asarray(devices)
    if (
        not len(t)
        or len(t) != len(d)
        or not np.isfinite(t).all()
        or np.any(np.diff(t) < 0)
        or t[-1] - t[0] > 86400
    ):
        raise ValueError("sorted finite same-day events with aligned devices required")
    return t, d


def local_rate(t: Any, at: Any) -> Any:
    left = np.searchsorted(t, at - 80) - np.searchsorted(t, at - 300)
    right = np.searchsorted(t, at + 300) - np.searchsorted(t, at + 80)
    exposure = np.maximum(0, np.minimum(300, at - t[0]) - 80) + np.maximum(
        0, np.minimum(300, t[-1] - at) - 80
    )
    return np.maximum((left + right) / np.maximum(exposure, 1), 1 / 600)


def representation(times: Any, devices: Any) -> dict[str, Any]:
    t, d = validate(times, devices)
    at = np.unique(t)
    count = np.column_stack(
        [
            np.searchsorted(t, at + w, side="left") - np.searchsorted(t, at, side="left")
            for w in SCALES
        ]
    )
    before = np.column_stack([np.searchsorted(t, at) - np.searchsorted(t, at - w) for w in SCALES])
    expected = local_rate(t, at)[:, None] * SCALES
    deviance = 2 * (count * np.log(np.maximum(count, 1) / expected) - count + expected)
    scan = np.sign(count - expected) * np.sqrt(np.maximum(deviance, 0))
    unique_devices = np.zeros_like(count)
    sync_mean = np.zeros_like(count, dtype=float)
    sync_var = np.zeros_like(count, dtype=float)
    known = [v for v in np.unique(d) if str(v) not in {"", "-1"}]
    for device in known:
        dt = t[d == device]
        rate = local_rate(dt, at)
        for j, w in enumerate(SCALES):
            has = np.searchsorted(dt, at + w) > np.searchsorted(dt, at)
            unique_devices[:, j] += has
            prob = 1 - np.exp(-rate * w)
            sync_mean[:, j] += prob
            sync_var[:, j] += prob * (1 - prob)
    sync = (unique_devices - sync_mean) / np.sqrt(np.maximum(sync_var, 0.1))
    sync[unique_devices < 2] = -100
    previous = np.r_[0, np.diff(at)]
    following = np.r_[np.diff(at), 0]
    extra = [np.log((count[:, j] + 0.5) / (before[:, j] + 0.5)) for j in [1, 3, 5]]
    features = np.column_stack(
        [
            np.arcsinh((count - expected) / np.sqrt(expected + 1)),
            *extra,
            np.log1p(previous),
            np.log1p(following),
            (count[:, -1] - count[:, 3]) / np.maximum(count[:, -1], 1),
        ]
    )
    return {
        "times": at,
        "counts": count,
        "features": features,
        "scan": scan,
        "sync": sync,
        "devices": unique_devices,
    }


def peaks(at: Any, scores: Any, widths: Any, threshold: float) -> Any:
    """Greedy score maxima with nonoverlapping forward support, no fixed onset width."""
    candidates = np.flatnonzero(scores > threshold)
    chosen: list[int] = []
    for i in sorted(candidates.tolist(), key=lambda i: (-scores[i], at[i])):
        if all(at[i] + widths[i] <= at[j] or at[j] + widths[j] <= at[i] for j in chosen):
            chosen.append(i)
    return np.asarray(sorted(chosen, key=lambda i: at[i]), dtype=int)


def controls(times: Any, devices: Any, seed: int, kind: str) -> tuple[Any, Any]:
    t, d = validate(times, devices)
    rng = np.random.default_rng(seed)
    out = t.copy()
    block = np.floor(t / 300)
    for b in np.unique(block):
        for device in np.unique(d[block == b]):
            select = (block == b) & (d == device)
            if kind == "uniform":
                out[select] = b * 300 + rng.integers(0, 300, select.sum())
            elif kind == "device_shift":
                out[select] = b * 300 + (t[select] - b * 300 + rng.integers(0, 300)) % 300
            else:
                raise ValueError("unknown control")
    order = np.argsort(out, kind="stable")
    return out[order], d[order]


class FeatureMixture:
    """Unsupervised feature mixture; real/control enrichment is not stop probability."""

    def fit(self, features: Any, real: Any, dates: Any) -> "FeatureMixture":
        if any(str(v) >= "2025-05-01" for v in dates):
            raise ValueError("feature fitting is restricted to Jan-Apr2025")
        x, real = np.asarray(features), np.asarray(real, dtype=bool)
        self.scaler = StandardScaler().fit(x)
        z = self.scaler.transform(x)
        self.selection = []
        candidates = []
        for k in [2, 3, 4, 5, 6]:
            model = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                n_init=2,
                random_state=20260926,
                reg_covar=1e-3,
            ).fit(z)
            self.selection.append(
                {"components": k, "bic": float(model.bic(z)), "converged": bool(model.converged_)}
            )
            candidates.append(model)
        best = int(np.argmin([v["bic"] for v in self.selection]))
        self.model = candidates[best]
        if not self.model.converged_:
            raise ValueError("selected feature mixture did not converge")
        probability = self.model.predict_proba(z)
        a, b = probability[real].sum(axis=0), probability[~real].sum(axis=0)
        self.enrichment = (
            (a + 0.5) / (real.sum() + 0.5 * len(a)) / ((b + 0.5) / ((~real).sum() + 0.5 * len(b)))
        )
        self.component_features = self.scaler.inverse_transform(self.model.means_)
        return self

    def score(self, features: Any) -> Any:
        return self.model.predict_proba(self.scaler.transform(features)) @ np.log(self.enrichment)

    def receipt(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "features": FEATURES,
            "enrichment": self.enrichment.tolist(),
            "component_features": self.component_features.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "weights": self.model.weights_.tolist(),
            "means": self.model.means_.tolist(),
            "covariances": self.model.covariances_.tolist(),
        }


def hdbscan_onsets(times: Any, minimum: int = 5, normalized: bool = True) -> Any:
    t = np.asarray(times, dtype=float)
    if len(t) < minimum:
        return np.array([])
    values = t - t[0]
    if normalized:
        grid = np.arange(np.floor(t[0]), np.ceil(t[-1]) + 2)
        clock = np.r_[0, np.cumsum(local_rate(t, grid)[:-1])]
        values = np.interp(t, grid, clock)
    labels = HDBSCAN(
        min_cluster_size=minimum, min_samples=minimum, cluster_selection_method="leaf", copy=True
    ).fit_predict(values[:, None])
    return np.array([t[labels == label].min() for label in sorted(set(labels) - {-1, -2, -3})])


def consensus(families: dict[str, Any], tolerance: float = 5, votes: int = 2) -> Any:
    """At most one event per family in a bounded-diameter group; no chaining."""
    pending = sorted((float(t), family) for family, times in families.items() for t in times)
    out = []
    while pending:
        best: tuple[int, float, list[int]] | None = None
        for i, (start, _) in enumerate(pending):
            members: dict[str, int] = {}
            for j in range(i, len(pending)):
                t, family = pending[j]
                if t - start > tolerance:
                    break
                members.setdefault(family, j)
            item = (len(members), -start, list(members.values()))
            if best is None or item[:2] > best[:2]:
                best = item
        if best is None or best[0] < votes:
            break
        selected = set(best[2])
        out.append(float(np.median([pending[i][0] for i in selected])))
        pending = [v for i, v in enumerate(pending) if i not in selected]
    return np.array(sorted(out))


def detect_all(
    times: Any, devices: Any, mixture: FeatureMixture, thresholds: dict[str, Any]
) -> dict[str, Any]:
    t, d = validate(times, devices)
    r = representation(t, d)
    at, scan, sync = r["times"], r["scan"], r["sync"]
    out = {}
    for j, w in enumerate(SCALES[:4]):
        ids = peaks(at, scan[:, j], np.full(len(at), w), thresholds[f"scan{w}"])
        out[f"scan{w}"] = at[ids]
    best = np.argmax(scan, axis=1)
    widths = SCALES[best]
    strength = scan[np.arange(len(at)), best]
    for suffix, q in [("", "95"), ("_q90", "90"), ("_q99", "99")]:
        ids = peaks(at, strength, widths, thresholds["adaptive"][q])
        out["adaptive_scan" + suffix] = at[ids]
    score = mixture.score(r["features"])
    ids = peaks(at, score, widths, thresholds["gmm"])
    out["feature_gmm"] = at[ids]
    sync_index = np.argmax(sync[:, :4], axis=1)
    ids = peaks(at, sync[np.arange(len(at)), sync_index], SCALES[sync_index], thresholds["sync"])
    out["synchrony"] = at[ids]
    for minimum in [3, 5, 8]:
        out[f"hdbscan{minimum}"] = hdbscan_onsets(t, minimum)
    out["hdbscan5_raw"] = hdbscan_onsets(t, 5, False)
    families = {k: out[k] for k in ["adaptive_scan", "feature_gmm", "hdbscan5", "synchrony"]}
    for tolerance in [2, 5, 10]:
        out[f"consensus2_{tolerance}s"] = consensus(families, tolerance, 2)
    out["consensus3_5s"] = consensus(families, 5, 3)
    out["algorithm_ml"] = consensus({k: out[k] for k in ["adaptive_scan", "hdbscan5"]}, 5, 2)
    return {k: np.sort(v) for k, v in out.items()}

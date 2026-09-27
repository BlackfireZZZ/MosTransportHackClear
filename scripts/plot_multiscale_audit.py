"""Render frozen multiscale aggregate evidence and a deterministic validator raster."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((a.run / "summary.json").read_text())["test"]
    methods = [
        "decay_onset",
        "scan3",
        "scan10",
        "adaptive_scan",
        "hdbscan5",
        "feature_gmm",
        "synchrony",
        "consensus2_5s",
        "consensus3_5s",
    ]
    labels = [
        "Шаблон 15 с",
        "Окно 3 с",
        "Окно 10 с",
        "Адаптивные окна",
        "HDBSCAN",
        "GMM признаков",
        "Синхронность",
        "≥2 семейства",
        "≥3 семейства",
    ]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.bar(
        x - 0.25,
        [summary["methods"][m]["onsets"] for m in methods],
        0.25,
        label="Реальные оплаты",
        color="#286c9b",
    )
    for offset, kind, title, color in [
        (0, "uniform", "Перемешано внутри 5 минут", "#bcc4cc"),
        (0.25, "device_shift", "Сдвиг каждого валидатора", "#c8854a"),
    ]:
        values = [np.mean(summary["methods"][m]["null"][kind]) for m in methods]
        ax.bar(x + offset, values, 0.25, label=title, color=color)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Кандидаты начала; не подтверждённые остановки")
    ax.set_title(
        f"Май–октябрь: {summary['windows']} окон, {summary['source_events']:,} оплат"
    )
    ax.legend(fontsize=9)
    fig.savefig(a.out / "methods.png", dpi=160)
    plt.close(fig)
    windows = json.loads((a.run / "windows.json").read_text())
    rows = json.loads((a.run / "rows.json").read_text())
    index = next(
        i
        for i, w in enumerate(windows)
        if w["date"] == "2025-05-15" and w["route"] == "17" and w["rank"] == 0
    )
    w, row = windows[index], rows[index]
    t, d = np.array(w["times"]), np.array(w["devices"])
    methods = [
        "scan3",
        "adaptive_scan",
        "hdbscan5",
        "feature_gmm",
        "synchrony",
        "consensus2_5s",
    ]
    labels = [
        "3 секунды",
        "Адаптивные окна",
        "HDBSCAN",
        "GMM признаков",
        "Синхронность",
        "Консенсус ≥2",
    ]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 6),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [1, 1]},
    )
    for device in np.unique(d):
        axes[0].plot(
            t[d == device] / 60,
            np.full((d == device).sum(), device + 1),
            "|",
            color="#286c9b",
            markersize=10,
        )
    axes[0].set_ylabel("Валидатор внутри выбранного окна")
    axes[0].set_yticks(np.unique(d) + 1)
    axes[0].set_title("Маршрут 17 • 15 мая 2025 • самое загруженное выбранное окно")
    for j, method in enumerate(methods):
        at = np.array(row["methods"][method]["times"])
        axes[1].plot(at / 60, np.full(len(at), j), "|", markersize=14)
    axes[1].set_yticks(range(len(methods)), labels)
    axes[1].set_xlabel("Минуты от начала выбранной пятиминутной границы")
    axes[1].set_xlim(0, 30)
    axes[1].grid(axis="x", alpha=0.2)
    fig.savefig(a.out / "validators-route17.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

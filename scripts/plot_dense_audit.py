"""Render saved aggregate detector diagnostics without reading individual identities."""

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
    p.add_argument("--example", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    s = json.loads((a.run / "summary.json").read_text())["overall"]
    methods = ["capped30_90", "gap30", "dbscan8", "rate_dbscan", "decay_onset"]
    labels = [
        "Группы 30/90",
        "Пауза 30 с",
        "DBSCAN",
        "DBSCAN / фон",
        "Начало + затухание",
    ]
    x = np.arange(len(methods))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    axes[0].bar(
        x - width / 2,
        [s[m]["bursts"] for m in methods],
        width,
        label="Реальные оплаты",
        color="#286c9b",
    )
    axes[0].bar(
        x + width / 2,
        [s[m]["null_bursts"] for m in methods],
        width,
        label="Контроль: время перемешано внутри 5 минут",
        color="#b8bec5",
    )
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_ylabel("Найденные всплески")
    axes[0].legend(fontsize=8)
    axes[0].set_title("536 одинаковых участков • 90 021 оплата")
    before = []
    early = []
    late = []
    for m in methods:
        h = s[m]["heldout_shape_counts"]
        b = h["before_events"] / 30
        before.append(1)
        early.append(h["early_events"] / 15 / b)
        late.append(h["late_events"] / 30 / b)
    axes[1].plot(x, early, "o-", label="Первые 15 секунд", color="#cf4a39")
    axes[1].plot(x, late, "o-", label="Через 30–60 секунд", color="#286c9b")
    axes[1].axhline(1, color="#777", linestyle="--", label="30 секунд до начала = 1")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_ylabel("Относительная интенсивность")
    axes[1].set_title("Форма на отложенных 20% оплат")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / "detector-comparison.png", dpi=160)
    plt.close(fig)
    e = json.loads(a.example.read_text())
    times = np.array(e["seconds"])
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 5.7),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].hist(
        times, bins=np.arange(0, 1805, 5), color="#286c9b", label="Оплаты за 5 секунд"
    )
    for t in e["onsets"]["decay_onset"]:
        axes[0].axvline(t, color="#d74632", alpha=0.6, lw=1)
    axes[0].set_ylabel("Количество оплат")
    axes[0].set_title(
        "Маршрут №17 • 15 января 2025 • плотный получасовой участок\nКрасные линии: детектор начала с затуханием; это ещё не названия остановок"
    )
    axes[0].legend(loc="upper right")
    for i, m in enumerate(["gap30", "rate_dbscan", "decay_onset"]):
        axes[1].scatter(e["onsets"][m], [i] * len(e["onsets"][m]), marker="|", s=150)
    axes[1].set_yticks([0, 1, 2], ["Пауза 30 с", "DBSCAN / фон", "Начало + затухание"])
    axes[1].set_xticks(np.arange(0, 1801, 300), [str(i) for i in range(0, 31, 5)])
    axes[1].set_xlabel("Минуты от начала участка")
    axes[1].set_xlim(0, 1800)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(a.out / "route17-onsets.png", dpi=160)


if __name__ == "__main__":
    main()

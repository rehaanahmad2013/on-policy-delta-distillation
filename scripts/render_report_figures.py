#!/usr/bin/env python3
"""Render the evidence figures used by the public reproduction report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "reproduction_summary.json").read_text())
OUT = ROOT / "reports" / "opd2-qwen17b" / "images"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {"OPD": "#5065A8", "OPD2": "#E07A5F", "Original": "#9AA0A6"}
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Headline: the exact paper comparison beside the matched reproduction.
fig, ax = plt.subplots(figsize=(9.4, 4.8))
benchmarks = ["MATH-500", "AIME 2024"]
x = np.arange(len(benchmarks))
width = 0.18
series = [
    ("Paper OPD", [DATA["paper"]["opd"]["math500"], DATA["paper"]["opd"]["aime24"]], COLORS["OPD"], "//"),
    ("Paper OPD2", [DATA["paper"]["opd2"]["math500"], DATA["paper"]["opd2"]["aime24"]], COLORS["OPD2"], "//"),
    ("Observed OPD", [DATA["runs"]["opd"]["final"]["math500"], DATA["runs"]["opd"]["final"]["aime24"]], COLORS["OPD"], None),
    ("Observed OPD2", [DATA["runs"]["opd2"]["final"]["math500"], DATA["runs"]["opd2"]["final"]["aime24"]], COLORS["OPD2"], None),
]
for index, (label, values, color, hatch) in enumerate(series):
    bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color, hatch=hatch, alpha=0.95)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)
ax.set_ylabel("Pass@1 (%)")
ax.set_xticks(x, benchmarks)
ax.set_ylim(0, 100)
ax.set_title(DATA["headline"])
ax.legend(ncol=2, frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.18)
save(fig, "headline_benchmarks.png")


# 2. Within-run change from the shared starting checkpoint.
fig, ax = plt.subplots(figsize=(8.8, 4.6))
labels = ["OPD\nMATH-500", "OPD2\nMATH-500", "OPD\nAIME 2024", "OPD2\nAIME 2024"]
values = [
    DATA["runs"]["opd"]["delta"]["math500"],
    DATA["runs"]["opd2"]["delta"]["math500"],
    DATA["runs"]["opd"]["delta"]["aime24"],
    DATA["runs"]["opd2"]["delta"]["aime24"],
]
bars = ax.bar(labels, values, color=[COLORS["OPD"], COLORS["OPD2"], COLORS["OPD"], COLORS["OPD2"]])
ax.axhline(0, color="#333333", linewidth=0.8)
ax.bar_label(bars, fmt="%+.1f pp", padding=3)
ax.set_ylabel("Change from original checkpoint (percentage points)")
ax.set_title("Matched before/after movement after 100 steps")
ax.grid(axis="y", alpha=0.18)
save(fig, "within_run_gains.png")


# 3. Mechanism and rollout behavior over training.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.4, 6.3), sharex=True)
for key, label in (("opd", "OPD"), ("opd2", "OPD2")):
    trajectory = DATA["runs"][key]["trajectory"]
    steps = [row["step"] for row in trajectory]
    lengths = [row["mean_completion_length"] for row in trajectory]
    ax1.plot(steps, lengths, label=label, color=COLORS[label], linewidth=1.8)
ax1.axhline(DATA["setup"]["max_completion_tokens"], color="#555555", linestyle=":", label="4K cap")
ax1.set_ylabel("Mean completion tokens")
ax1.set_title("Rollouts lengthened and frequently reached the bounded 4K cap")
ax1.legend(frameon=False, ncol=3)
ax1.grid(alpha=0.18)
opd2_trajectory = DATA["runs"]["opd2"]["trajectory"]
ax2.plot(
    [row["step"] for row in opd2_trajectory],
    [100 * row["active_fraction"] for row in opd2_trajectory],
    color=COLORS["OPD2"],
    linewidth=1.8,
)
ax2.set_xlabel("Training step")
ax2.set_ylabel("OPD2 tokens retained (%)")
ax2.set_ylim(60, 85)
ax2.set_title("Sign agreement kept a stable subset of OPD2 token rewards")
ax2.grid(alpha=0.18)
save(fig, "training_dynamics.png")


# 4. Runtime comparison, explicitly separating H100 paper evidence from RTX evidence.
fig, ax = plt.subplots(figsize=(9.2, 4.8))
x = np.arange(2)
paper = [DATA["paper"]["runtime_hours"]["opd"], DATA["paper"]["runtime_hours"]["opd2"]]
observed = [DATA["runs"]["opd"]["timing_hours"]["total"], DATA["runs"]["opd2"]["timing_hours"]["total"]]
bars1 = ax.bar(x - 0.18, paper, 0.36, label="Paper: 8× H100", color="#8D99AE", hatch="//")
bars2 = ax.bar(x + 0.18, observed, 0.36, label="Observed: 8× RTX PRO 6000", color="#2A9D8F")
ax.bar_label(bars1, fmt="%.2f h", padding=3)
ax.bar_label(bars2, fmt="%.2f h", padding=3)
ax.set_xticks(x, ["OPD", "OPD2"])
ax.set_ylabel("End-to-end elapsed hours")
ax.set_title("Both 100-step runs completed in a short post-training window")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.18)
save(fig, "runtime_comparison.png")

print(f"Rendered report figures to {OUT}")

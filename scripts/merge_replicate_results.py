#!/usr/bin/env python3
"""Merge terminal orx log JSON from the robustness pair into public results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def result_from_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"=== FINAL_RESULT_JSON ===\n(.*?)\n=== END_FINAL_RESULT_JSON ===",
        text,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"No terminal result JSON in {path}")
    raw = json.loads(match.group(1))
    timing = raw["timing_seconds"]
    return {
        "seed": raw["config"]["seed"],
        "original": raw["metrics"]["original"],
        "final": raw["metrics"]["final"],
        "delta": raw["delta_points"],
        "timing_hours": {
            "training": timing["training"] / 3600,
            "total": timing["total"] / 3600,
            "setup": timing["setup"] / 3600,
            "pre_eval": timing["pre_eval"] / 3600,
            "post_eval": timing["post_eval"] / 3600,
        },
        "trajectory": raw["trajectory"],
        "data": raw["data"],
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--opd-log", type=Path, required=True)
    parser.add_argument("--opd2-log", type=Path, required=True)
    parser.add_argument("--campaign-hours", type=float, required=True)
    args = parser.parse_args()

    data = json.loads(args.summary.read_text(encoding="utf-8"))
    seed2 = {
        "opd": result_from_log(args.opd_log),
        "opd2": result_from_log(args.opd2_log),
    }
    data["replicates"] = {"seed2": seed2}
    data["campaign_wall_hours"] = args.campaign_hours

    aggregate: dict[str, dict] = {}
    for method in ("opd", "opd2"):
        runs = [data["runs"][method], seed2[method]]
        aggregate[method] = {
            "original": {
                benchmark: mean([run["original"][benchmark] for run in runs])
                for benchmark in ("math500", "aime24")
            },
            "final": {
                benchmark: mean([run["final"][benchmark] for run in runs])
                for benchmark in ("math500", "aime24")
            },
            "delta": {
                benchmark: mean([run["delta"][benchmark] for run in runs])
                for benchmark in ("math500", "aime24")
            },
            "timing_hours": {
                key: mean([run["timing_hours"][key] for run in runs])
                for key in ("training", "total", "setup", "pre_eval", "post_eval")
            },
            "mean_trajectory": [
                {
                    key: mean([run["trajectory"][index][key] for run in runs])
                    for key in runs[0]["trajectory"][index]
                }
                for index in range(100)
            ],
        }
    data["aggregate"] = aggregate
    args.summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired image bootstrap and three-level decision for C0/C1/F1/F2."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_microquery_end2end import write_csv
from scripts.train_microquery_end2end import json_safe, resolve_repo_path


RUNS = {
    "C0": "C0_one_query",
    "C1": "C1_independent_aux",
    "F1": "F1_soft_gate",
    "F2": "F2_gate_token",
}
COMPARISONS = (("C0", "C1"), ("C1", "F1"), ("F1", "F2"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/microquery/end2end_full/IRSTD-1k")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict], indices: np.ndarray, name: str) -> float:
    selected = [rows[int(index)] for index in indices]
    value = lambda row, key: float(row.get(key, 0) or 0)
    if name == "global_iou":
        return sum(value(row, "intersection_pixels") for row in selected) / max(
            1.0, sum(value(row, "union_pixels") for row in selected)
        )
    if name == "mean_niou":
        return float(np.mean([value(row, "iou") for row in selected]))
    if name == "f1":
        return float(np.mean([value(row, "f1") for row in selected]))
    if name == "pd":
        return sum(value(row, "detected_components") for row in selected) / max(
            1.0, sum(value(row, "target_components") for row in selected)
        )
    if name == "fa":
        return sum(value(row, "false_pixels") for row in selected) / max(
            1.0, sum(value(row, "pixels") for row in selected)
        )
    if name == "ctr":
        return sum(value(row, "covered_detected") for row in selected) / max(
            1.0, sum(value(row, "covered_components") for row in selected)
        )
    if name == "mask_auprc":
        values = [value(row, "mask_auprc") for row in selected if row.get("mask_auprc") not in (None, "")]
        return float(np.mean(values)) if values else float("nan")
    raise ValueError(name)


def bootstrap(reference, method, samples, seed, comparison):
    names_ref = [row["image"] for row in reference]
    names_method = [row["image"] for row in method]
    if names_ref != names_method:
        raise RuntimeError(f"image order mismatch for {comparison}")
    count = len(reference)
    full = np.arange(count)
    rng = np.random.default_rng(seed)
    rows = []
    for metric_name in ("global_iou", "mean_niou", "f1", "pd", "fa", "ctr", "mask_auprc"):
        observed = metric(method, full, metric_name) - metric(reference, full, metric_name)
        values = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            draw = rng.integers(0, count, size=count)
            values[index] = metric(method, draw, metric_name) - metric(reference, draw, metric_name)
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append(
            {
                "comparison": comparison,
                "metric": metric_name,
                "delta_method_minus_reference": observed,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_samples": samples,
                "seed": seed,
            }
        )
    return rows


def classify(winner: dict, c1: dict, c0: dict, counterfactual: bool) -> str:
    delta_c1_iou = float(winner["global_iou"]) - float(c1["global_iou"])
    delta_c1_niou = float(winner["mean_niou"]) - float(c1["mean_niou"])
    delta_c0_iou = float(winner["global_iou"]) - float(c0["global_iou"])
    delta_c0_niou = float(winner["mean_niou"]) - float(c0["mean_niou"])
    fa_improvement_c1 = (float(c1["fa"]) - float(winner["fa"])) / max(1e-12, float(c1["fa"]))
    pd_delta_c1 = float(winner["pd"]) - float(c1["pd"])
    strong = (
        max(delta_c1_iou, delta_c1_niou) >= 0.005
        and min(delta_c1_iou, delta_c1_niou) >= -0.005
        and fa_improvement_c1 >= 0.10
        and pd_delta_c1 >= -0.0085
        and max(delta_c0_iou, delta_c0_niou) >= 0.005
        and float(winner["fa"]) <= float(c0["fa"])
        and float(winner["pd"]) >= float(c0["pd"])
        and counterfactual
    )
    partial_a = (
        max(delta_c1_iou, delta_c1_niou) >= 0.005
        and (float(winner["fa"]) - float(c1["fa"])) / max(1e-12, float(c1["fa"])) <= 0.05
        and pd_delta_c1 >= -0.0085
    )
    partial_b = (
        fa_improvement_c1 >= 0.10
        and min(delta_c1_iou, delta_c1_niou) >= -0.005
        and pd_delta_c1 >= -0.0085
    )
    if strong:
        return "Strong Success"
    if counterfactual and (partial_a or partial_b):
        return "Useful Partial Success"
    return "Failure"


def main():
    args = parse_args()
    root = resolve_repo_path(args.root)
    tables = {name: read_csv(root / directory / "per_image.csv") for name, directory in RUNS.items()}
    summaries = {}
    for name, directory in RUNS.items():
        with (root / directory / "evaluation_summary.json").open(encoding="utf-8") as handle:
            summaries[name] = json.load(handle)["validation_selected"]
    winner_name = max(("F1", "F2"), key=lambda name: float(summaries[name]["global_iou"]))
    comparisons = list(COMPARISONS) + [("C0", winner_name)]
    bootstrap_rows = []
    for reference, method in comparisons:
        bootstrap_rows.extend(
            bootstrap(
                tables[reference], tables[method], args.bootstrap_samples,
                args.seed + len(bootstrap_rows), f"{method} vs {reference}",
            )
        )
    counterfactual_path = root / "counterfactuals" / f"{RUNS[winner_name].lower()}_counterfactual_summary.json"
    counterfactual_pass = False
    if counterfactual_path.is_file():
        with counterfactual_path.open(encoding="utf-8") as handle:
            values = json.load(handle)
        counterfactual_pass = bool(
            values.get("mechanism_counterfactual_pass")
            and values.get("coordinate_counterfactual_pass")
        )
    outcome = classify(summaries[winner_name], summaries["C1"], summaries["C0"], counterfactual_pass)
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "paired_bootstrap.csv", bootstrap_rows)
    summary = {
        "schema_version": 1,
        "split_role": "validation",
        "test_split_read": False,
        "bootstrap_samples": args.bootstrap_samples,
        "winner_by_validation_global_iou": winner_name,
        "counterfactual_pass": counterfactual_pass,
        "three_level_outcome": outcome,
        "selected_threshold_metrics": json_safe(summaries),
        "follow_up": (
            "Run matched three seeds and the second dataset."
            if outcome == "Strong Success"
            else (
                "Allow one additional seed and one NUAA confirmation; do not claim the paper main model yet."
                if outcome == "Useful Partial Success"
                else "Do not expand to test/multi-seed/second-dataset under this plan."
            )
        ),
    }
    (comparison_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

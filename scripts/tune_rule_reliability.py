#!/usr/bin/env python3
"""Tune A3 rule coefficients from cached validation clusters without rerunning the encoder."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.prompt_metrics import PromptMetricAccumulator
from efficient_sam.prompt_proposal import PromptProposal
from scripts.eval_prompt_quality import write_csv
from sirst_dataset import make_loader


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def reference_at20(directory: Path) -> tuple[dict, dict]:
    budget = next(
        row for row in read_csv(directory / "candidate_budget_curve.csv")
        if int(row["budget"]) == 20
    )
    tiny = next(
        row for row in read_csv(directory / "area_bin_metrics.csv")
        if row["area_bin"] == "1-9" and int(row["budget"]) == 20
    )
    return budget, tiny


def score_row(row: dict, mode: str, alpha: float, beta: float, gamma: float) -> float:
    if mode == "max":
        return float(row["max_score"])
    if mode == "mean":
        return float(row["mean_score"])
    if mode == "support":
        return float(row["support_fraction"])
    return float(
        float(row["mean_score"])
        * float(row["support_fraction"]) ** alpha
        * math.exp(-beta * float(row["center_dispersion"]))
        * math.exp(-gamma * float(row["score_variance"]))
    )


def evaluate_config(
    grouped_rows: dict[str, list[dict]],
    masks: dict[str, torch.Tensor],
    mode: str,
    alpha: float,
    beta: float,
    gamma: float,
    min_support: int,
    max_dispersion: float,
    reliability_threshold: float,
    max_candidates: int,
) -> dict:
    accumulator = PromptMetricAccumulator()
    for name, mask in masks.items():
        candidates = []
        for row in grouped_rows.get(name, []):
            score = score_row(row, mode, alpha, beta, gamma)
            if mode == "rule" and (
                int(row["support_count"]) < min_support
                or float(row["center_dispersion"]) > max_dispersion
                or score < reliability_threshold
            ):
                continue
            candidates.append(
                (score, float(row["x"]), float(row["y"]), int(row["cluster_rank"]))
            )
        candidates.sort(key=lambda item: (-item[0], item[2], item[1], item[3]))
        candidates = candidates[:max_candidates]
        coords = torch.zeros((1, max_candidates, 2), dtype=torch.float32)
        scores = torch.zeros((1, max_candidates), dtype=torch.float32)
        valid = torch.zeros((1, max_candidates), dtype=torch.bool)
        for index, (score, x, y, _) in enumerate(candidates):
            coords[0, index] = torch.tensor((x, y))
            scores[0, index] = score
            valid[0, index] = True
        proposal = PromptProposal(
            dense_logits=None,
            dense_probs=None,
            candidate_xy=coords,
            candidate_scores=scores,
            candidate_valid=valid,
        )
        accumulator.update(proposal, mask.unsqueeze(0), [name])
    return accumulator.finalize()


def summarize(result: dict) -> dict:
    at5 = next(row for row in result["budget_rows"] if int(row["budget"]) == 5)
    at20 = next(row for row in result["budget_rows"] if int(row["budget"]) == 20)
    tiny20 = next(
        row for row in result["area_rows"]
        if row["area_bin"] == "1-9" and int(row["budget"]) == 20
    )
    return {
        "candidate_auprc": float(result["summary"]["candidate_score_auprc"]),
        "recall_at_5": float(at5["component_recall"]),
        "precision_at_5": float(at5["prompt_precision"]),
        "recall_at_20": float(at20["component_recall"]),
        "tiny_recall_at_20": float(tiny20["component_recall"]),
        "false_prompts_per_mp_at_20": float(at20["false_prompts_per_million_pixels"]),
        "mean_candidates_at_20": float(at20["mean_candidates_per_image"]),
        "zero_prompt_fraction": float(at20["zero_prompt_fraction"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--cluster_csv", required=True)
    parser.add_argument("--a1_reference_dir", required=True)
    parser.add_argument("--a2_reference_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max_candidates", type=int, default=32)
    parser.add_argument("--alphas", default="0,0.5,1,1.5,2")
    parser.add_argument("--betas", default="0,0.25,0.5,1")
    parser.add_argument("--gammas", default="0,0.5,1,2")
    parser.add_argument("--min_supports", default="1,2,3")
    parser.add_argument("--max_displacements", default="1.5,2,3")
    parser.add_argument("--reliability_threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.cluster_csv).resolve())
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["image"])].append(row)

    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    loader = make_loader(
        str(data_root),
        str(split_path),
        batch_size=8,
        size=args.size,
        augment=False,
        keep_ratio_pad=False,
        workers=0,
        shuffle=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )
    masks = {}
    for batch in loader:
        for name, mask in zip(list(batch["name"]), batch["mask"]):
            masks[str(name)] = mask
    missing = sorted(set(grouped_rows).difference(masks))
    if missing:
        raise ValueError(f"Cluster CSV contains names absent from split: {missing[:5]}")

    max_result = evaluate_config(
        grouped_rows, masks, "max", 0, 0, 0, 1, math.inf, 0, args.max_candidates
    )
    max_metrics = summarize(max_result)
    a1_budget, a1_tiny = reference_at20(Path(args.a1_reference_dir).resolve())
    a2_budget, _ = reference_at20(Path(args.a2_reference_dir).resolve())
    a1_tiny_recall = float(a1_tiny["component_recall"])
    a2_recall = float(a2_budget["component_recall"])
    a2_false = float(a2_budget["false_prompts_per_million_pixels"])

    sweep_rows = []
    for alpha, beta, gamma, min_support, max_dispersion in itertools.product(
        parse_floats(args.alphas),
        parse_floats(args.betas),
        parse_floats(args.gammas),
        parse_ints(args.min_supports),
        parse_floats(args.max_displacements),
    ):
        result = evaluate_config(
            grouped_rows,
            masks,
            "rule",
            alpha,
            beta,
            gamma,
            min_support,
            max_dispersion,
            args.reliability_threshold,
            args.max_candidates,
        )
        metrics = summarize(result)
        passes = (
            metrics["false_prompts_per_mp_at_20"] <= 0.9 * a2_false
            and metrics["recall_at_20"] >= a2_recall - 0.005
            and metrics["tiny_recall_at_20"] >= a1_tiny_recall
            and metrics["candidate_auprc"] > max_metrics["candidate_auprc"]
        )
        sweep_rows.append(
            {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "min_support": min_support,
                "max_dispersion": max_dispersion,
                "reliability_threshold": args.reliability_threshold,
                **metrics,
                "false_reduction_vs_a2": 1.0 - metrics["false_prompts_per_mp_at_20"] / a2_false,
                "recall_delta_vs_a2": metrics["recall_at_20"] - a2_recall,
                "tiny_delta_vs_a1": metrics["tiny_recall_at_20"] - a1_tiny_recall,
                "auprc_delta_vs_max": metrics["candidate_auprc"] - max_metrics["candidate_auprc"],
                "passes_all_prompt_gates": int(passes),
            }
        )

    passing = [row for row in sweep_rows if row["passes_all_prompt_gates"]]
    passing.sort(
        key=lambda row: (
            row["false_prompts_per_mp_at_20"],
            -row["tiny_recall_at_20"],
            -row["recall_at_20"],
            -row["candidate_auprc"],
            row["min_support"],
            row["alpha"],
            row["beta"],
            row["gamma"],
        )
    )
    selected = passing[0] if passing else None
    payload = {
        "selection_rule": "pass all preregistered prompt gates, then lowest False Prompts/MP@20, highest tiny/overall Recall@20, highest AUPRC",
        "references": {
            "a1_tiny_recall_at_20": a1_tiny_recall,
            "a2_recall_at_20": a2_recall,
            "a2_false_prompts_per_mp_at_20": a2_false,
            "max_score_candidate_auprc": max_metrics["candidate_auprc"],
        },
        "grid_size": len(sweep_rows),
        "passing_configs": len(passing),
        "selected": selected,
    }
    write_csv(output_dir / "rule_grid.csv", sweep_rows)
    (output_dir / "selected_rule.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Zero-training C1/F1 deployment-gate audit on the frozen IRSTD validation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.microquery_gate_deployment import (
    GateDeploymentConfig,
    compute_deployment_gate,
    gate_config_id,
    resolve_gate_parameters,
)
from efficient_sam.microquery_metrics import (
    assign_candidates,
    detected_component_indices,
)
from efficient_sam.prompt_metrics import extract_components
from scripts.eval_microquery_end2end import (
    add_per_image_auprc,
    area_bin_rows,
    build_from_checkpoint,
    candidate_diagnostics,
    evaluate_threshold,
    write_csv,
)
from scripts.microquery_end2end_dataset import MicroQueryEndToEndDataset
from scripts.microquery_end2end_runtime import forward_deployable
from scripts.train_microquery_end2end import (
    DEFAULT_A1,
    DEFAULT_DATA_ROOT,
    DEFAULT_VAL_CACHE,
    DEFAULT_WEIGHTS,
    autocast_context,
    json_safe,
    resolve_repo_path,
    set_deterministic,
    sha256_file,
)


DEFAULT_C1 = "outputs/microquery/end2end_full/IRSTD-1k/C1_independent_aux/best_fixed05_global_iou.pt"
DEFAULT_F1 = "outputs/microquery/end2end_full/IRSTD-1k/F1_soft_gate/best_fixed05_global_iou.pt"
THRESHOLDS = tuple(round(float(value), 2) for value in np.arange(0.05, 1.0, 0.05))
TOTAL_TARGETS = 117
REFERENCE_DETECTED = 106
PD_FLOOR_DETECTED = REFERENCE_DETECTED - 1
PD_FLOOR = PD_FLOOR_DETECTED / TOTAL_TARGETS
FA_REFERENCE = 34.7137451171875e-6
BOOTSTRAP_REPEATS = 2000
AREA_ORDER = ("1-9", "10-16", "17-25", ">25")


@dataclass(frozen=True)
class ConditionSpec:
    short_id: str
    directory: str
    config: GateDeploymentConfig
    legacy: bool = False


def condition_specs() -> tuple[ConditionSpec, ...]:
    return (
        ConditionSpec("A0", "A0", GateDeploymentConfig("all_one")),
        ConditionSpec("R0", "R0", GateDeploymentConfig("raw", rho=0.0, temperature=1.0)),
        ConditionSpec("R1", "R1", GateDeploymentConfig("residual", rho=0.1, temperature=1.0)),
        ConditionSpec("R2", "R2", GateDeploymentConfig("residual", rho=0.2, temperature=1.0)),
        ConditionSpec("R3", "R3", GateDeploymentConfig("residual", rho=0.1, temperature=1.5)),
        ConditionSpec("R4", "R4", GateDeploymentConfig("residual", rho=0.2, temperature=1.5)),
        ConditionSpec(
            "L", "Legacy", GateDeploymentConfig("legacy_checkpoint_schedule"), legacy=True
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1_checkpoint", default=DEFAULT_C1)
    parser.add_argument("--f1_checkpoint", default=DEFAULT_F1)
    parser.add_argument("--output_dir", default="outputs/microquery/gate_deployment_audit/IRSTD-1k")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--val_split", default="splits/experiment1_seed20260825/val.txt")
    parser.add_argument("--val_candidate_cache", default=DEFAULT_VAL_CACHE)
    parser.add_argument("--a1_checkpoint", default=DEFAULT_A1)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--query_chunk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp_dtype", choices=("bfloat16", "off"), default="bfloat16")
    parser.add_argument("--bootstrap_repeats", type=int, default=BOOTSTRAP_REPEATS)
    return parser.parse_args()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_tree_inputs(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


@torch.no_grad()
def collect_base(model, head, variant, epoch, loader, device, args) -> dict:
    """Decode each query once; all 7 deployment policies are then exact algebra."""

    rows = {
        "names": [],
        "image": [],
        "all_one_probability": [],
        "query_probability": [],
        "object_logits": [],
        "gt": [],
        "xy": [],
        "raw": [],
        "valid": [],
        "semantic": [],
        "component_ids": [],
    }
    for batch in loader:
        deployable = {key: value.to(device) for key, value in batch["deployable"].items()}
        supervision = {key: value.to(device) for key, value in batch["supervision"].items()}
        with autocast_context(device, args.amp_dtype):
            output = forward_deployable(
                model,
                head,
                deployable,
                variant=variant,
                gate_deployment_config=GateDeploymentConfig("all_one"),
                query_chunk=args.query_chunk,
            )
        rows["names"].extend(list(batch["meta"]["name"]))
        rows["image"].append(deployable["image"][:, :1].detach().float().cpu().numpy().astype(np.float16))
        rows["all_one_probability"].append(output.final_probability.detach().float().cpu().numpy())
        rows["query_probability"].append(
            torch.sigmoid(output.query_logits).detach().float().cpu().numpy().astype(np.float16)
        )
        rows["object_logits"].append(output.object_logits.detach().float().cpu().numpy())
        rows["gt"].append(supervision["full_mask"].cpu().numpy().astype(np.uint8))
        rows["xy"].append(deployable["candidate_xy"].float().cpu().numpy())
        rows["raw"].append(deployable["candidate_scores"].float().cpu().numpy())
        rows["valid"].append(deployable["candidate_valid"].cpu().numpy().astype(bool))
        rows["semantic"].append(supervision["semantic_labels"].cpu().numpy().astype(bool))
        rows["component_ids"].append(supervision["component_ids"].cpu().numpy())
    return {
        key: (np.asarray(value) if key == "names" else np.concatenate(value, axis=0))
        for key, value in rows.items()
    }


def cache_for_condition(base: dict, config: GateDeploymentConfig, checkpoint_epoch: int) -> dict:
    logits = torch.from_numpy(base["object_logits"])
    valid = torch.from_numpy(base["valid"])
    raw, effective, rho, temperature = compute_deployment_gate(
        logits,
        valid,
        config,
        checkpoint_epoch=checkpoint_epoch if config.mode == "legacy_checkpoint_schedule" else None,
    )
    if config.mode == "all_one":
        probability = base["all_one_probability"].copy()
    else:
        query = base["query_probability"].astype(np.float32)
        probability = np.max(query * effective.numpy()[..., None, None], axis=1).astype(np.float32)
    return {
        "names": base["names"],
        "probability": probability,
        "query_probability": base["query_probability"],
        "gt": base["gt"],
        "xy": base["xy"],
        "raw": base["raw"],
        "valid": base["valid"],
        "semantic": base["semantic"],
        "component_ids": base["component_ids"],
        "gates": effective.numpy().astype(np.float32),
        "raw_gates": raw.numpy().astype(np.float32),
        "rho": rho,
        "temperature": temperature,
    }


def component_metrics(cache: dict, threshold: float) -> tuple[dict, list[dict]]:
    rows: list[dict] = []
    for index, name in enumerate(cache["names"]):
        components = extract_components(cache["gt"][index] > 0)
        assignment = assign_candidates(
            cache["xy"][index], cache["valid"][index], components, budget=10
        )
        detected = detected_component_indices(
            cache["probability"][index] >= float(threshold), components
        )
        for component in components:
            rows.append(
                {
                    "image": str(name),
                    "component_index": int(component.index),
                    "area": int(component.area),
                    "area_bin": (
                        "1-9" if component.area <= 9 else "10-16" if component.area <= 16
                        else "17-25" if component.area <= 25 else ">25"
                    ),
                    "covered": int(component.index in assignment.covered_components),
                    "final_detected": int(component.index in detected),
                }
            )
    tiny = [row for row in rows if row["area_bin"] == "1-9"]
    covered = [row for row in rows if row["covered"]]
    return {
        "target_components": len(rows),
        "detected_components": sum(row["final_detected"] for row in rows),
        "tiny_components": len(tiny),
        "tiny_detected": sum(row["final_detected"] for row in tiny),
        "tiny_pd": float(np.mean([row["final_detected"] for row in tiny])) if tiny else float("nan"),
        "ctr": float(np.mean([row["final_detected"] for row in covered])) if covered else float("nan"),
    }, rows


def threshold_curve(cache: dict) -> tuple[list[dict], dict[float, list[dict]], dict[float, list[dict]]]:
    curve: list[dict] = []
    per_image: dict[float, list[dict]] = {}
    per_component: dict[float, list[dict]] = {}
    for threshold in THRESHOLDS:
        metrics, image_rows = evaluate_threshold(cache, threshold)
        component_summary, component_rows = component_metrics(cache, threshold)
        curve.append({"threshold": threshold, **metrics, **component_summary})
        per_image[threshold] = image_rows
        per_component[threshold] = component_rows
    return curve, per_image, per_component


def matched_pd_selection(rows: Sequence[dict], pd_floor: float = PD_FLOOR) -> dict | None:
    eligible = [row for row in rows if float(row["pd"]) + 1e-12 >= float(pd_floor)]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (float(row["fa"]), -float(row["mean_niou"]), -float(row["global_iou"])),
    )


def matched_fa_selection(rows: Sequence[dict], fa_reference: float = FA_REFERENCE) -> dict | None:
    eligible = [row for row in rows if float(row["fa"]) <= float(fa_reference) + 1e-15]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (float(row["pd"]), float(row["tiny_pd"]), float(row["mean_niou"])),
    )


def pareto_frontier(rows: Sequence[dict]) -> list[dict]:
    frontier = []
    for index, row in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            at_least = (
                float(other["pd"]) >= float(row["pd"])
                and float(other["mean_niou"]) >= float(row["mean_niou"])
                and float(other["fa"]) <= float(row["fa"])
            )
            strict = (
                float(other["pd"]) > float(row["pd"])
                or float(other["mean_niou"]) > float(row["mean_niou"])
                or float(other["fa"]) < float(row["fa"])
            )
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(dict(row))
    return sorted(frontier, key=lambda row: float(row["threshold"]))


def gate_distribution(cache: dict, per_query_rows: Sequence[dict]) -> list[dict]:
    valid = cache["valid"]
    semantic = cache["semantic"] & valid
    groups = {"target": semantic, "background": (~cache["semantic"]) & valid}
    output = []
    for gate_name in ("raw_gates", "gates"):
        values_all = cache[gate_name]
        for group, selected in groups.items():
            values = values_all[selected].astype(np.float64)
            output.append(
                {
                    "gate": "raw" if gate_name == "raw_gates" else "effective",
                    "group": group,
                    "count": int(values.size),
                    "mean": float(values.mean()) if values.size else float("nan"),
                    "std": float(values.std()) if values.size else float("nan"),
                    "q05": float(np.quantile(values, 0.05)) if values.size else float("nan"),
                    "q25": float(np.quantile(values, 0.25)) if values.size else float("nan"),
                    "q50": float(np.quantile(values, 0.50)) if values.size else float("nan"),
                    "q75": float(np.quantile(values, 0.75)) if values.size else float("nan"),
                    "q95": float(np.quantile(values, 0.95)) if values.size else float("nan"),
                }
            )
    query_gate = np.asarray([float(row["object_score"]) for row in per_query_rows])
    query_iou = np.asarray([float(row["best_query_iou"]) for row in per_query_rows])
    false_pixels = np.asarray([float(row["false_mask_pixels"]) for row in per_query_rows])
    output.append(
        {
            "gate": "effective",
            "group": "correlation",
            "count": int(query_gate.size),
            "spearman_best_query_iou": float(spearmanr(query_gate, query_iou).statistic),
            "spearman_false_mask_pixels": float(spearmanr(query_gate, false_pixels).statistic),
        }
    )
    return output


def enriched_area_bins(
    component_rows: Sequence[dict], per_query_rows: Sequence[dict]
) -> list[dict]:
    query_lookup = {
        (str(row["image"]), int(row["component_index"])): row
        for row in per_query_rows
        if str(row["assignment"]) == "primary"
    }
    output = []
    for label in AREA_ORDER:
        rows = [row for row in component_rows if str(row["area_bin"]) == label]
        covered = [row for row in rows if int(row["covered"])]
        qrows = [
            query_lookup[(str(row["image"]), int(row["component_index"]))]
            for row in rows
            if (str(row["image"]), int(row["component_index"])) in query_lookup
        ]
        output.append(
            {
                "area_bin": label,
                "components": len(rows),
                "candidate_coverage": float(np.mean([int(row["covered"]) for row in rows])) if rows else float("nan"),
                "pd": float(np.mean([int(row["final_detected"]) for row in rows])) if rows else float("nan"),
                "ctr": float(np.mean([int(row["final_detected"]) for row in covered])) if covered else float("nan"),
                "best_query_iou": float(np.mean([float(row["best_query_iou"]) for row in rows])) if rows else float("nan"),
                "gate_mean": float(np.mean([float(row["object_score"]) for row in qrows])) if qrows else float("nan"),
                "target_gate_below_0_5": sum(float(row["object_score"]) < 0.5 for row in qrows),
            }
        )
    return output


def plot_condition_curves(curve: Sequence[dict], pareto: Sequence[dict], output_dir: Path) -> None:
    pareto_thresholds = {float(row["threshold"]) for row in pareto}
    for y_key, filename, ylabel in (
        ("pd", "pd_fa_curve.png", "Pd"),
        ("mean_niou", "niou_fa_curve.png", "nIoU"),
    ):
        fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=150)
        x = [float(row["fa_per_million"]) for row in curve]
        y = [float(row[y_key]) for row in curve]
        ax.plot(x, y, marker="o", linewidth=1.2, markersize=3)
        for row in curve:
            if float(row["threshold"]) in pareto_thresholds:
                ax.scatter(float(row["fa_per_million"]), float(row[y_key]), c="tab:red", s=22)
        ax.set_xlabel("Fa × 10⁻⁶")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def condition_summary(
    model_id: str,
    spec: ConditionSpec,
    cache: dict,
    checkpoint_epoch: int,
    anchor: float,
    output_dir: Path,
) -> dict:
    curve, per_image_by_threshold, _ = threshold_curve(cache)
    fixed = next(row for row in curve if math.isclose(float(row["threshold"]), 0.5))
    anchor_row = next(row for row in curve if math.isclose(float(row["threshold"]), anchor))
    matched_pd = matched_pd_selection(curve)
    matched_fa = matched_fa_selection(curve)
    selected = matched_pd or max(curve, key=lambda row: (float(row["pd"]), -float(row["fa"])))
    threshold = float(selected["threshold"])
    image_rows = per_image_by_threshold[threshold]
    add_per_image_auprc(image_rows, cache)
    diagnostics = candidate_diagnostics(cache, "c1_independent_aux", threshold)
    component_rows = diagnostics["per_component_rows"]
    per_query_rows = diagnostics["per_query_rows"]
    per_image_diag = {str(row["image"]): row for row in diagnostics["per_image_rows"]}
    component_by_image: dict[str, list[dict]] = {}
    for row in component_rows:
        component_by_image.setdefault(str(row["image"]), []).append(row)
    for row in image_rows:
        name = str(row["image"])
        diag = per_image_diag[name]
        tiny = [item for item in component_by_image.get(name, []) if str(item["area_bin"]) == "1-9"]
        row.update(
            {
                "condition": f"{model_id}-{spec.short_id}",
                "covered_components": int(diag["covered_components"]),
                "covered_detected": int(diag["covered_detected"]),
                "tiny_target_components": len(tiny),
                "tiny_detected_components": sum(int(item["final_detected"]) for item in tiny),
            }
        )
    for row in per_query_rows:
        rank = int(row["candidate_rank"]) - 1
        image_index = int(np.where(cache["names"] == str(row["image"]))[0][0])
        row["raw_gate"] = float(cache["raw_gates"][image_index, rank])
        row["effective_gate"] = float(cache["gates"][image_index, rank])
        row["object_score"] = row["effective_gate"]
    areas = enriched_area_bins(component_rows, per_query_rows)
    pareto = pareto_frontier(curve)
    rho, temperature = resolve_gate_parameters(
        spec.config,
        checkpoint_epoch=checkpoint_epoch if spec.legacy else None,
    )
    resolved = {
        "schema_version": 1,
        "zero_training": True,
        "split_role": "validation",
        "test_split_read": False,
        "model": model_id,
        "condition": spec.short_id,
        "gate_config": asdict(spec.config),
        "resolved_rho": rho,
        "resolved_temperature": temperature,
        "gate_config_id": gate_config_id(
            spec.config,
            checkpoint_epoch=checkpoint_epoch if spec.legacy else None,
        ),
        "checkpoint_epoch": checkpoint_epoch,
        "thresholds": list(THRESHOLDS),
        "anchor_threshold": anchor,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(output_dir / "resolved_config.json", resolved)
    dump_json(output_dir / "fixed05_summary.json", fixed)
    dump_json(output_dir / "anchor_threshold_summary.json", anchor_row)
    dump_json(output_dir / "matched_pd_summary.json", matched_pd)
    dump_json(output_dir / "matched_fa_summary.json", matched_fa)
    write_csv(output_dir / "threshold_curve.csv", curve)
    write_csv(output_dir / "pd_fa_pareto.csv", pareto)
    write_csv(output_dir / "per_image.csv", image_rows)
    write_csv(output_dir / "per_component.csv", component_rows)
    write_csv(output_dir / "per_query.csv", per_query_rows)
    write_csv(output_dir / "gate_distribution.csv", gate_distribution(cache, per_query_rows))
    write_csv(output_dir / "area_bin_metrics.csv", areas)
    plot_condition_curves(curve, pareto, output_dir)
    return {
        "model": model_id,
        "condition": spec.short_id,
        "explicit": not spec.legacy and spec.short_id != "A0",
        "rho": rho,
        "temperature": temperature,
        "anchor": anchor_row,
        "fixed05": fixed,
        "matched_pd": matched_pd,
        "matched_fa": matched_fa,
        "selected": selected,
        "per_image": image_rows,
        "area_bins": areas,
        "cache": cache,
    }


def select_common_explicit(results: dict[str, dict[str, dict]]) -> dict:
    candidates = []
    for condition in ("R0", "R1", "R2", "R3", "R4"):
        reasons = []
        fa_values = []
        for model in ("C1", "F1"):
            current = results[model][condition]["matched_pd"]
            baseline = results[model]["A0"]["matched_pd"]
            if current is None or baseline is None:
                reasons.append(f"{model}: no matched-Pd point")
                continue
            if int(current["detected_components"]) < PD_FLOOR_DETECTED:
                reasons.append(f"{model}: Pd loses more than one component")
            if int(current["tiny_detected"]) < int(baseline["tiny_detected"]) - 1:
                reasons.append(f"{model}: tiny 1-9 loses more than one component")
            if float(current["ctr"]) < float(baseline["ctr"]) - 0.0085:
                reasons.append(f"{model}: CTR drop exceeds 0.85pp")
            if float(current["mean_niou"]) < float(baseline["mean_niou"]) - 0.005:
                reasons.append(f"{model}: nIoU drop exceeds 0.5pp")
            if float(current["f1"]) < float(baseline["f1"]) - 0.005:
                reasons.append(f"{model}: F1 drop exceeds 0.5pp")
            fa_values.append(float(current["fa"]))
        candidates.append(
            {
                "condition": condition,
                "safe": not reasons,
                "reasons": reasons,
                "mean_matched_pd_fa": float(np.mean(fa_values)) if fa_values else float("inf"),
            }
        )
    safe = [row for row in candidates if row["safe"]]
    pool = safe or candidates
    chosen = min(pool, key=lambda row: float(row["mean_matched_pd_fa"]))
    return {
        "selected_condition": chosen["condition"],
        "paper_safe": bool(chosen["safe"]),
        "selection_pool": "strict_safe" if safe else "diagnostic_fallback",
        "candidates": candidates,
    }


def paired_bootstrap(
    first: Sequence[dict], second: Sequence[dict], repeats: int, seed: int
) -> list[dict]:
    if [row["image"] for row in first] != [row["image"] for row in second]:
        raise ValueError("paired bootstrap requires identical image order")
    rng = np.random.default_rng(seed)
    n = len(first)
    metrics = {key: [] for key in ("global_iou", "mean_niou", "f1", "pd", "fa", "tiny_pd")}

    def aggregate(rows, indices):
        chosen = [rows[int(index)] for index in indices]
        return {
            "global_iou": sum(row["intersection_pixels"] for row in chosen) / max(1, sum(row["union_pixels"] for row in chosen)),
            "mean_niou": float(np.mean([row["iou"] for row in chosen])),
            "f1": float(np.mean([row["f1"] for row in chosen])),
            "pd": sum(row["detected_components"] for row in chosen) / max(1, sum(row["target_components"] for row in chosen)),
            "fa": sum(row["false_pixels"] for row in chosen) / max(1, sum(row["pixels"] for row in chosen)),
            "tiny_pd": sum(row["tiny_detected_components"] for row in chosen) / max(1, sum(row["tiny_target_components"] for row in chosen)),
        }

    for _ in range(int(repeats)):
        indices = rng.integers(0, n, size=n)
        a = aggregate(first, indices)
        b = aggregate(second, indices)
        for key in metrics:
            metrics[key].append(a[key] - b[key])
    output = []
    for key, values in metrics.items():
        array = np.asarray(values)
        output.append(
            {
                "metric": key,
                "mean_difference": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_difference_positive": float((array > 0).mean()),
                "repeats": int(repeats),
            }
        )
    return output


def counterfactual_rows(result: dict, threshold: float) -> list[dict]:
    cache = result["cache"]
    valid = cache["valid"]
    gates = cache["gates"]
    conditions = {
        "correct": gates,
        "candidate_shuffled": np.roll(gates, 1, axis=1),
        "batch_shuffled": np.roll(gates, 1, axis=0),
        "inverted": np.where(valid, 1.0 - gates, 0.0),
        "all_one": valid.astype(np.float32),
    }
    output = []
    for name, intervention in conditions.items():
        condition_cache = dict(cache)
        condition_cache["gates"] = np.where(valid, intervention, 0.0).astype(np.float32)
        condition_cache["probability"] = np.max(
            cache["query_probability"].astype(np.float32)
            * condition_cache["gates"][..., None, None],
            axis=1,
        )
        metrics, _ = evaluate_threshold(condition_cache, threshold)
        output.append({"condition": name, **metrics})
    return output


def export_cases(results: dict[str, dict[str, dict]], selected_id: str, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    c1_a0 = results["C1"]["A0"]
    c1_gate = results["C1"][selected_id]
    f1_a0 = results["F1"]["A0"]
    f1_gate = results["F1"][selected_id]
    c1_raw = results["C1"]["R0"]
    c1_residual = results["C1"]["R2"]
    scenarios = {
        "gate_reduces_fa_keeps_target": lambda i: (
            c1_gate["per_image"][i]["detected_components"] >= c1_a0["per_image"][i]["detected_components"]
            and c1_gate["per_image"][i]["false_pixels"] < c1_a0["per_image"][i]["false_pixels"]
        ),
        "gate_suppresses_true_target": lambda i: c1_gate["per_image"][i]["detected_components"] < c1_a0["per_image"][i]["detected_components"],
        "c1_mask_survives_f1_gate_loses": lambda i: c1_a0["per_image"][i]["detected_components"] > f1_gate["per_image"][i]["detected_components"],
        "c1_gate_better_than_f1": lambda i: c1_gate["per_image"][i]["iou"] > f1_gate["per_image"][i]["iou"] + 0.02,
        "f1_gate_better_than_c1": lambda i: f1_gate["per_image"][i]["iou"] > c1_gate["per_image"][i]["iou"] + 0.02,
        "raw_fails_residual_recovers": lambda i: c1_residual["per_image"][i]["detected_components"] > c1_raw["per_image"][i]["detected_components"],
        "residual_restores_background_fa": lambda i: c1_residual["per_image"][i]["false_pixels"] > c1_raw["per_image"][i]["false_pixels"],
    }
    manifest = []
    base = c1_a0["cache"]
    for scenario, predicate in scenarios.items():
        candidates = [index for index in range(len(base["names"])) if predicate(index)]
        if not candidates:
            manifest.append({"scenario": scenario, "status": "no qualifying validation image"})
            continue
        index = max(
            candidates,
            key=lambda i: abs(c1_gate["per_image"][i]["iou"] - c1_a0["per_image"][i]["iou"])
            + abs(f1_gate["per_image"][i]["iou"] - f1_a0["per_image"][i]["iou"]),
        )
        name = str(base["names"][index])
        query = base["query_probability"][index].astype(np.float32)
        montage = np.concatenate([np.concatenate(list(query[j:j + 5]), axis=1) for j in (0, 5)], axis=0)
        raw_prob = c1_raw["cache"]["probability"][index]
        residual_prob = c1_residual["cache"]["probability"][index]
        fig, axes = plt.subplots(3, 3, figsize=(10, 9), dpi=150)
        panels = [
            (base["image"][index, 0], "image"),
            (base["gt"][index], "GT"),
            (base["image"][index, 0], "candidate points / gate"),
            (montage, "per-query masks (2×5)"),
            (c1_a0["cache"]["probability"][index], "all-one final"),
            (raw_prob, "raw final"),
            (residual_prob, "residual final"),
            (residual_prob - raw_prob, "residual − raw"),
            (c1_gate["cache"]["probability"][index], f"selected {selected_id}"),
        ]
        for axis, (panel, title) in zip(axes.flat, panels):
            axis.imshow(panel, cmap="gray")
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        axis = axes.flat[2]
        gates = c1_gate["cache"]["gates"][index]
        for rank, ((x, y), gate, valid) in enumerate(zip(base["xy"][index], gates, base["valid"][index])):
            if valid:
                axis.scatter(x, y, c=[[1.0 - gate, gate, 0.0]], s=20)
                axis.text(x + 2, y, f"{rank + 1}:{gate:.2f}", color="yellow", fontsize=5)
        fig.suptitle(f"{scenario}: {name}")
        fig.tight_layout()
        path = output_dir / f"{scenario}__{name}.png"
        fig.savefig(path)
        plt.close(fig)
        manifest.append({"scenario": scenario, "status": "exported", "image": name, "file": path.name})
    write_csv(output_dir / "case_manifest.csv", manifest)
    return manifest


def classify_outcome(results: dict[str, dict[str, dict]], selected_id: str, decision: dict) -> dict:
    best_improvement = -float("inf")
    for model in ("C1", "F1"):
        gate = results[model][selected_id]["matched_pd"]
        base = results[model]["A0"]["matched_pd"]
        if gate and base:
            best_improvement = max(best_improvement, 1.0 - float(gate["fa"]) / max(1e-15, float(base["fa"])))
    if decision["paper_safe"] and best_improvement >= 0.10:
        level = "Strong Gate Success"
        next_stage = "允许 IRSTD 第二 seed 与 NUAA 单 seed"
    elif decision["paper_safe"] and best_improvement >= 0.05:
        level = "Useful Gate Calibration"
        next_stage = "允许额外一个 IRSTD seed 与 NUAA；gate 仅作为部署校准"
    else:
        level = "Gate Failure"
        next_stage = "停止 soft-gate 主线；保留 C1 independent-query + per-query supervision"
    return {
        "level": level,
        "best_matched_pd_fa_relative_reduction": best_improvement,
        "next_stage": next_stage,
    }


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    data_root = resolve_repo_path(args.data_root)
    val_split = Path(args.val_split).resolve() if Path(args.val_split).is_absolute() else data_root / args.val_split
    val_cache = resolve_repo_path(args.val_candidate_cache)
    a1 = resolve_repo_path(args.a1_checkpoint)
    weights = resolve_repo_path(args.weights)
    checkpoints = {"C1": resolve_repo_path(args.c1_checkpoint), "F1": resolve_repo_path(args.f1_checkpoint)}
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "zero_training": True,
        "test_split_read": False,
        "seed": args.seed,
        "thresholds": list(THRESHOLDS),
        "pd_reference": REFERENCE_DETECTED / TOTAL_TARGETS,
        "pd_floor_absolute": {"detected_components": PD_FLOOR_DETECTED, "targets": TOTAL_TARGETS, "pd": PD_FLOOR},
        "fa_reference": FA_REFERENCE,
        "bootstrap_repeats": args.bootstrap_repeats,
        "paths": {
            "data_root": str(data_root), "val_split": str(val_split), "val_candidate_cache": str(val_cache),
            "a1_checkpoint": str(a1), "weights": str(weights), **{f"{key.lower()}_checkpoint": str(value) for key, value in checkpoints.items()},
        },
        "sha256": {
            "val_split": sha256_file(val_split), "val_candidate_cache": sha256_file(val_cache),
            "a1_checkpoint": sha256_file(a1), "weights": sha256_file(weights),
            **{f"{key.lower()}_checkpoint": sha256_file(value) for key, value in checkpoints.items()},
        },
    }
    dump_json(output_dir / "protocol" / "resolved_protocol.json", protocol)
    dataset = MicroQueryEndToEndDataset(
        data_root=data_root, split=val_split, candidate_cache=val_cache, augment=False, budget=10, seed=args.seed
    )
    if len(dataset) != 80:
        raise RuntimeError(f"frozen validation split must contain 80 images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    results: dict[str, dict[str, dict]] = {"C1": {}, "F1": {}}
    for model_id, expected_variant, anchor in (
        ("C1", "c1_independent_aux", 0.4),
        ("F1", "f1_soft_gate", 0.15),
    ):
        checkpoint, variant, model, head = build_from_checkpoint(checkpoints[model_id], a1, weights, device)
        if variant != expected_variant:
            raise RuntimeError(f"{model_id} checkpoint variant mismatch: {variant}")
        epoch = int(checkpoint["epoch"])
        base = collect_base(model, head, variant, epoch, loader, device, args)
        for spec in condition_specs():
            cache = cache_for_condition(base, spec.config, epoch)
            directory = "LegacyPredicted" if model_id == "C1" and spec.short_id == "L" else spec.directory
            results[model_id][spec.short_id] = condition_summary(
                model_id, spec, cache, epoch, anchor, output_dir / model_id / directory
            )
        del model, head, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_rows = []
    for model_id in ("C1", "F1"):
        for spec in condition_specs():
            result = results[model_id][spec.short_id]
            row = {
                "id": f"{model_id}-{spec.short_id}", "model": model_id, "condition": spec.short_id,
                "rho": result["rho"], "temperature": result["temperature"],
            }
            for prefix in ("anchor", "fixed05", "matched_pd", "matched_fa"):
                values = result[prefix]
                if values:
                    for key in ("threshold", "global_iou", "mean_niou", "f1", "pd", "fa_per_million", "tiny_pd", "ctr"):
                        if key in values:
                            row[f"{prefix}_{key}"] = values[key]
            summary_rows.append(row)
    write_csv(output_dir / "comparison" / "full_gate_matrix.csv", summary_rows)
    decision = select_common_explicit(results)
    selected_id = str(decision["selected_condition"])
    selected_spec = next(spec for spec in condition_specs() if spec.short_id == selected_id)
    decision["selected_gate_config"] = asdict(selected_spec.config)
    attribution = []
    for model in ("C1", "F1"):
        for condition in ("A0", selected_id):
            values = results[model][condition]["matched_pd"]
            attribution.append({"model": model, "inference": "all_one" if condition == "A0" else "predicted_gate", **values})
    write_csv(output_dir / "comparison" / "attribution_2x2.csv", attribution)
    comparisons = (
        (f"C1-{selected_id}_vs_C1-A0", results["C1"][selected_id], results["C1"]["A0"]),
        (f"F1-{selected_id}_vs_F1-A0", results["F1"][selected_id], results["F1"]["A0"]),
        (f"F1-{selected_id}_vs_C1-{selected_id}", results["F1"][selected_id], results["C1"][selected_id]),
        ("F1-A0_vs_C1-A0", results["F1"]["A0"], results["C1"]["A0"]),
        (f"F1-{selected_id}_vs_F1-L", results["F1"][selected_id], results["F1"]["L"]),
    )
    all_bootstrap = []
    for index, (name, first, second) in enumerate(comparisons):
        rows = paired_bootstrap(first["per_image"], second["per_image"], args.bootstrap_repeats, args.seed + index)
        for row in rows:
            row["comparison"] = name
        all_bootstrap.extend(rows)
        write_csv(output_dir / "bootstrap" / f"{name}.csv", rows)
    counterfactual = []
    for model in ("C1", "F1"):
        result = results[model][selected_id]
        threshold = float(result["matched_pd"]["threshold"])
        for row in counterfactual_rows(result, threshold):
            counterfactual.append({"model": model, "gate": selected_id, "threshold": threshold, **row})
    write_csv(output_dir / "comparison" / "gate_counterfactuals.csv", counterfactual)
    area_rows = []
    for model, condition in (("C1", "A0"), ("C1", selected_id), ("F1", "A0"), ("F1", selected_id), ("F1", "L")):
        for row in results[model][condition]["area_bins"]:
            area_rows.append({"model": model, "condition": condition, **row})
    write_csv(output_dir / "area_bins" / "priority_conditions.csv", area_rows)
    cases = export_cases(results, selected_id, output_dir / "cases")
    outcome = classify_outcome(results, selected_id, decision)
    decision.update({"outcome": outcome, "test_split_read": False, "zero_training": True, "case_exports": cases})
    dump_json(output_dir / "comparison" / "decision.json", decision)
    dump_json(
        output_dir / "comparison" / "audit_summary.json",
        {"protocol": protocol, "decision": decision, "full_gate_matrix": summary_rows, "bootstrap": all_bootstrap},
    )
    print(json.dumps(json_safe(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

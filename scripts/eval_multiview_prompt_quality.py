#!/usr/bin/env python3
"""Evaluate A2/A3 shared-weight multi-view prompt proposals without GT sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.multiview_prompt import DEFAULT_VIEWS, multiview_propose, rule_reliability
from efficient_sam.prompt_metrics import PromptMetricAccumulator
from efficient_sam.prompt_proposal import DenseHeadProposalAdapter, DoGLoGProposalAdapter, PGAPProposalAdapter
from sirst_dataset import make_loader
from scripts.eval_prompt_quality import (
    build_probe,
    enforce_test_config_freeze,
    parse_budgets,
    select_probe_features,
    set_deterministic,
    sha256_file,
    write_csv,
)


class ProbeImageGenerator:
    """Image-only boundary around a frozen encoder and spatial probe."""

    def __init__(self, model, head, level: str, adapter: DenseHeadProposalAdapter, size: int):
        self.model = model
        self.head = head
        self.level = level
        self.adapter = adapter
        self.size = int(size)

    def __call__(self, images: torch.Tensor):
        encoded = self.model.get_image_embeddings(images)
        neck, multi_scale = encoded[0], encoded[2]
        features = select_probe_features(self.level, neck, multi_scale)
        return self.adapter(features, output_size=(self.size, self.size))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split_role", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frozen_config_manifest", default=None)
    parser.add_argument("--generator", choices=("pgap", "doglog", "probe"), required=True)
    parser.add_argument("--probe_checkpoint", default=None)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--candidate_k_raw", type=int, default=32)
    parser.add_argument("--nms_radius", type=float, default=3.0)
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--cluster_radius", type=float, default=3.0)
    parser.add_argument("--gate", choices=("none", "rule"), default="none")
    parser.add_argument("--score_mode", choices=("mean_max", "mean", "max", "support"), default="mean_max")
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--max_dispersion", type=float, default=2.0)
    parser.add_argument("--reliability_threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--budgets", default="1,3,5,10,20,32")
    parser.add_argument("--max_batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_test_config_freeze(args.split_role, args.frozen_config_manifest)
    if args.generator == "probe" and not args.probe_checkpoint:
        raise ValueError("--generator probe requires --probe_checkpoint")
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    if not views:
        raise ValueError("--views cannot be empty")
    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(
        str(data_root),
        str(split_path),
        size=args.size,
        batch_size=args.batch_size,
        augment=False,
        shuffle=False,
        workers=args.workers,
        keep_ratio_pad=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=True,
        sc_use_noise=False,
        sc_use_gamma=False,
        sc_eval_crop="resize",
        mllm_features_path=None,
    )

    probe_info = None
    if args.generator == "pgap":
        generator = PGAPProposalAdapter(
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
        ).to(device)
    elif args.generator == "doglog":
        generator = DoGLoGProposalAdapter(
            args.candidate_k_raw, args.nms_radius, args.score_threshold
        ).to(device)
    else:
        checkpoint_path = Path(args.probe_checkpoint).resolve()
        model, head, level, checkpoint = build_probe(
            checkpoint_path, Path(args.weights).resolve(), device
        )
        adapter = DenseHeadProposalAdapter(
            head,
            candidate_k_raw=args.candidate_k_raw,
            nms_radius=args.nms_radius,
            score_threshold=args.score_threshold,
        )
        generator = ProbeImageGenerator(model, head, level, adapter, args.size)
        probe_info = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)),
            "level": level,
        }

    accumulator = PromptMetricAccumulator(budgets=parse_budgets(args.budgets))
    cluster_rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=f"multiview:{args.gate}")):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            proposal, clusters_per_image, _, _ = multiview_propose(
                images,
                generator,
                views=views,
                cluster_radius=args.cluster_radius,
                max_candidates=args.candidate_k_raw,
                gate=args.gate,
                min_support=args.min_support,
                max_dispersion=args.max_dispersion,
                reliability_threshold=args.reliability_threshold,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                score_mode=args.score_mode,
            )
            names = list(batch["name"])
            accumulator.update(proposal, batch["mask"], names)
            for name, clusters in zip(names, clusters_per_image):
                for rank, cluster in enumerate(clusters, start=1):
                    reliability = rule_reliability(
                        cluster, alpha=args.alpha, beta=args.beta, gamma=args.gamma
                    )
                    accepted = args.gate == "none" or (
                        cluster.support_count >= args.min_support
                        and cluster.center_dispersion <= args.max_dispersion
                        and reliability >= args.reliability_threshold
                    )
                    cluster_rows.append(
                        {
                            "image": name,
                            "cluster_rank": rank,
                            "x": cluster.center_xy[0],
                            "y": cluster.center_xy[1],
                            "mean_score": cluster.mean_score,
                            "max_score": cluster.max_score,
                            "score_variance": cluster.score_variance,
                            "support_count": cluster.support_count,
                            "support_fraction": cluster.support_fraction,
                            "center_dispersion": cluster.center_dispersion,
                            "rank_mean": cluster.rank_mean,
                            "rank_variance": cluster.rank_variance,
                            "mean_map_value": cluster.mean_map_value,
                            "variance_map_value": cluster.variance_map_value,
                            "local_contrast": cluster.local_contrast,
                            "rule_reliability": reliability,
                            "accepted": int(accepted),
                        }
                    )

    result = accumulator.finalize()
    manifest = {
        "schema_version": 1,
        "experiment": "TIRST-SAM Experiment 1 multi-view prompt screening",
        "generator": args.generator,
        "probe": probe_info,
        "split_role": args.split_role,
        "split_sha256": sha256_file(split_path),
        "seed": args.seed,
        "views": list(views),
        "shared_generator_weights": True,
        "gate": args.gate,
        "score_mode": args.score_mode,
        "candidate_k_raw": args.candidate_k_raw,
        "nms_radius": args.nms_radius,
        "score_threshold": args.score_threshold,
        "cluster_radius": args.cluster_radius,
        "rule": {
            "min_support": args.min_support,
            "max_dispersion": args.max_dispersion,
            "reliability_threshold": args.reliability_threshold,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        },
        "gt_boundary": "GT enters PromptMetricAccumulator only after multi-view proposals and gate decisions",
        "deterministic_eval": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "prompt_metrics_summary.json").write_text(
        json.dumps({"manifest": manifest, "metrics": result["summary"]}, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "prompt_metrics_per_image.csv", result["per_image_rows"])
    write_csv(output_dir / "prompt_metrics_per_component.csv", result["per_component_rows"])
    write_csv(output_dir / "candidate_budget_curve.csv", result["budget_rows"])
    write_csv(output_dir / "area_bin_metrics.csv", result["area_rows"])
    write_csv(output_dir / "candidate_cluster_metrics.csv", cluster_rows)
    print(json.dumps({"manifest": manifest, "metrics": result["summary"]}, ensure_ascii=False, indent=2, allow_nan=True))
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()

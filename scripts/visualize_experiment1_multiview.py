#!/usr/bin/env python3
"""Create balanced success/failure visualizations for Experiment 1 A3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.multiview_prompt import DEFAULT_VIEWS, multiview_propose, rule_reliability
from efficient_sam.prompt_metrics import candidate_component_options, extract_components
from scripts.train_experiment1_single_view import ImageOnlyProposalGenerator, make_sam_inputs, set_deterministic
from sirst_dataset import make_loader


class SharedProbeGenerator:
    def __init__(self, model, generator):
        self.model = model
        self.generator = generator

    def __call__(self, images: torch.Tensor):
        encoded = self.model.get_image_embeddings(images)
        return self.generator(images, encoded[0], encoded[2])


def normalize_display(array: np.ndarray) -> np.ndarray:
    lower, upper = float(array.min()), float(array.max())
    return (array - lower) / max(upper - lower, 1e-8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--mask_checkpoint", default=None)
    parser.add_argument("--weights", default="weights/efficient_sam_vitt.pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--min_support", type=int, default=2)
    parser.add_argument("--max_dispersion", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--reliability_threshold", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(20260825)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.probe_checkpoint, map_location="cpu", weights_only=False)
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        init_from_baseline=str(Path(args.weights).resolve()),
        use_adapter=False,
        return_encoder_multi_scale=True,
    ).to(device)
    mask_checkpoint = None
    if args.mask_checkpoint:
        mask_checkpoint = torch.load(args.mask_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(mask_checkpoint["model_state"], strict=True)
    model.eval()
    generator_args = SimpleNamespace(
        generator="probe",
        probe_checkpoint=args.probe_checkpoint,
        candidate_k_raw=32,
        nms_radius=3.0,
        score_threshold=0.1,
    )
    generator = ImageOnlyProposalGenerator(generator_args, device)
    shared_generator = SharedProbeGenerator(model, generator)
    views = tuple(item.strip() for item in args.views.split(",") if item.strip())
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    loader = make_loader(
        str(data_root),
        str(split_path),
        batch_size=1,
        size=256,
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

    desired = (
        "correct_accept",
        "correct_reject_clutter",
        "false_reject_tiny",
        "accepted_clutter",
        "multi_target",
        "single_view_correct_multiview_fail",
        "representative",
    )
    selected = {}
    records = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            gt = batch["mask"][0].numpy() > 0.5
            proposal, clusters_per_image, mean_map, variance_map = multiview_propose(
                images,
                shared_generator,
                views=views,
                cluster_radius=3.0,
                max_candidates=32,
                gate="rule",
                min_support=args.min_support,
                max_dispersion=args.max_dispersion,
                reliability_threshold=args.reliability_threshold,
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
            )
            predicted_mask = None
            if mask_checkpoint is not None:
                encoded = model.get_image_embeddings(images)
                points, labels, dense = make_sam_inputs(model, proposal, "points", 5)
                height, width = images.shape[-2:]
                mask_logits, _ = model.predict_masks(
                    encoded[0],
                    encoded[1],
                    points,
                    labels,
                    batched_masks=dense,
                    text_sparse_embeddings=None,
                    multimask_output=False,
                    input_h=height,
                    input_w=width,
                    output_h=height,
                    output_w=width,
                    hq_token_only=bool(
                        int(mask_checkpoint["epoch"])
                        <= int(mask_checkpoint["resolved_args"].get("hq_warmup_epochs", 30))
                    ),
                )
                threshold = float(
                    mask_checkpoint["resolved_args"].get("segmentation_threshold", 0.5)
                )
                predicted_mask = (
                    torch.sigmoid(mask_logits[0, 0, 0]) >= threshold
                ).detach().cpu()
            clusters = clusters_per_image[0]
            components = extract_components(gt, dilation_radius=2)
            cluster_info = []
            for cluster in clusters:
                q_value = rule_reliability(
                    cluster, alpha=args.alpha, beta=args.beta, gamma=args.gamma
                )
                positive = bool(candidate_component_options(cluster.center_xy, components, 3.0))
                accepted = (
                    cluster.support_count >= args.min_support
                    and cluster.center_dispersion <= args.max_dispersion
                    and q_value >= args.reliability_threshold
                )
                matched_tiny = False
                for _, component_index in candidate_component_options(cluster.center_xy, components, 3.0):
                    if components[component_index].area <= 9:
                        matched_tiny = True
                cluster_info.append((cluster, q_value, positive, accepted, matched_tiny))
            identity_hits = {
                component_index
                for cluster, _, positive, _, _ in cluster_info
                if positive and any(item.view_index == 0 for item in cluster.observations)
                for _, component_index in candidate_component_options(cluster.center_xy, components, 3.0)
            }
            accepted_hits = {
                component_index
                for cluster, _, positive, accepted, _ in cluster_info
                if positive and accepted
                for _, component_index in candidate_component_options(cluster.center_xy, components, 3.0)
            }
            categories = {"representative"}
            if any(positive and accepted for _, _, positive, accepted, _ in cluster_info):
                categories.add("correct_accept")
            if any((not positive) and (not accepted) for _, _, positive, accepted, _ in cluster_info):
                categories.add("correct_reject_clutter")
            if any(positive and (not accepted) and tiny for _, _, positive, accepted, tiny in cluster_info):
                categories.add("false_reject_tiny")
            if any((not positive) and accepted for _, _, positive, accepted, _ in cluster_info):
                categories.add("accepted_clutter")
            if len(components) >= 2:
                categories.add("multi_target")
            if identity_hits.difference(accepted_hits):
                categories.add("single_view_correct_multiview_fail")
            record = {
                "name": str(batch["name"][0]),
                "image": images[0, 0].detach().cpu(),
                "gt": torch.from_numpy(gt),
                "proposal": proposal,
                "clusters": cluster_info,
                "view_maps": proposal.auxiliary["view_maps"][:, 0, 0].detach().cpu(),
                "mean_map": mean_map[0, 0].detach().cpu(),
                "variance_map": variance_map[0, 0].detach().cpu(),
                "predicted_mask": predicted_mask,
            }
            records.append(record)
            for category in desired:
                if category in categories and category not in selected:
                    selected[category] = record
            if len(selected) >= min(args.max_images, len(desired)):
                break

    chosen = []
    seen_names = set()
    for category in desired:
        record = selected.get(category)
        if record is not None and record["name"] not in seen_names:
            chosen.append((category, record))
            seen_names.add(record["name"])
        if len(chosen) >= args.max_images:
            break
    for record in records:
        if len(chosen) >= args.max_images:
            break
        if record["name"] not in seen_names:
            chosen.append(("fallback", record))
            seen_names.add(record["name"])

    manifest_rows = []
    for category, record in chosen:
        image = normalize_display(record["image"].numpy())
        gt = record["gt"].numpy()
        figure, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
        axes = axes.ravel()
        axes[0].imshow(image, cmap="gray")
        axes[0].set_title(f"input: {record['name']}")
        for view_index, view in enumerate(views[:5], start=1):
            axes[view_index].imshow(record["view_maps"][view_index - 1], cmap="magma", vmin=0, vmax=1)
            axes[view_index].set_title(f"inverse {view}")
        axes[6].imshow(record["mean_map"], cmap="magma", vmin=0, vmax=1)
        axes[6].set_title("mean targetness")
        axes[7].imshow(record["variance_map"], cmap="viridis")
        axes[7].set_title("variance map")
        axes[8].imshow(image, cmap="gray")
        for cluster, _, positive, _, _ in record["clusters"]:
            axes[8].scatter(*cluster.center_xy, s=20, c="lime" if positive else "orange", marker="o")
        axes[8].set_title("raw clusters (GT-colored analysis)")
        axes[9].imshow(image, cmap="gray")
        for cluster, q_value, _, accepted, _ in record["clusters"]:
            axes[9].scatter(
                *cluster.center_xy,
                s=20 + 80 * q_value,
                c="lime" if accepted else "red",
                marker="o" if accepted else "x",
            )
        axes[9].set_title("accepted / rejected")
        axes[10].imshow(image, cmap="gray")
        axes[10].imshow(np.ma.masked_where(~gt, gt), cmap="winter", alpha=0.55)
        axes[10].set_title("GT overlay (analysis only)")
        lines = [f"category={category}"]
        for index, (cluster, q_value, positive, accepted, _) in enumerate(record["clusters"][:10]):
            lines.append(
                f"#{index + 1} q={q_value:.3f} support={cluster.support_count}/5 "
                f"disp={cluster.center_dispersion:.2f} gt={int(positive)} keep={int(accepted)}"
            )
        if record["predicted_mask"] is not None:
            axes[11].imshow(image, cmap="gray")
            predicted = record["predicted_mask"].numpy().astype(bool)
            axes[11].imshow(np.ma.masked_where(~predicted, predicted), cmap="autumn", alpha=0.55)
            axes[11].set_title("final A3 SAM mask")
        else:
            axes[11].axis("off")
            axes[11].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
        figure.suptitle(lines[0] + " | " + "; ".join(lines[1:4]), fontsize=9)
        for axis in axes:
            axis.set_axis_off()
        output_path = output_dir / f"{category}_{record['name']}.png"
        figure.savefig(output_path, dpi=160)
        plt.close(figure)
        manifest_rows.append(
            {
                "category": category,
                "name": record["name"],
                "file": output_path.name,
                "clusters": [
                    {
                        "x": cluster.center_xy[0],
                        "y": cluster.center_xy[1],
                        "q": q_value,
                        "support": cluster.support_count,
                        "dispersion": cluster.center_dispersion,
                        "gt_match": bool(positive),
                        "accepted": bool(accepted),
                    }
                    for cluster, q_value, positive, accepted, _ in record["clusters"]
                ],
            }
        )

    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import os
import time
import argparse
import json
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
try:
    from skimage import measure
except Exception:
    measure = None
# Cross-version AMP import (PyTorch>=2.0 uses torch.amp, older uses torch.cuda.amp)
try:
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScaler_new  # type: ignore
    def autocast_ctx(device: str):
        if device.startswith("cuda") and torch.cuda.is_available():
            return _autocast_new("cuda")
        return nullcontext()
    def make_scaler(device: str):
        if device == "cuda" and torch.cuda.is_available():
            return _GradScaler_new("cuda")
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def unscale_(self, optimizer):
                pass
            def update(self):
                pass
        return _DummyScaler()
except Exception:
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScaler_old  # type: ignore
    def autocast_ctx(device: str):
        if device.startswith("cuda") and torch.cuda.is_available():
            return _autocast_old()
        return nullcontext()
    def make_scaler(device: str):
        if torch.cuda.is_available():
            return _GradScaler_old()
        class _DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def unscale_(self, optimizer):
                pass
            def update(self):
                pass
        return _DummyScaler()


from sirst_dataset import make_loader
from efficient_sam.efficient_sam_hq import build_efficient_sam_hq
from efficient_sam.text_conditioner import (
    build_backbone_bifusion_block_adapter,
    build_bifusion_adapter_lite,
    build_gated_backbone_bifusion_block_adapter,
    build_targetness_aware_semantic_slot_generator,
    build_text_conditioner,
    build_text_dense_mask_prompt_generator,
    build_text_dense_mask_prompt_generator_v2,
    build_text_sparse_prompt_projector,
    cosine_distill_loss,
    masked_token_set_cosine_loss,
    targetness_aux_loss,
)
from efficient_sam.self_prompting_head import (
    boundary_aware_self_prompt_loss,
    boundary_aware_channel_sampled_point_loss,
    boundary_aware_expected_hit_loss,
    boundary_aware_sampled_point_loss,
    build_coarse_mask_prompt_head,
    build_dog_log_saliency,
    build_dynamic_sparse_prompt_head,
    build_mask_guided_self_prompting_head,
    build_multilevel_dynamic_sparse_prompt_head,
    build_self_prompting_head,
    coarse_mask_prompt_loss,
    dynamic_sparse_targetness_loss,
    dynamic_sparse_token_diversity_loss,
    foreground_component_peak_loss,
    self_prompt_heatmap_loss,
)
from efficient_sam.contrastive_prompt import build_contrastive_prompt_learning


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1.0) / (denom + 1.0)).mean()


def configure_self_prompt_sampler(head, variant: str) -> None:
    if head is None:
        return
    variant = str(variant or "legacy")
    head.fill_shortfall = False
    head.fill_pos_shortfall = None
    head.fill_neg_shortfall = None
    head.pos_boundary_ratio = None
    head.neg_boundary_ratio = None
    head.suppress_fg_from_inner = True
    head.positive_objectness_thr = 0.0
    head.pos_min_dist = None
    head.neg_min_dist = None
    head.neg_suppress_radius = None
    head.neg_fill_from_background = False
    head.component_aware_pos = False
    head.component_positive_mode = "full"
    head.component_score_mode = "foreground"
    head.component_rel_thr = 0.5
    head.component_abs_thr = float(getattr(head, "peak_thr", 0.05))
    head.component_min_area = 1

    base_min_dist = int(getattr(head, "_base_min_dist", getattr(head, "min_dist", 8)))
    head._base_min_dist = base_min_dist
    head.min_dist = base_min_dist

    if variant == "legacy":
        return
    if variant == "fill":
        head.fill_shortfall = True
        return
    if variant == "nofg_fill":
        head.fill_shortfall = True
        head.suppress_fg_from_inner = False
        return
    if variant == "nofg_fill_md3":
        head.fill_shortfall = True
        head.suppress_fg_from_inner = False
        head.min_dist = 3
        return
    if variant == "nofg_fill_md1":
        head.fill_shortfall = True
        head.suppress_fg_from_inner = False
        head.min_dist = 1
        return
    if variant in ("fg_bg", "fgbg", "foreground_background"):
        head.pos_boundary_ratio = 0.0
        head.neg_boundary_ratio = 0.0
        head.fill_pos_shortfall = True
        head.fill_neg_shortfall = True
        head.suppress_fg_from_inner = False
        head.pos_min_dist = 1
        head.neg_min_dist = base_min_dist
        head.neg_suppress_radius = max(1, base_min_dist // 2)
        head.neg_fill_from_background = True
        return
    if variant.startswith("fgcomp_bg"):
        spec = variant[len("fgcomp_bg"):]
        outer_ratio = 0.0
        if spec.startswith("_outer"):
            outer_spec, _, spec = spec[1:].partition("_")
            outer_suffix = outer_spec.replace("outer", "")
            outer_ratio = float(outer_suffix) / 100.0 if outer_suffix else 0.25
        head.pos_boundary_ratio = 0.0
        head.neg_boundary_ratio = max(0.0, min(1.0, outer_ratio))
        head.component_aware_pos = True
        head.component_positive_mode = "full"
        head.component_score_mode = "foreground"
        if "r03" in spec:
            head.component_rel_thr = 0.3
        elif "r07" in spec:
            head.component_rel_thr = 0.7
        else:
            head.component_rel_thr = 0.5
        head.component_abs_thr = float(getattr(head, "peak_thr", 0.05))
        head.component_min_area = 1
        head.fill_pos_shortfall = True
        head.fill_neg_shortfall = True
        head.suppress_fg_from_inner = False
        head.pos_min_dist = 1
        head.neg_min_dist = base_min_dist
        head.neg_suppress_radius = max(1, base_min_dist // 2)
        head.neg_fill_from_background = True
        return
    if variant.startswith("fg_bg_outer") or variant.startswith("fgbg_outer"):
        suffix = variant.replace("fg_bg_outer", "").replace("fgbg_outer", "")
        outer_ratio = 0.25
        if suffix:
            if suffix.startswith("p"):
                suffix = suffix[1:]
            outer_ratio = float(suffix) / 100.0
        head.pos_boundary_ratio = 0.0
        head.neg_boundary_ratio = max(0.0, min(1.0, outer_ratio))
        head.fill_pos_shortfall = True
        head.fill_neg_shortfall = True
        head.suppress_fg_from_inner = False
        head.pos_min_dist = 1
        head.neg_min_dist = base_min_dist
        head.neg_suppress_radius = max(1, base_min_dist // 2)
        head.neg_fill_from_background = True
        return
    if variant.startswith("posrel_negsafe_md"):
        suffix = variant.replace("posrel_negsafe_md", "")
        pos_min_dist = int(suffix)
        head.fill_pos_shortfall = True
        head.fill_neg_shortfall = False
        head.suppress_fg_from_inner = False
        head.pos_min_dist = max(1, pos_min_dist)
        head.neg_min_dist = base_min_dist
        head.neg_suppress_radius = max(1, base_min_dist // 2)
        head.neg_fill_from_background = True
        return
    if variant.startswith("comp_"):
        spec = variant[len("comp_"):]
        head.component_aware_pos = True
        head.component_positive_mode = "foreground_slots" if spec.startswith("slot") or "slot" in spec else "full"
        head.component_score_mode = "fg_inner" if "fginner" in spec else "foreground"
        if "r03" in spec:
            head.component_rel_thr = 0.3
        elif "r07" in spec:
            head.component_rel_thr = 0.7
        else:
            head.component_rel_thr = 0.5
        head.component_abs_thr = float(getattr(head, "peak_thr", 0.05))
        head.component_min_area = 1
        head.fill_pos_shortfall = True
        head.fill_neg_shortfall = False
        head.suppress_fg_from_inner = False
        head.pos_min_dist = 3 if "md3" in spec else 1
        head.neg_min_dist = base_min_dist
        head.neg_suppress_radius = max(1, base_min_dist // 2)
        head.neg_fill_from_background = True
        return
    raise ValueError(f"Unknown self_prompt_sampler_variant: {variant}")


def configure_self_prompt_guidance(head, args, prefix: str = "self_prompt") -> None:
    if head is None:
        return

    def get(name: str, base: str, default):
        value = getattr(args, name, None)
        if value is not None:
            return value
        return getattr(args, base, default)

    head.positive_guidance_alpha = float(get(
        f"{prefix}_guidance_alpha",
        "self_prompt_guidance_alpha",
        0.75,
    ))
    head.positive_guidance_bg_alpha = float(get(
        f"{prefix}_guidance_bg_alpha",
        "self_prompt_guidance_bg_alpha",
        0.5,
    ))
    head.positive_guidance_power = float(get(
        f"{prefix}_guidance_power",
        "self_prompt_guidance_power",
        1.0,
    ))


class NWDLoss(nn.Module):
    def __init__(self, constant: float = 12.0, eps: float = 1e-6):
        super().__init__()
        self.C = float(constant)
        self.eps = float(eps)

    def get_gaussian_params(self, mask, grid_x, grid_y):
        mass = mask.sum(dim=(2, 3), keepdim=True) + self.eps
        mu_x = (mask * grid_x).sum(dim=(2, 3), keepdim=True) / mass
        mu_y = (mask * grid_y).sum(dim=(2, 3), keepdim=True) / mass
        var_x = (mask * (grid_x - mu_x).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
        var_y = (mask * (grid_y - mu_y).pow(2)).sum(dim=(2, 3), keepdim=True) / mass
        sigma_x = torch.sqrt(var_x + self.eps)
        sigma_y = torch.sqrt(var_y + self.eps)
        return mu_x, mu_y, sigma_x, sigma_y

    def forward(self, preds, targets):
        probs = torch.sigmoid(preds)
        B, _, H, W = probs.shape
        device = probs.device
        y = torch.arange(H, device=device, dtype=probs.dtype) + 0.5
        x = torch.arange(W, device=device, dtype=probs.dtype) + 0.5
        try:
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        except TypeError:
            grid_y, grid_x = torch.meshgrid(y, x)
        valid_mask = (targets.sum(dim=(2, 3)) > 0).float().view(B)

        mu_x_p, mu_y_p, sig_x_p, sig_y_p = self.get_gaussian_params(probs, grid_x, grid_y)
        mu_x_t, mu_y_t, sig_x_t, sig_y_t = self.get_gaussian_params(targets, grid_x, grid_y)

        wd2 = (mu_x_p - mu_x_t).pow(2) + (mu_y_p - mu_y_t).pow(2) + \
              (sig_x_p - sig_x_t).pow(2) + (sig_y_p - sig_y_t).pow(2)
        wd2 = wd2.view(B)
        nwd = torch.exp(-torch.sqrt(wd2 + self.eps) / self.C)
        loss = 1.0 - nwd
        return (loss * valid_mask).sum() / (valid_mask.sum() + self.eps)


def radial_frequency_profile(mask: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Compute normalized radial energy profile for each mask."""
    if mask.dim() != 4:
        raise ValueError("mask tensor must have shape [B, C, H, W]")
    B, C, H, W = mask.shape
    mask = mask.float()
    spec = torch.fft.rfft2(mask, dim=(-2, -1), norm="forward")
    energy = spec.real.pow(2) + spec.imag.pow(2)

    fy = torch.fft.fftfreq(H, d=1.0, device=mask.device)
    fx = torch.fft.rfftfreq(W, d=1.0, device=mask.device)
    fy = fy.to(mask.dtype).view(1, 1, H, 1)
    fx = fx.to(mask.dtype).view(1, 1, 1, fx.numel())
    radius = torch.sqrt(fy.pow(2) + fx.pow(2))
    radius = radius / radius.max().clamp(min=1e-6)
    bin_idx = torch.clamp((radius * (num_bins - 1)).long(), max=num_bins - 1)

    energy_flat = energy.reshape(B, C, -1)
    idx_flat = bin_idx.reshape(1, 1, -1).expand_as(energy_flat)
    profile = torch.zeros(B, C, num_bins, device=mask.device, dtype=energy.dtype)
    profile.scatter_add_(2, idx_flat, energy_flat)

    counts = torch.zeros(num_bins, device=mask.device, dtype=energy.dtype)
    counts.scatter_add_(0, bin_idx.reshape(-1), torch.ones(bin_idx.numel(), device=mask.device, dtype=energy.dtype))
    counts = counts.clamp_min_(1.0)
    profile = profile / counts.view(1, 1, -1)
    profile = profile / (profile.sum(dim=-1, keepdim=True) + 1e-6)
    return profile


class PD_FA:
    def __init__(self, distance_thresh: int = 3):
        if measure is None:
            raise RuntimeError("scikit-image is required for PD/FA metrics; please install scikit-image.")
        self.distance_thresh = int(distance_thresh)
        self.reset()

    def update(self, preds, labels, size_hw):
        predits = np.array(preds.cpu()).astype("int64")
        labelss = np.array(labels.cpu()).astype("int64")

        image = measure.label(predits, connectivity=2)
        coord_image = list(measure.regionprops(image))
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.all_pixel += int(size_hw[0] * size_hw[1])

        matched = 0
        for region in coord_label:
            centroid_label = np.array(region.centroid)
            for idx in range(len(coord_image)):
                centroid_image = np.array(coord_image[idx].centroid)
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < self.distance_thresh:
                    matched += 1
                    del coord_image[idx]
                    break

        unmatched_areas = [r.area for r in coord_image]
        self.dismatch_pixel += int(np.sum(unmatched_areas)) if unmatched_areas else 0
        self.PD += matched

    def get(self):
        pd = self.PD / (self.target + 1e-6)
        fa = self.dismatch_pixel / (self.all_pixel + 1e-6)
        return float(pd), float(fa)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


def _boundary_map_from_mask(mask_2d_float: torch.Tensor) -> torch.Tensor:
    # mask_2d_float: [H,W] in {0,1}
    m = mask_2d_float.unsqueeze(0).unsqueeze(0)
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    erode = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    b = (dil - erode).clamp(min=0.0, max=1.0)
    return (b[0, 0] > 0).to(mask_2d_float.dtype)


def sample_points_from_mask(mask_bhw: torch.Tensor, n_pos=4, n_neg=4, boundary_prior: bool = False, boundary_ratio: float = 0.5):
    B, H, W = mask_bhw.shape
    device = mask_bhw.device
    pts, labels = [], []
    for b in range(B):
        pos_idx = (mask_bhw[b] > 0).nonzero(as_tuple=False)
        neg_idx = (mask_bhw[b] == 0).nonzero(as_tuple=False)
        if boundary_prior:
            bmap = _boundary_map_from_mask(mask_bhw[b].float())
            bpos = ((mask_bhw[b] > 0) & (bmap > 0)).nonzero(as_tuple=False)
            bneg = ((mask_bhw[b] == 0) & (bmap > 0)).nonzero(as_tuple=False)
            # how many from boundary
            bp = int(min(n_pos, len(pos_idx)) * boundary_ratio)
            bn = int(min(n_neg, len(neg_idx)) * boundary_ratio)
            sel_bpos = bpos[torch.randint(len(bpos), (bp,), device=bpos.device)] if bp > 0 and len(bpos) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            sel_bneg = bneg[torch.randint(len(bneg), (bn,), device=bneg.device)] if bn > 0 and len(bneg) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            rem_p = max(0, min(n_pos, len(pos_idx)) - sel_bpos.size(0))
            rem_n = max(0, min(n_neg, len(neg_idx)) - sel_bneg.size(0))
            sel_pos_rest = pos_idx[torch.randint(len(pos_idx), (rem_p,), device=pos_idx.device)] if rem_p > 0 and len(pos_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            sel_neg_rest = neg_idx[torch.randint(len(neg_idx), (rem_n,), device=neg_idx.device)] if rem_n > 0 and len(neg_idx) > 0 else torch.zeros((0, 2), dtype=torch.long, device=device)
            pos = torch.cat([sel_bpos, sel_pos_rest], dim=0)
            neg = torch.cat([sel_bneg, sel_neg_rest], dim=0)
        else:
            # original purely random sampling
            pass
        npos = min(n_pos, len(pos_idx)) if len(pos_idx) > 0 else 0
        nneg = min(n_neg, len(neg_idx)) if len(neg_idx) > 0 else 0
        if not boundary_prior:
            pos = (pos_idx[torch.randint(len(pos_idx), (npos,), device=pos_idx.device)] if npos > 0 else torch.zeros((0, 2), dtype=torch.long, device=device))
            neg = (neg_idx[torch.randint(len(neg_idx), (nneg,), device=neg_idx.device)] if nneg > 0 else torch.zeros((0, 2), dtype=torch.long, device=device))
        p = torch.cat([pos, neg], dim=0)
        l = torch.cat([torch.ones(npos), torch.zeros(nneg)], dim=0)
        if p.numel() == 0:
            if len(neg_idx) > 0:
                p = neg_idx[torch.randint(len(neg_idx), (n_neg,), device=neg_idx.device)]
                l = torch.zeros(p.shape[0], device=device)
            else:
                p = torch.zeros((1, 2), dtype=torch.long, device=device)
                l = torch.zeros(1, device=device)
        xy = torch.stack([p[:, 1], p[:, 0]], dim=-1).float()
        pts.append(xy[None, ...])
        labels.append(l[None, ...])
    max_pts = max(x.size(1) for x in pts)
    bpts, blbl = [], []
    for xy, l in zip(pts, labels):
        if xy.size(1) < max_pts:
            pad = max_pts - xy.size(1)
            xy = F.pad(xy, (0, 0, 0, pad), value=-1.0)
            l = F.pad(l, (0, pad), value=-1.0)
        bpts.append(xy)
        blbl.append(l)
    return torch.stack(bpts, 0), torch.stack(blbl, 0)


def make_empty_point_prompt(batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    pts = torch.zeros((batch_size, 1, 0, 2), dtype=torch.float32, device=device)
    labels = torch.zeros((batch_size, 1, 0), dtype=torch.float32, device=device)
    return pts, labels


def point_sample(input: torch.Tensor, point_coords: torch.Tensor, align_corners: bool = False) -> torch.Tensor:
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    out = F.grid_sample(input, 2.0 * point_coords - 1.0, align_corners=align_corners)
    if add_dim:
        out = out.squeeze(3)
    return out


def _calc_uncertainty(logits: torch.Tensor) -> torch.Tensor:
    # uncertainty = -|logit|
    return -torch.abs(logits)


def _get_uncertain_point_coords(coarse_logits: torch.Tensor, num_points: int, oversample_ratio: float = 3.0, importance_sample_ratio: float = 0.75) -> torch.Tensor:
    assert oversample_ratio >= 1.0 and 0.0 <= importance_sample_ratio <= 1.0
    N, C, H, W = coarse_logits.shape
    num_sampled = int(num_points * oversample_ratio)
    # random coords in [0,1]
    point_coords = torch.rand(N, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)  # [N,C,num_sampled]
    point_uncertainties = _calc_uncertainty(point_logits)  # [N,C,num_sampled]
    # for binary mask C=1
    topk = max(1, int(num_points * importance_sample_ratio))
    idx = torch.topk(point_uncertainties[:, 0, :], k=topk, dim=1)[1]
    shift = num_sampled * torch.arange(N, dtype=torch.long, device=coarse_logits.device)[:, None]
    idx = (idx + shift).view(-1)
    coords_topk = point_coords.view(-1, 2)[idx].view(N, topk, 2)
    num_random = num_points - topk
    if num_random > 0:
        rand_coords = torch.rand(N, num_random, 2, device=coarse_logits.device)
        coords = torch.cat([coords_topk, rand_coords], dim=1)
    else:
        coords = coords_topk
    return coords  # [N,P,2]


def compute_metrics(logits_b1hw, target_b1hw, thr=0.5):
    prob = torch.sigmoid(logits_b1hw)
    pred = (prob >= thr).float()
    target = target_b1hw.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    iou = torch.where(union > 0, inter / union, torch.ones_like(union))
    tp = inter
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    precision = torch.where((tp + fp) > 0, tp / (tp + fp), torch.ones_like(tp))
    recall = torch.where((tp + fn) > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    return iou.mean().item(), f1.mean().item()


def format_metric_tag(epoch: int, miou: float, niou: float, f1: float, pd: float, fa: float, thr: float = None):
    def fmt(name: str, value: float, scale: float = 1.0):
        scaled = value * scale
        return f"{name}{scaled:.2f}".replace(".", "p")
    parts = [
        f"ep{epoch:03d}",
        fmt("miou", miou, 100.0),
        fmt("niou", niou, 100.0),
        fmt("f1", f1, 100.0),
        fmt("pd", pd, 100.0),
        fmt("fa", fa, 1e6),
    ]
    if thr is not None:
        parts.append(fmt("thr", thr, 100.0))
    return "_".join(parts)


def log_line(message: str, log_path: str = None):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    print(message)


def _dedup_trainable_params(params):
    out = []
    seen = set()
    for p in params:
        if p is None or (not getattr(p, "requires_grad", False)):
            continue
        pid = id(p)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _exclude_params(params, exclude_params):
    exclude_ids = {id(p) for p in exclude_params}
    out = []
    seen = set()
    for p in params:
        if p is None or (not getattr(p, "requires_grad", False)):
            continue
        pid = id(p)
        if pid in exclude_ids or pid in seen:
            continue
        seen.add(pid)
        out.append(p)
    return out


def _make_optimizer(head_params, enc_params, args):
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": args.lr_head})
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr_encoder})
    if not param_groups:
        raise RuntimeError("No trainable parameters were provided to the optimizer.")
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def _select_topk_points(point_coords: torch.Tensor, point_labels: torch.Tensor, top_k: int):
    bsz, _, _ = point_coords.shape
    top_k = max(1, int(top_k))
    coords_out = torch.full((bsz, top_k, 2), -1.0, device=point_coords.device, dtype=point_coords.dtype)
    labels_out = torch.full((bsz, top_k), -1, device=point_labels.device, dtype=point_labels.dtype)
    for b in range(bsz):
        valid = point_labels[b] >= 0
        coords = point_coords[b][valid]
        if coords.numel() == 0:
            continue
        k = min(top_k, coords.shape[0])
        coords_out[b, :k] = coords[:k]
        labels_out[b, :k] = 1
    return coords_out, labels_out


def _run_pgap_with_text_prior(pgap, images, args, text_prior=None):
    if text_prior is not None and getattr(args, "pgap_text_fuse_internal", False):
        return pgap(
            images,
            text_prior=text_prior,
            text_fuse_weight=float(getattr(args, "pgap_text_fuse_weight", 0.5)),
            text_fuse_mode=str(getattr(args, "pgap_text_fuse_mode", "mul")),
        )
    return pgap(images)


def _build_pgap_prompts(pgap, images, masks, args, text_prior=None):
    pgap_pts, pgap_lbl, saliency = _run_pgap_with_text_prior(
        pgap, images, args, text_prior=text_prior
    )
    if getattr(args, "pgap_label_by_gt", False):
        pgap_pts, pgap_lbl = pgap.label_points_by_gt(
            pgap_pts,
            pgap_lbl,
            masks,
            saliency_map=saliency,
            min_pos=args.pgap_min_pos,
            max_neg=args.pgap_max_neg,
        )
    return pgap_pts, pgap_lbl, saliency


def _merge_dense_mask_prompts(base_prompt, text_prompt, alpha: float):
    if text_prompt is None:
        return base_prompt
    if base_prompt is None:
        return text_prompt
    a = max(0.0, min(1.0, float(alpha)))
    return (1.0 - a) * base_prompt + a * text_prompt


def _merge_sparse_prompt_embeddings(base_prompt, extra_prompt):
    if base_prompt is None:
        return extra_prompt
    if extra_prompt is None:
        return base_prompt
    extra_prompt = extra_prompt.to(device=base_prompt.device, dtype=base_prompt.dtype)
    return torch.cat([base_prompt, extra_prompt], dim=1)


def _build_self_prompt_sparse_tokens(
    args,
    contrastive_prompt,
    pts,
    lbl,
    image_size,
    training: bool,
):
    if contrastive_prompt is None:
        return None, None
    prompt_embed, contrastive_loss = contrastive_prompt(
        pts[:, 0].float(),
        lbl[:, 0].long(),
        image_size=tuple(image_size),
        return_loss=training,
    )
    if not bool(getattr(args, "self_prompt_inject_sparse_tokens", False)):
        prompt_embed = None
    return prompt_embed, contrastive_loss


def _select_text_sparse_prompt_source(
    args,
    raw_clip_feat,
    fused_clip_feat,
    fused_clip_token_feat=None,
    fused_clip_token_mask=None,
):
    source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
    if source == "raw_global":
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        return None, None
    if source == "fused_global":
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        return None, None
    if source == "fused_tokens":
        if fused_clip_token_feat is not None:
            return fused_clip_token_feat, fused_clip_token_mask
        if fused_clip_feat is not None:
            return fused_clip_feat, None
        if raw_clip_feat is not None:
            return raw_clip_feat, None
        return None, None
    raise ValueError(f"Unsupported text_sparse_prompt_source: {source}")


def _build_text_prompt_inputs(
    model,
    args,
    img_emb,
    clip_feat,
    raw_clip_feat=None,
    clip_token_feat=None,
    clip_token_mask=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
):
    if raw_clip_feat is None and clip_feat is None and clip_token_feat is None:
        return None, None
    sparse_prompt = None
    dense_prompt = None
    if text_sparse_prompt is not None:
        sparse_source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
        sparse_input, sparse_mask = _select_text_sparse_prompt_source(
            args,
            raw_clip_feat=raw_clip_feat,
            fused_clip_feat=clip_feat,
            fused_clip_token_feat=clip_token_feat,
            fused_clip_token_mask=clip_token_mask,
        )
        if sparse_input is not None:
            if sparse_input.dim() == 3:
                sparse_prompt = text_sparse_prompt(sparse_input, attention_mask=sparse_mask)
            else:
                sparse_prompt = text_sparse_prompt(
                    sparse_input,
                    use_global_prompt_enhance=(sparse_source == "raw_global"),
                )
    if text_dense_prompt is not None:
        target_size = getattr(model.prompt_encoder, "mask_input_size", None)
        if getattr(text_dense_prompt, "expects_token_level", False):
            dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
            if dense_text_input is not None:
                dense_prompt = text_dense_prompt(
                    img_emb,
                    dense_text_input,
                    attention_mask=clip_token_mask if clip_token_feat is not None else None,
                    output_size=tuple(target_size) if target_size is not None else None,
                )
        elif clip_feat is not None:
            dense_prompt = text_dense_prompt(
                img_emb,
                clip_feat,
                output_size=tuple(target_size) if target_size is not None else None,
            )
        if dense_prompt is not None:
            dense_prompt = dense_prompt * float(getattr(args, "text_dense_prompt_scale", 1.0))
    return sparse_prompt, dense_prompt


def _resolve_semantic_source(args, epoch: Optional[int] = None) -> str:
    source = str(getattr(args, "semantic_source", "teacher")).lower()
    if source not in ("teacher", "student", "none"):
        raise ValueError(f"Unsupported semantic_source={source!r}")
    start_epoch = int(getattr(args, "student_only_start_epoch", -1))
    if source == "teacher" and epoch is not None and start_epoch >= 0 and epoch >= start_epoch:
        source = "student"
    return source


def _any_tassg_loss_enabled(args) -> bool:
    return any(
        float(getattr(args, name, 0.0)) > 0.0
        for name in (
            "lambda_tassg_global",
            "lambda_tassg_token",
            "lambda_tassg_prompt",
            "lambda_tassg_targetness",
        )
    )


def _zero_scalar(device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _compute_tassg_distillation_losses(
    args,
    tassg_out,
    masks: torch.Tensor,
    teacher_global: Optional[torch.Tensor],
    teacher_tokens: Optional[torch.Tensor],
    teacher_mask: Optional[torch.Tensor],
    text_sparse_prompt=None,
):
    device = masks.device
    losses = {
        "global": _zero_scalar(device),
        "token": _zero_scalar(device),
        "prompt": _zero_scalar(device),
        "targetness": _zero_scalar(device),
    }
    if tassg_out is None:
        return losses

    student_global = tassg_out.get("global", None)
    student_tokens = tassg_out.get("tokens", None)
    student_mask = tassg_out.get("attn_mask", None)

    if teacher_global is not None and student_global is not None:
        losses["global"] = cosine_distill_loss(student_global, teacher_global)
    if teacher_tokens is not None and student_tokens is not None:
        losses["token"] = masked_token_set_cosine_loss(
            student_tokens,
            teacher_tokens,
            teacher_mask,
            bidirectional=True,
        )
    if text_sparse_prompt is not None:
        source = str(getattr(args, "text_sparse_prompt_source", "fused_tokens"))
        teacher_sparse = None
        student_sparse = None
        if source == "raw_global" and teacher_global is not None and student_global is not None:
            with torch.no_grad():
                teacher_sparse = text_sparse_prompt(
                    teacher_global,
                    use_global_prompt_enhance=True,
                )
            student_sparse = text_sparse_prompt(
                student_global,
                use_global_prompt_enhance=True,
            )
        elif source == "fused_global" and teacher_global is not None and student_global is not None:
            with torch.no_grad():
                teacher_sparse = text_sparse_prompt(teacher_global)
            student_sparse = text_sparse_prompt(student_global)
        elif teacher_tokens is not None and student_tokens is not None:
            with torch.no_grad():
                teacher_sparse = text_sparse_prompt(
                    teacher_tokens,
                    attention_mask=teacher_mask,
                )
            student_sparse = text_sparse_prompt(
                student_tokens,
                attention_mask=student_mask,
            )
        if teacher_sparse is not None and student_sparse is not None:
            losses["prompt"] = F.mse_loss(student_sparse.float(), teacher_sparse.float())
    if "targetness_logits" in tassg_out:
        losses["targetness"] = targetness_aux_loss(
            tassg_out["targetness_logits"],
            masks,
        )
    return losses


def _activate_tassg_semantics(args, tassg, img_emb, epoch: Optional[int] = None):
    semantic_source = _resolve_semantic_source(args, epoch=epoch)
    if semantic_source == "student" and tassg is None:
        raise RuntimeError("--semantic_source student requires --use_tassg.")
    if tassg is None:
        return semantic_source, None, None, None, None
    tassg_out = tassg(img_emb)
    if semantic_source != "student":
        return semantic_source, tassg_out, None, None, None
    return (
        semantic_source,
        tassg_out,
        tassg_out["global"],
        tassg_out["tokens"],
        tassg_out["attn_mask"],
    )


def _build_bifusion_text_inputs(
    clip_feat: Optional[torch.Tensor],
    clip_token_feat: Optional[torch.Tensor],
    clip_token_mask: Optional[torch.Tensor],
):
    if clip_token_feat is not None:
        if clip_token_mask is None:
            clip_token_mask = torch.ones(
                (clip_token_feat.shape[0], clip_token_feat.shape[1]),
                device=clip_token_feat.device,
                dtype=torch.long,
            )
        return clip_token_feat, clip_token_mask
    if clip_feat is not None:
        return clip_feat.unsqueeze(1), torch.ones(
            (clip_feat.shape[0], 1),
            device=clip_feat.device,
            dtype=torch.long,
        )
    return None, None


def _masked_text_mean(text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if text_mask is None:
        return text_tokens.mean(dim=1)
    mask = (text_mask > 0).to(text_tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (text_tokens * mask).sum(dim=1) / denom


def _parse_int_list_arg(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        vals = list(default or [])
    return sorted({max(0, int(v)) for v in vals})


def _parse_int_list_raw_arg(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
    else:
        vals = [int(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        vals = list(default or [])
    return [int(v) for v in vals]


def _parse_float_list_arg(value, default=None):
    if value is None:
        return list(default or [])
    if isinstance(value, (list, tuple)):
        vals = [float(v) for v in value]
    else:
        vals = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    if not vals:
        vals = list(default or [])
    return [float(v) for v in vals]


def _unpack_image_embeddings_output(output, use_detail_outputs: bool = False):
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("model.get_image_embeddings() must return at least (img_emb, interms).")
    if use_detail_outputs:
        detail_ms_embeddings = output[2] if len(output) >= 3 else None
        return output[0], output[1], detail_ms_embeddings
    return output[0], output[1], None


def _extract_encoder_ms_embeddings(output, use_detail_outputs: bool = False):
    if not isinstance(output, (tuple, list)):
        return None
    if use_detail_outputs:
        return output[3] if len(output) >= 4 else None
    return output[2] if len(output) >= 3 else None


def _unpack_image_embeddings_with_text_output(output, use_detail_outputs: bool = False):
    if not isinstance(output, (tuple, list)) or len(output) < 4:
        raise ValueError(
            "model.get_image_embeddings_with_text() must return at least (img_emb, interms, text_tokens, text_mask)."
        )
    if use_detail_outputs:
        detail_ms_embeddings = output[2] if len(output) >= 5 else None
        if len(output) >= 5:
            return output[0], output[1], detail_ms_embeddings, output[3], output[4]
        return output[0], output[1], detail_ms_embeddings, output[2], output[3]
    return output[0], output[1], None, output[-2], output[-1]


def _uses_dsb_amgd(args_or_model) -> bool:
    if args_or_model is None:
        return False
    if hasattr(args_or_model, "use_detail_branch_amgd"):
        return bool(getattr(args_or_model, "use_detail_branch_amgd", False))
    return bool(
        getattr(args_or_model, "use_amgd", False)
        and str(getattr(args_or_model, "amgd_branch_design", "legacy")).lower() == "dsb_v1"
    )


def _detail_branch_embeddings_arg(args_or_model, detail_ms_embeddings):
    return detail_ms_embeddings if _uses_dsb_amgd(args_or_model) else None


def _apply_backbone_bifusion_adapter(
    model,
    backbone_bifusion_adapter,
    images,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
):
    use_detail_outputs = bool(getattr(model, "use_hldf", False) or _uses_dsb_amgd(model))
    if backbone_bifusion_adapter is None:
        img_emb, interms, detail_ms_embeddings = _unpack_image_embeddings_output(
            model.get_image_embeddings(images),
            use_detail_outputs=use_detail_outputs,
        )
        return img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask
    text_tokens, text_mask = _build_bifusion_text_inputs(
        clip_feat=clip_feat,
        clip_token_feat=clip_token_feat,
        clip_token_mask=clip_token_mask,
    )
    if text_tokens is None or not hasattr(model, "get_image_embeddings_with_text"):
        img_emb, interms, detail_ms_embeddings = _unpack_image_embeddings_output(
            model.get_image_embeddings(images),
            use_detail_outputs=use_detail_outputs,
        )
        return img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask
    img_emb, interms, detail_ms_embeddings, text_tokens_out, text_mask_out = _unpack_image_embeddings_with_text_output(
        model.get_image_embeddings_with_text(
            images,
            text_tokens,
            text_attention_mask=text_mask,
        ),
        use_detail_outputs=use_detail_outputs,
    )
    text_global_out = _masked_text_mean(text_tokens_out, text_mask_out)
    return img_emb, interms, detail_ms_embeddings, text_global_out, text_tokens_out, text_mask_out


def _apply_bifusion_adapter(
    bifusion_adapter,
    img_emb,
    interms,
    detail_ms_embeddings,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
):
    if bifusion_adapter is None:
        return img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask
    text_tokens, text_mask = _build_bifusion_text_inputs(
        clip_feat=clip_feat,
        clip_token_feat=clip_token_feat,
        clip_token_mask=clip_token_mask,
    )
    if text_tokens is None:
        return img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask

    img_emb, interms, text_tokens_out, text_mask_out, text_global_out = bifusion_adapter(
        img_emb,
        interms,
        text_tokens,
        attention_mask=text_mask,
    )
    return img_emb, interms, detail_ms_embeddings, text_global_out, text_tokens_out, text_mask_out


def _build_pgap_text_prior(
    model,
    args,
    img_emb,
    clip_feat,
    clip_token_feat=None,
    clip_token_mask=None,
    text_dense_prompt=None,
    output_size=None,
):
    if not getattr(args, "pgap_text_fuse_internal", False):
        return None
    if text_dense_prompt is None:
        return None
    if getattr(text_dense_prompt, "expects_token_level", False):
        dense_text_input = clip_token_feat if clip_token_feat is not None else clip_feat
        if dense_text_input is None:
            return None
        prior = text_dense_prompt(
            img_emb,
            dense_text_input,
            attention_mask=clip_token_mask if clip_token_feat is not None else None,
            output_size=tuple(output_size) if output_size is not None else None,
        )
        if prior is not None:
            prior = prior * float(getattr(args, "text_dense_prompt_scale", 1.0))
        return prior
    if clip_feat is None:
        return None
    prior = text_dense_prompt(
        img_emb,
        clip_feat,
        output_size=tuple(output_size) if output_size is not None else None,
    )
    if prior is not None:
        prior = prior * float(getattr(args, "text_dense_prompt_scale", 1.0))
    return prior


def build_center_gaussian_targets(masks: torch.Tensor, sigma: float) -> torch.Tensor:
    if masks.dim() == 4:
        masks = masks[:, 0]
    if masks.dim() != 3:
        raise ValueError("build_center_gaussian_targets expects masks with shape [B,H,W] or [B,1,H,W].")

    masks_bin = masks > 0.5
    bsz, height, width = masks_bin.shape
    device = masks.device
    sigma = max(float(sigma), 1e-3)
    ys = torch.arange(height, device=device, dtype=torch.float32).view(height, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, width)
    targets = torch.zeros((bsz, 1, height, width), device=device, dtype=torch.float32)
    masks_np = masks_bin.detach().cpu().numpy() if measure is not None else None

    for b_idx in range(bsz):
        centroids = []
        if measure is not None:
            labeled = measure.label(masks_np[b_idx].astype(np.uint8), connectivity=2)
            props = measure.regionprops(labeled)
            for prop in props:
                cy, cx = prop.centroid
                centroids.append((float(cy), float(cx)))
        if not centroids:
            coords = masks_bin[b_idx].nonzero(as_tuple=False).float()
            if coords.numel() > 0:
                centroids.append((float(coords[:, 0].mean().item()), float(coords[:, 1].mean().item())))
        for cy, cx in centroids:
            gaussian = torch.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2.0 * sigma * sigma))
            targets[b_idx, 0] = torch.maximum(targets[b_idx, 0], gaussian)
    return targets


def _arg_with_fallback(args, name: str, fallback: str, default):
    value = getattr(args, name, None)
    if value is not None:
        return value
    return getattr(args, fallback, default)


def compute_boundary_prompt_aux_losses(
    args,
    prompt_logits,
    point_coords,
    point_labels,
    point_types,
    masks,
    prefix: str = "self_prompt",
):
    if not isinstance(prompt_logits, dict):
        heat = self_prompt_heatmap_loss(
            prompt_logits,
            masks,
            pos_weight=float(_arg_with_fallback(args, f"{prefix}_pos_weight", "self_prompt_pos_weight", 10.0)),
        )
        return heat, None, None, None

    heat = boundary_aware_self_prompt_loss(
        prompt_logits,
        masks,
        pos_weight=float(_arg_with_fallback(args, f"{prefix}_pos_weight", "self_prompt_pos_weight", 10.0)),
        boundary_pos_weight=float(_arg_with_fallback(args, f"{prefix}_boundary_pos_weight", "self_prompt_boundary_pos_weight", 10.0)),
        background_pos_weight=float(_arg_with_fallback(args, f"{prefix}_background_pos_weight", "self_prompt_background_pos_weight", 1.0)),
        boundary_width=int(_arg_with_fallback(args, f"{prefix}_boundary_width", "self_prompt_boundary_width", 1)),
        background_margin=int(_arg_with_fallback(args, f"{prefix}_background_margin", "self_prompt_background_margin", 3)),
        background_loss_weight=float(_arg_with_fallback(args, f"{prefix}_background_loss_weight", "self_prompt_background_loss_weight", 0.25)),
        target_mode=str(_arg_with_fallback(args, f"{prefix}_target_mode", "self_prompt_target_mode", "soft")),
        foreground_mode=str(_arg_with_fallback(args, f"{prefix}_foreground_mode", "self_prompt_foreground_mode", "mask")),
        soft_band_radius=int(_arg_with_fallback(args, f"{prefix}_soft_band_radius", "self_prompt_soft_band_radius", 3)),
        soft_sigma=float(_arg_with_fallback(args, f"{prefix}_soft_sigma", "self_prompt_soft_sigma", 1.5)),
        background_fade_radius=int(_arg_with_fallback(args, f"{prefix}_background_fade_radius", "self_prompt_background_fade_radius", 4)),
        active_channels=str(_arg_with_fallback(args, f"{prefix}_loss_channels", "self_prompt_loss_channels", "all")),
    )

    sampled = None
    if float(_arg_with_fallback(args, f"{prefix}_sampled_point_weight", "self_prompt_sampled_point_weight", 0.0)) > 0.0:
        sampled_loss_mode = str(_arg_with_fallback(args, f"{prefix}_sampled_loss_mode", "self_prompt_sampled_loss_mode", "legacy")).lower()
        if sampled_loss_mode == "channel_error" and point_types is not None:
            sampled = boundary_aware_channel_sampled_point_loss(
                prompt_logits,
                point_coords,
                point_labels,
                point_types,
                masks,
                pos_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_pos_weight", "self_prompt_sampled_pos_weight", 1.0)),
                neg_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_neg_weight", "self_prompt_sampled_neg_weight", 1.0)),
                boundary_width=int(_arg_with_fallback(args, f"{prefix}_boundary_width", "self_prompt_boundary_width", 1)),
                background_margin=int(_arg_with_fallback(args, f"{prefix}_background_margin", "self_prompt_background_margin", 3)),
                target_mode=str(_arg_with_fallback(args, f"{prefix}_target_mode", "self_prompt_target_mode", "hard")),
                foreground_mode=str(_arg_with_fallback(args, f"{prefix}_foreground_mode", "self_prompt_foreground_mode", "mask")),
                soft_band_radius=int(_arg_with_fallback(args, f"{prefix}_soft_band_radius", "self_prompt_soft_band_radius", 3)),
                soft_sigma=float(_arg_with_fallback(args, f"{prefix}_soft_sigma", "self_prompt_soft_sigma", 1.5)),
                background_fade_radius=int(_arg_with_fallback(args, f"{prefix}_background_fade_radius", "self_prompt_background_fade_radius", 4)),
                error_only=True,
                target_threshold=float(_arg_with_fallback(args, f"{prefix}_channel_error_target_threshold", "self_prompt_channel_error_target_threshold", 0.5)),
            )
        else:
            sampled = boundary_aware_sampled_point_loss(
                prompt_logits,
                point_coords,
                point_labels,
                masks,
                pos_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_pos_weight", "self_prompt_sampled_pos_weight", 1.0)),
                neg_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_neg_weight", "self_prompt_sampled_neg_weight", 1.0)),
            )

    expected = None
    if float(_arg_with_fallback(args, f"{prefix}_expected_hit_weight", "self_prompt_expected_hit_weight", 0.0)) > 0.0:
        expected = boundary_aware_expected_hit_loss(
            prompt_logits,
            masks,
            pos_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_pos_weight", "self_prompt_sampled_pos_weight", 1.0)),
            neg_weight=float(_arg_with_fallback(args, f"{prefix}_sampled_neg_weight", "self_prompt_sampled_neg_weight", 1.0)),
        )

    component = None
    if float(_arg_with_fallback(args, f"{prefix}_component_peak_weight", "self_prompt_component_peak_weight", 0.0)) > 0.0:
        component = foreground_component_peak_loss(
            prompt_logits,
            masks,
            temperature=float(_arg_with_fallback(args, f"{prefix}_component_peak_temperature", "self_prompt_component_peak_temperature", 0.5)),
            min_area=int(_arg_with_fallback(args, f"{prefix}_component_peak_min_area", "self_prompt_component_peak_min_area", 1)),
            max_components=int(_arg_with_fallback(args, f"{prefix}_component_peak_max_components", "self_prompt_component_peak_max_components", 256)),
        )
    return heat, sampled, expected, component


def build_self_prompt_positive_guidance(args, images: torch.Tensor, output_size, prefix: str = "self_prompt"):
    mode = str(_arg_with_fallback(args, f"{prefix}_positive_guidance", "self_prompt_positive_guidance", "none")).lower()
    if mode in ("", "none", "off", "false", "0"):
        return None
    if mode not in ("dog_log", "doglog", "dog-log"):
        raise ValueError(f"Unsupported {prefix}_positive_guidance={mode!r}")
    guidance = build_dog_log_saliency(
        images,
        dog_sigmas=str(getattr(args, "dog_log_dog_sigmas", "0.7-1.4,1.0-2.0,1.5-3.0,2.0-4.0")),
        log_sigmas=str(getattr(args, "dog_log_log_sigmas", "0.8,1.2,1.6,2.4")),
        truncate=float(getattr(args, "dog_log_truncate", 3.0)),
    )
    if output_size is not None and tuple(guidance.shape[-2:]) != tuple(output_size):
        guidance = F.interpolate(guidance, size=tuple(output_size), mode="bilinear", align_corners=False)
    return guidance


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch,
    args,
    pgap=None,
    fab_criterion=None,
    scr_criterion=None,
    text_conditioner=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
    tassg=None,
    bifusion_adapter=None,
    backbone_bifusion_adapter=None,
    self_prompt_head=None,
    refine_self_prompt_head=None,
    coarse_mask_head=None,
    dynamic_sparse_prompt_head=None,
    contrastive_prompt=None,
    lca_prompt=None,
):
    model.train()
    if pgap is not None:
        pgap.train()
    if self_prompt_head is not None:
        self_prompt_head.train()
    if refine_self_prompt_head is not None:
        refine_self_prompt_head.train()
    if coarse_mask_head is not None:
        coarse_mask_head.train()
    if dynamic_sparse_prompt_head is not None:
        dynamic_sparse_prompt_head.train()
    if contrastive_prompt is not None:
        contrastive_prompt.train()
    if tassg is not None:
        tassg.train()
    if bool(getattr(args, "dynamic_sparse_train_head_only", False)):
        model.eval()
        if pgap is not None:
            pgap.eval()
        if self_prompt_head is not None:
            self_prompt_head.eval()
        if refine_self_prompt_head is not None:
            refine_self_prompt_head.eval()
        if coarse_mask_head is not None:
            coarse_mask_head.eval()
        if contrastive_prompt is not None:
            contrastive_prompt.eval()
        if tassg is not None:
            tassg.eval()
        if dynamic_sparse_prompt_head is not None:
            dynamic_sparse_prompt_head.train()
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    bce = nn.BCEWithLogitsLoss()
    nwd_weight = float(getattr(args, "nwd_weight", 0.0))
    nwd_criterion = NWDLoss(constant=float(getattr(args, "nwd_constant", 12.0))).to(device) if nwd_weight > 0.0 else None
    meter_loss, meter_center_loss, meter_center_contain, n = 0.0, 0.0, 0.0, 0
    meter_sp_sampled, meter_sp_expected, meter_sp_component, meter_coarse_mask, meter_dyn_sparse = 0.0, 0.0, 0.0, 0.0, 0.0
    meter_stage2_sp, meter_stage2_sampled, meter_stage2_expected, meter_stage2_component = 0.0, 0.0, 0.0, 0.0
    meter_stage1_mask = 0.0
    meter_dyn_target = 0.0
    meter_tassg_global = 0.0
    meter_tassg_token = 0.0
    meter_tassg_prompt = 0.0
    meter_tassg_targetness = 0.0
    skipped_nonfinite = 0
    grad_accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
    accum_count = 0
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape

        encoder_ms_embeddings = None
        semantic_source = _resolve_semantic_source(args, epoch=epoch)
        teacher_global = None
        teacher_tokens = None
        teacher_mask = None
        tassg_out = None
        needs_teacher_features = (
            semantic_source == "teacher"
            or (tassg is not None and _any_tassg_loss_enabled(args))
        )
        with autocast_ctx(device):
            clip_feat = None
            raw_clip_feat = None
            clip_token_feat = None
            clip_token_mask = None
            if needs_teacher_features and "clip_text_feat" in batch:
                teacher_global = batch["clip_text_feat"].to(device, non_blocking=True)
            if needs_teacher_features and "clip_text_token_feat" in batch:
                teacher_tokens = batch["clip_text_token_feat"].to(device, non_blocking=True)
                if "clip_text_attn_mask" in batch:
                    teacher_mask = batch["clip_text_attn_mask"].to(device, non_blocking=True)
            if semantic_source == "teacher":
                clip_feat = teacher_global
                raw_clip_feat = teacher_global
                clip_token_feat = teacher_tokens
                clip_token_mask = teacher_mask
            elif semantic_source == "student" and tassg is None:
                raise RuntimeError("--semantic_source student requires --use_tassg.")

            use_teacher_backbone = (
                semantic_source == "teacher"
                and (not pgap_text_prior_only)
                and backbone_bifusion_adapter is not None
            )
            if use_teacher_backbone:
                img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                    model=model,
                    backbone_bifusion_adapter=backbone_bifusion_adapter,
                    images=images,
                    clip_feat=clip_feat,
                    clip_token_feat=clip_token_feat,
                    clip_token_mask=clip_token_mask,
                )
            else:
                use_detail_outputs = bool(getattr(args, "use_hldf", False) or _uses_dsb_amgd(args))
                image_embedding_output = model.get_image_embeddings(images)
                img_emb, interms, detail_ms_embeddings = _unpack_image_embeddings_output(
                    image_embedding_output,
                    use_detail_outputs=use_detail_outputs,
                )
                encoder_ms_embeddings = _extract_encoder_ms_embeddings(
                    image_embedding_output,
                    use_detail_outputs=use_detail_outputs,
                )

        if tassg is not None and (semantic_source == "student" or _any_tassg_loss_enabled(args)):
            tassg_out = tassg(img_emb)
            if semantic_source == "student":
                clip_feat = tassg_out["global"]
                raw_clip_feat = clip_feat
                clip_token_feat = tassg_out["tokens"]
                clip_token_mask = tassg_out["attn_mask"]
                if (
                    bool(getattr(args, "tassg_two_pass_backbone", False))
                    and (not pgap_text_prior_only)
                    and backbone_bifusion_adapter is not None
                ):
                    with autocast_ctx(device):
                        img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                            model=model,
                            backbone_bifusion_adapter=backbone_bifusion_adapter,
                            images=images,
                            clip_feat=tassg_out["global"],
                            clip_token_feat=tassg_out["tokens"],
                            clip_token_mask=tassg_out["attn_mask"],
                        )
                    raw_clip_feat = tassg_out["global"]

        if (not pgap_text_prior_only) and semantic_source != "none" and bifusion_adapter is not None:
            img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_bifusion_adapter(
                bifusion_adapter=bifusion_adapter,
                img_emb=img_emb,
                interms=interms,
                detail_ms_embeddings=detail_ms_embeddings,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        if (not pgap_text_prior_only) and semantic_source != "none" and text_conditioner is not None and clip_feat is not None:
            img_emb = text_conditioner(img_emb, clip_feat)
        pgap_text_prior = None
        if pgap is not None:
            pgap_text_prior = _build_pgap_text_prior(
                model,
                args,
                img_emb,
                clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_dense_prompt=text_dense_prompt,
                output_size=(H, W),
            )
        mask_prompt = None
        gt_pts = None
        gt_lbl = None
        self_prompt_heat_loss = None
        self_prompt_sampled_loss = None
        self_prompt_expected_loss = None
        self_prompt_component_loss = None
        stage2_self_prompt_heat_loss = None
        stage2_self_prompt_sampled_loss = None
        stage2_self_prompt_expected_loss = None
        stage2_self_prompt_component_loss = None
        stage1_mask_loss_val = None
        coarse_mask_loss_val = None
        coarse_mask_prompt = None
        self_prompt_positive_guidance = None
        stage2_positive_guidance = None
        if pgap is not None:
            pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(
                pgap, images, masks, args, text_prior=pgap_text_prior
            )
            if args.use_feature_mod:
                img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pts = pgap_pts.unsqueeze(1)
                lbl = pgap_lbl.unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
            else:
                if str(getattr(args, "prompt_mode", "gt_points")) == "assp_only":
                    pts, lbl = make_empty_point_prompt(int(masks.shape[0]), device)
                else:
                    gt_pts, gt_lbl = sample_points_from_mask(
                        masks,
                        n_pos=args.n_pos,
                        n_neg=args.n_neg,
                        boundary_prior=bool(args.boundary_prior_sampling),
                        boundary_ratio=float(args.boundary_ratio),
                    )
                    pts, lbl = gt_pts.to(device), gt_lbl.to(device)
        else:
            if str(getattr(args, "prompt_mode", "gt_points")) == "assp_only":
                gt_pts, gt_lbl = None, None
                pts, lbl = make_empty_point_prompt(int(masks.shape[0]), device)
            else:
                gt_pts, gt_lbl = sample_points_from_mask(
                    masks,
                    n_pos=args.n_pos,
                    n_neg=args.n_neg,
                    boundary_prior=bool(args.boundary_prior_sampling),
                    boundary_ratio=float(args.boundary_ratio),
                )
                gt_pts, gt_lbl = gt_pts.to(device), gt_lbl.to(device)
                pts, lbl = gt_pts, gt_lbl
            if self_prompt_head is not None:
                self_prompt_positive_guidance = build_self_prompt_positive_guidance(
                    args,
                    images,
                    (H, W),
                    prefix="self_prompt",
                )
                _, sp_pts, sp_lbl, sp_logits = self_prompt_head(
                    img_emb,
                    output_size=(H, W),
                    gt_mask=masks,
                    positive_guidance=self_prompt_positive_guidance,
                )
                if isinstance(sp_logits, dict):
                    self_prompt_heat_loss = boundary_aware_self_prompt_loss(
                        sp_logits,
                        masks,
                        pos_weight=float(getattr(args, "self_prompt_pos_weight", 10.0)),
                        boundary_pos_weight=float(getattr(args, "self_prompt_boundary_pos_weight", getattr(args, "self_prompt_pos_weight", 10.0))),
                        background_pos_weight=float(getattr(args, "self_prompt_background_pos_weight", 1.0)),
                        boundary_width=int(getattr(args, "self_prompt_boundary_width", 1)),
                        background_margin=int(getattr(args, "self_prompt_background_margin", 3)),
                        background_loss_weight=float(getattr(args, "self_prompt_background_loss_weight", 0.25)),
                        target_mode=str(getattr(args, "self_prompt_target_mode", "soft")),
                        foreground_mode=str(getattr(args, "self_prompt_foreground_mode", "mask")),
                        soft_band_radius=int(getattr(args, "self_prompt_soft_band_radius", 3)),
                        soft_sigma=float(getattr(args, "self_prompt_soft_sigma", 1.5)),
                        background_fade_radius=int(getattr(args, "self_prompt_background_fade_radius", 4)),
                        active_channels=str(getattr(args, "self_prompt_loss_channels", "all")),
                    )
                    component_peak_weight = float(getattr(args, "self_prompt_component_peak_weight", 0.0))
                    if component_peak_weight > 0.0:
                        self_prompt_component_loss = foreground_component_peak_loss(
                            sp_logits,
                            masks,
                            temperature=float(getattr(args, "self_prompt_component_peak_temperature", 0.5)),
                            min_area=int(getattr(args, "self_prompt_component_peak_min_area", 1)),
                            max_components=int(getattr(args, "self_prompt_component_peak_max_components", 256)),
                        )
                    sampled_weight = float(getattr(args, "self_prompt_sampled_point_weight", 0.0))
                    expected_weight = float(getattr(args, "self_prompt_expected_hit_weight", 0.0))
                    if sampled_weight > 0.0:
                        sampled_loss_mode = str(getattr(args, "self_prompt_sampled_loss_mode", "legacy")).lower()
                        sp_types = getattr(self_prompt_head, "last_point_types", None)
                        if sampled_loss_mode == "channel_error" and sp_types is not None:
                            self_prompt_sampled_loss = boundary_aware_channel_sampled_point_loss(
                                sp_logits,
                                sp_pts,
                                sp_lbl,
                                sp_types,
                                masks,
                                pos_weight=float(getattr(args, "self_prompt_sampled_pos_weight", 1.0)),
                                neg_weight=float(getattr(args, "self_prompt_sampled_neg_weight", 1.0)),
                                boundary_width=int(getattr(args, "self_prompt_boundary_width", 1)),
                                background_margin=int(getattr(args, "self_prompt_background_margin", 3)),
                                target_mode=str(getattr(args, "self_prompt_target_mode", "hard")),
                                foreground_mode=str(getattr(args, "self_prompt_foreground_mode", "mask")),
                                soft_band_radius=int(getattr(args, "self_prompt_soft_band_radius", 3)),
                                soft_sigma=float(getattr(args, "self_prompt_soft_sigma", 1.5)),
                                background_fade_radius=int(getattr(args, "self_prompt_background_fade_radius", 4)),
                                error_only=True,
                                target_threshold=float(getattr(args, "self_prompt_channel_error_target_threshold", 0.5)),
                            )
                        else:
                            self_prompt_sampled_loss = boundary_aware_sampled_point_loss(
                                sp_logits,
                                sp_pts,
                                sp_lbl,
                                masks,
                                pos_weight=float(getattr(args, "self_prompt_sampled_pos_weight", 1.0)),
                                neg_weight=float(getattr(args, "self_prompt_sampled_neg_weight", 1.0)),
                            )
                    if expected_weight > 0.0:
                        self_prompt_expected_loss = boundary_aware_expected_hit_loss(
                            sp_logits,
                            masks,
                            pos_weight=float(getattr(args, "self_prompt_sampled_pos_weight", 1.0)),
                            neg_weight=float(getattr(args, "self_prompt_sampled_neg_weight", 1.0)),
                        )
                else:
                    self_prompt_heat_loss = self_prompt_heatmap_loss(
                        sp_logits,
                        masks,
                        pos_weight=float(getattr(args, "self_prompt_pos_weight", 10.0)),
                    )
                mix_ratio = max(0.0, min(1.0, float(getattr(args, "self_prompt_mix_ratio", 0.5))))
                if getattr(args, "self_prompt_mix_schedule", False):
                    start_epoch = max(1, int(getattr(args, "self_prompt_mix_start_epoch", 1)))
                    warmup = max(1, int(getattr(args, "self_prompt_warmup", 30)))
                    if epoch < start_epoch:
                        mix_ratio = 0.0
                    else:
                        ramp_epoch = epoch - start_epoch + 1
                        mix_ratio = mix_ratio * min(1.0, float(ramp_epoch) / float(warmup))
                if float(torch.rand(1).item()) < mix_ratio:
                    pts, lbl = sp_pts.to(device), sp_lbl.to(device)

        if coarse_mask_head is not None:
            target_size = getattr(model.prompt_encoder, "mask_input_size", (H, W))
            coarse_logits = coarse_mask_head(img_emb, output_size=target_size)
            coarse_mask_prompt = torch.sigmoid(coarse_logits)
            coarse_mask_loss_val = coarse_mask_prompt_loss(
                coarse_logits,
                masks,
                pos_weight=float(getattr(args, "coarse_mask_prompt_pos_weight", 10.0)),
                bce_weight=float(getattr(args, "coarse_mask_prompt_bce_weight", 1.0)),
                dice_weight=float(getattr(args, "coarse_mask_prompt_dice_weight", 1.0)),
            )

        # LCA-Prompt: mixed GT/LCA prompting
        lca_loss_val = None
        if lca_prompt is not None and pgap is None and self_prompt_head is None:
            lca_prompt.train()
            lca_pts, lca_lbl, lca_contrast, lca_loss_val = lca_prompt(
                images,
                neck_features=img_emb.detach(),
                gt_mask=masks,
            )
            # Determine mix ratio (schedule: ramp up over warmup epochs)
            lca_warmup = max(1, int(getattr(args, "lca_sup_warmup", 30)))
            lca_target_ratio = float(getattr(args, "lca_mix_ratio", 0.5))
            if getattr(args, "lca_mix_schedule", False):
                lca_ratio = min(1.0, epoch / lca_warmup) * lca_target_ratio
            else:
                lca_ratio = lca_target_ratio
            # Stochastic mixing per batch
            import random
            if random.random() < lca_ratio:
                lca_pts = lca_pts.unsqueeze(1).to(device)  # [B,1,K,2]
                lca_lbl = lca_lbl.unsqueeze(1).to(device)  # [B,1,K]
                pts = lca_pts
                lbl = lca_lbl
        if pgap_text_prior_only:
            text_sparse_embeds, text_dense_mask = None, None
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                coarse_mask_prompt,
                getattr(args, "coarse_mask_prompt_merge_alpha", 0.5),
            )
        else:
            text_sparse_embeds, text_dense_mask = _build_text_prompt_inputs(
                model, args, img_emb, clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                coarse_mask_prompt,
                getattr(args, "coarse_mask_prompt_merge_alpha", 0.5),
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt_eff,
                text_dense_mask,
                getattr(args, "text_dense_prompt_merge_alpha", 0.5),
            )
        contrastive_sparse_prompt = None
        contrastive_loss = None
        if contrastive_prompt is not None and pgap is None:
            contrastive_sparse_prompt, contrastive_loss = _build_self_prompt_sparse_tokens(
                args,
                contrastive_prompt,
                pts,
                lbl,
                image_size=(H, W),
                training=True,
            )
        dynamic_sparse_tokens = None
        dynamic_sparse_div_loss = None
        dynamic_sparse_target_loss = None
        if dynamic_sparse_prompt_head is not None:
            dyn_input = img_emb.detach() if bool(getattr(args, "dynamic_sparse_detach_image_embedding", False)) else img_emb
            if bool(getattr(args, "dynamic_sparse_multilevel", False)):
                dyn_levels = encoder_ms_embeddings
                if bool(getattr(args, "dynamic_sparse_detach_image_embedding", False)) and dyn_levels is not None:
                    dyn_levels = [feat.detach() for feat in dyn_levels]
                dynamic_sparse_out = dynamic_sparse_prompt_head(dyn_input, dyn_levels)
                if isinstance(dynamic_sparse_out, (tuple, list)):
                    dynamic_sparse_tokens, dynamic_sparse_aux = dynamic_sparse_out[0], dynamic_sparse_out[1]
                else:
                    dynamic_sparse_tokens, dynamic_sparse_aux = dynamic_sparse_out, {}
                if (
                    dynamic_sparse_aux is not None
                    and "targetness_logits" in dynamic_sparse_aux
                    and float(getattr(args, "dynamic_sparse_target_weight", 0.0)) > 0.0
                ):
                    dynamic_sparse_target_loss = dynamic_sparse_targetness_loss(
                        dynamic_sparse_aux["targetness_logits"],
                        masks,
                        pos_weight=float(getattr(args, "dynamic_sparse_target_pos_weight", 20.0)),
                        dice_weight=float(getattr(args, "dynamic_sparse_target_dice_weight", 1.0)),
                    )
            else:
                dynamic_sparse_tokens = dynamic_sparse_prompt_head(dyn_input)
            if float(getattr(args, "dynamic_sparse_div_weight", 0.0)) > 0.0:
                dynamic_sparse_div_loss = dynamic_sparse_token_diversity_loss(dynamic_sparse_tokens)
        sparse_prompt_eff = _merge_sparse_prompt_embeddings(
            text_sparse_embeds,
            contrastive_sparse_prompt,
        )
        sparse_prompt_eff = _merge_sparse_prompt_embeddings(
            sparse_prompt_eff,
            dynamic_sparse_tokens,
        )
        # HQ warmup: force using only HQ mask during early epochs
        use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
        if refine_self_prompt_head is not None and pgap is None:
            pred_masks_stage1, _ = model.predict_masks(
                img_emb,
                interms,
                pts,
                lbl,
                multi_scale_embeddings=detail_ms_embeddings,
                detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                batched_masks=mask_prompt_eff,
                text_sparse_embeddings=sparse_prompt_eff,
                multimask_output=False,
                input_h=H,
                input_w=W,
                output_h=H,
                output_w=W,
                hq_token_only=use_hq_only,
            )
            stage1_logits = pred_masks_stage1[:, 0, 0, ...].unsqueeze(1)
            if float(getattr(args, "two_stage_stage1_loss_weight", 0.0)) > 0.0:
                stage1_mask_loss_val = bce(stage1_logits, masks.unsqueeze(1)) + dice_loss(stage1_logits, masks.unsqueeze(1))

            _, stage2_pts, stage2_lbl, stage2_logits = refine_self_prompt_head(
                img_emb,
                coarse_logits=stage1_logits,
                output_size=(H, W),
                gt_mask=masks,
                positive_guidance=stage2_positive_guidance
                if stage2_positive_guidance is not None
                else build_self_prompt_positive_guidance(args, images, (H, W), prefix="stage2_self_prompt"),
            )
            stage2_types = getattr(refine_self_prompt_head, "last_point_types", None)
            (
                stage2_self_prompt_heat_loss,
                stage2_self_prompt_sampled_loss,
                stage2_self_prompt_expected_loss,
                stage2_self_prompt_component_loss,
            ) = compute_boundary_prompt_aux_losses(
                args,
                stage2_logits,
                stage2_pts,
                stage2_lbl,
                stage2_types,
                masks,
                prefix="stage2_self_prompt",
            )
            pts, lbl = stage2_pts.to(device), stage2_lbl.to(device)
            if contrastive_prompt is not None:
                contrastive_sparse_prompt, contrastive_loss = _build_self_prompt_sparse_tokens(
                    args,
                    contrastive_prompt,
                    pts,
                    lbl,
                    image_size=(H, W),
                    training=True,
                )
                sparse_prompt_eff = _merge_sparse_prompt_embeddings(
                    text_sparse_embeds,
                    contrastive_sparse_prompt,
                )
                sparse_prompt_eff = _merge_sparse_prompt_embeddings(
                    sparse_prompt_eff,
                    dynamic_sparse_tokens,
                )
        center_logits = None
        center_prob = None
        if getattr(args, "use_center_mask_decoder", False):
            pred_masks, _, aux = model.predict_masks(
                img_emb,
                interms,
                pts,
                lbl,
                multi_scale_embeddings=detail_ms_embeddings,
                detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                batched_masks=mask_prompt_eff,
                text_sparse_embeddings=sparse_prompt_eff,
                multimask_output=False,
                input_h=H,
                input_w=W,
                output_h=H,
                output_w=W,
                hq_token_only=use_hq_only,
                return_aux=True,
            )
            center_logits = aux.get("center_logits")
            center_prob = aux.get("center_prob")
            if center_logits is not None:
                center_logits = center_logits[:, 0, 0, ...].unsqueeze(1)
            if center_prob is not None:
                center_prob = center_prob[:, 0, 0, ...].unsqueeze(1)
        else:
            pred_masks, _ = model.predict_masks(
                img_emb,
                interms,
                pts,
                lbl,
                multi_scale_embeddings=detail_ms_embeddings,
                detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                batched_masks=mask_prompt_eff,
                text_sparse_embeddings=sparse_prompt_eff,
                multimask_output=False,
                input_h=H,
                input_w=W,
                output_h=H,
                output_w=W,
                hq_token_only=use_hq_only,
            )
        logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
        if not torch.isfinite(logits).all():
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            log_line(
                f"[warn] Skip non-finite logits at epoch {epoch:03d}, batch {batch_idx}",
                args.log_file,
            )
            continue
        loss = bce(logits, masks.unsqueeze(1)) + dice_loss(logits, masks.unsqueeze(1))
        if nwd_criterion is not None:
            loss = loss + nwd_weight * nwd_criterion(logits, masks.unsqueeze(1))
        tassg_losses = {
            "global": _zero_scalar(device),
            "token": _zero_scalar(device),
            "prompt": _zero_scalar(device),
            "targetness": _zero_scalar(device),
        }
        if tassg is not None:
            tassg_losses = _compute_tassg_distillation_losses(
                args=args,
                tassg_out=tassg_out,
                masks=masks,
                teacher_global=teacher_global,
                teacher_tokens=teacher_tokens,
                teacher_mask=teacher_mask,
                text_sparse_prompt=text_sparse_prompt,
            )
            loss = (
                loss
                + float(getattr(args, "lambda_tassg_global", 0.0)) * tassg_losses["global"]
                + float(getattr(args, "lambda_tassg_token", 0.0)) * tassg_losses["token"]
                + float(getattr(args, "lambda_tassg_prompt", 0.0)) * tassg_losses["prompt"]
                + float(getattr(args, "lambda_tassg_targetness", 0.0)) * tassg_losses["targetness"]
            )
        freq_weight = float(getattr(args, "freq_consistency_weight", 0.0))
        if freq_weight > 0.0:
            bins = max(2, int(getattr(args, "freq_consistency_bins", 32)))
            pred_prob = torch.sigmoid(logits)
            gt_mask = masks.unsqueeze(1).float()
            pred_profile = radial_frequency_profile(pred_prob, bins)
            gt_profile = radial_frequency_profile(gt_mask, bins)
            freq_loss = torch.mean(torch.abs(pred_profile - gt_profile))
            loss = loss + freq_weight * freq_loss
        # FAB Loss (Frequency-Aware Boundary Loss)
        if fab_criterion is not None:
            fab_loss_val = fab_criterion(logits, masks.unsqueeze(1))
            loss = loss + args.fab_weight * fab_loss_val
        # SCR Loss (Signal-to-Clutter Ratio Loss)
        if scr_criterion is not None:
            scr_loss_val = scr_criterion(logits, masks.unsqueeze(1), images)
            loss = loss + args.scr_weight * scr_loss_val
        if self_prompt_heat_loss is not None:
            loss = loss + float(getattr(args, "self_prompt_sup_weight", 0.3)) * self_prompt_heat_loss
        if self_prompt_sampled_loss is not None:
            loss = loss + float(getattr(args, "self_prompt_sampled_point_weight", 0.0)) * self_prompt_sampled_loss
        if self_prompt_expected_loss is not None:
            loss = loss + float(getattr(args, "self_prompt_expected_hit_weight", 0.0)) * self_prompt_expected_loss
        if self_prompt_component_loss is not None:
            loss = loss + float(getattr(args, "self_prompt_component_peak_weight", 0.0)) * self_prompt_component_loss
        if stage2_self_prompt_heat_loss is not None:
            loss = loss + float(getattr(args, "stage2_self_prompt_sup_weight", getattr(args, "self_prompt_sup_weight", 0.3))) * stage2_self_prompt_heat_loss
        if stage2_self_prompt_sampled_loss is not None:
            loss = loss + float(getattr(args, "stage2_self_prompt_sampled_point_weight", getattr(args, "self_prompt_sampled_point_weight", 0.0))) * stage2_self_prompt_sampled_loss
        if stage2_self_prompt_expected_loss is not None:
            loss = loss + float(getattr(args, "stage2_self_prompt_expected_hit_weight", getattr(args, "self_prompt_expected_hit_weight", 0.0))) * stage2_self_prompt_expected_loss
        if stage2_self_prompt_component_loss is not None:
            loss = loss + float(getattr(args, "stage2_self_prompt_component_peak_weight", getattr(args, "self_prompt_component_peak_weight", 0.0))) * stage2_self_prompt_component_loss
        if stage1_mask_loss_val is not None:
            loss = loss + float(getattr(args, "two_stage_stage1_loss_weight", 0.0)) * stage1_mask_loss_val
        if coarse_mask_loss_val is not None:
            loss = loss + float(getattr(args, "coarse_mask_prompt_weight", 0.0)) * coarse_mask_loss_val
        if contrastive_loss is not None and float(getattr(args, "self_prompt_cl_weight", 0.0)) > 0.0:
            loss = loss + float(getattr(args, "self_prompt_cl_weight", 0.0)) * contrastive_loss
        if dynamic_sparse_div_loss is not None:
            loss = loss + float(getattr(args, "dynamic_sparse_div_weight", 0.0)) * dynamic_sparse_div_loss
        if dynamic_sparse_target_loss is not None:
            loss = loss + float(getattr(args, "dynamic_sparse_target_weight", 0.0)) * dynamic_sparse_target_loss
        center_loss_val = None
        center_contain_val = None
        if center_logits is not None:
            center_gt = build_center_gaussian_targets(
                masks,
                sigma=float(getattr(args, "center_gaussian_sigma", 2.0)),
            )
            center_loss_val = self_prompt_heatmap_loss(
                center_logits,
                center_gt,
                pos_weight=float(getattr(args, "center_pos_weight", 10.0)),
            )
            if center_prob is None:
                center_prob = torch.sigmoid(center_logits)
            mask_prob = torch.sigmoid(logits)
            center_contain_val = torch.mean(center_prob * (1.0 - mask_prob))
            loss = loss + float(getattr(args, "center_loss_weight", 0.2)) * center_loss_val
            loss = loss + float(getattr(args, "center_contain_weight", 0.05)) * center_contain_val
        # Prompt-Robust Consistency Loss
        if getattr(args, 'use_prompt_robust_loss', False) and pgap is None:
            # 生成扰动 prompt 点
            with torch.no_grad():
                perturb_std = float(getattr(args, 'prompt_robust_perturb_std', 3.0))
                pts_perturbed = pts.clone()
                noise = torch.randn_like(pts_perturbed.float()) * perturb_std
                pts_perturbed = pts_perturbed + noise
                # Clamp to valid image range
                pts_perturbed[..., 0] = pts_perturbed[..., 0].clamp(0, W - 1)
                pts_perturbed[..., 1] = pts_perturbed[..., 1].clamp(0, H - 1)
            text_sparse_embeds_p, text_dense_mask_p = _build_text_prompt_inputs(
                model, args, img_emb.detach(), clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            contrastive_sparse_prompt_p, _ = _build_self_prompt_sparse_tokens(
                args,
                contrastive_prompt,
                pts_perturbed,
                lbl,
                image_size=(H, W),
                training=False,
            )
            sparse_prompt_eff_p = _merge_sparse_prompt_embeddings(
                text_sparse_embeds_p,
                contrastive_sparse_prompt_p,
            )
            pred_masks_p, _ = model.predict_masks(
                img_emb.detach(),  # detach to avoid double backward through encoder
                interms,
                pts_perturbed,
                lbl,
                multi_scale_embeddings=detail_ms_embeddings,
                detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                batched_masks=_merge_dense_mask_prompts(
                    mask_prompt,
                    text_dense_mask_p,
                    getattr(args, "text_dense_prompt_merge_alpha", 0.5),
                ),
                text_sparse_embeddings=sparse_prompt_eff_p,
                multimask_output=False,
                input_h=H, input_w=W,
                output_h=H, output_w=W,
                hq_token_only=use_hq_only,
            )
            logits_p = pred_masks_p[:, 0, 0, ...].unsqueeze(1)
            # 一致性损失: 两次预测应该相似 (Dice)
            prob_clean = torch.sigmoid(logits.detach())
            prob_perturb = torch.sigmoid(logits_p)
            inter_c = (prob_clean * prob_perturb).sum(dim=(1, 2, 3))
            denom_c = prob_clean.sum(dim=(1, 2, 3)) + prob_perturb.sum(dim=(1, 2, 3))
            consist_dice = 1 - ((2 * inter_c + 1.0) / (denom_c + 1.0)).mean()
            # 扰动预测也应对齐 GT
            consist_bce = F.binary_cross_entropy_with_logits(logits_p, masks.unsqueeze(1))
            prompt_robust_w = float(getattr(args, 'prompt_robust_weight', 0.1))
            loss = loss + prompt_robust_w * (consist_dice + 0.5 * consist_bce)

        if args.use_point_loss:
            # point-based loss on uncertain points
            num_points = args.point_loss_points
            coords = _get_uncertain_point_coords(logits.detach(), num_points=num_points,
                                                 oversample_ratio=args.point_loss_oversample,
                                                 importance_sample_ratio=args.point_loss_importance)
            gt_points = point_sample(masks.unsqueeze(1).float(), coords, align_corners=False)  # [B,1,P]
            pr_points = point_sample(logits, coords, align_corners=False)  # [B,1,P]
            # BCE at points
            bce_points = F.binary_cross_entropy_with_logits(pr_points, gt_points, reduction='none').mean(1).mean()
            # Dice at points (use same formula as image-wise)
            pr_sig = torch.sigmoid(pr_points)
            num = 2 * (pr_sig * gt_points).sum(dim=2)
            den = pr_sig.sum(dim=2) + gt_points.sum(dim=2)
            dice_pts = 1 - (num + 1) / (den + 1)
            dice_pts = dice_pts.mean()
            loss = loss + args.point_loss_weight * (bce_points + dice_pts)

        # LCA-Prompt auxiliary supervision loss
        if lca_loss_val is not None:
            lca_warmup = max(1, int(getattr(args, "lca_sup_warmup", 30)))
            lca_weight = float(getattr(args, "lca_sup_weight", 0.5))
            # Warmup schedule: ramp up loss weight
            if epoch <= lca_warmup:
                lca_weight = lca_weight * (epoch / lca_warmup)
            loss = loss + lca_weight * lca_loss_val

        if not torch.isfinite(loss):
            skipped_nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            log_line(
                f"[warn] Skip non-finite loss at epoch {epoch:03d}, batch {batch_idx}",
                args.log_file,
            )
            continue

        scaler.scale(loss / float(grad_accum_steps)).backward()
        accum_count += 1
        should_step = accum_count >= grad_accum_steps or batch_idx == num_batches
        if should_step:
            scaler.unscale_(optimizer)
            # Correct the final partial accumulation window so it has the same
            # gradient scale as a full window.
            if accum_count < grad_accum_steps:
                correction = float(grad_accum_steps) / float(max(1, accum_count))
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.mul_(correction)
            if grad_clip_norm > 0.0:
                params_with_grad = [
                    param
                    for group in optimizer.param_groups
                    for param in group["params"]
                    if param.grad is not None
                ]
                if params_with_grad:
                    torch.nn.utils.clip_grad_norm_(params_with_grad, grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        meter_loss += loss.item() * B
        if center_loss_val is not None:
            meter_center_loss += center_loss_val.item() * B
        if center_contain_val is not None:
            meter_center_contain += center_contain_val.item() * B
        if self_prompt_sampled_loss is not None:
            meter_sp_sampled += self_prompt_sampled_loss.item() * B
        if self_prompt_expected_loss is not None:
            meter_sp_expected += self_prompt_expected_loss.item() * B
        if self_prompt_component_loss is not None:
            meter_sp_component += self_prompt_component_loss.item() * B
        if stage2_self_prompt_heat_loss is not None:
            meter_stage2_sp += stage2_self_prompt_heat_loss.item() * B
        if stage2_self_prompt_sampled_loss is not None:
            meter_stage2_sampled += stage2_self_prompt_sampled_loss.item() * B
        if stage2_self_prompt_expected_loss is not None:
            meter_stage2_expected += stage2_self_prompt_expected_loss.item() * B
        if stage2_self_prompt_component_loss is not None:
            meter_stage2_component += stage2_self_prompt_component_loss.item() * B
        if stage1_mask_loss_val is not None:
            meter_stage1_mask += stage1_mask_loss_val.item() * B
        if coarse_mask_loss_val is not None:
            meter_coarse_mask += coarse_mask_loss_val.item() * B
        if dynamic_sparse_div_loss is not None:
            meter_dyn_sparse += dynamic_sparse_div_loss.item() * B
        if dynamic_sparse_target_loss is not None:
            meter_dyn_target += dynamic_sparse_target_loss.item() * B
        if tassg is not None:
            meter_tassg_global += float(tassg_losses["global"].detach().item()) * B
            meter_tassg_token += float(tassg_losses["token"].detach().item()) * B
            meter_tassg_prompt += float(tassg_losses["prompt"].detach().item()) * B
            meter_tassg_targetness += float(tassg_losses["targetness"].detach().item()) * B
        n += B
    if skipped_nonfinite > 0:
        log_line(
            f"[warn] Skipped {skipped_nonfinite} non-finite train batches at epoch {epoch:03d}",
            args.log_file,
        )
    denom = max(n, 1)
    return {
        "loss": meter_loss / denom,
        "center_loss": meter_center_loss / denom,
        "center_contain": meter_center_contain / denom,
        "self_prompt_sampled": meter_sp_sampled / denom,
        "self_prompt_expected": meter_sp_expected / denom,
        "self_prompt_component": meter_sp_component / denom,
        "stage2_self_prompt": meter_stage2_sp / denom,
        "stage2_self_prompt_sampled": meter_stage2_sampled / denom,
        "stage2_self_prompt_expected": meter_stage2_expected / denom,
        "stage2_self_prompt_component": meter_stage2_component / denom,
        "stage1_mask": meter_stage1_mask / denom,
        "coarse_mask": meter_coarse_mask / denom,
        "dynamic_sparse": meter_dyn_sparse / denom,
        "dynamic_sparse_target": meter_dyn_target / denom,
        "tassg_global": meter_tassg_global / denom,
        "tassg_token": meter_tassg_token / denom,
        "tassg_prompt": meter_tassg_prompt / denom,
        "tassg_targetness": meter_tassg_targetness / denom,
        "semantic_source": _resolve_semantic_source(args, epoch=epoch),
    }


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    args,
    epoch: int,
    pgap=None,
    text_conditioner=None,
    text_sparse_prompt=None,
    text_dense_prompt=None,
    tassg=None,
    bifusion_adapter=None,
    backbone_bifusion_adapter=None,
    self_prompt_head=None,
    refine_self_prompt_head=None,
    coarse_mask_head=None,
    dynamic_sparse_prompt_head=None,
    contrastive_prompt=None,
    lca_prompt=None,
):
    model.eval()
    if pgap is not None:
        pgap.eval()
    if self_prompt_head is not None:
        self_prompt_head.eval()
    if refine_self_prompt_head is not None:
        refine_self_prompt_head.eval()
    if coarse_mask_head is not None:
        coarse_mask_head.eval()
    if dynamic_sparse_prompt_head is not None:
        dynamic_sparse_prompt_head.eval()
    if contrastive_prompt is not None:
        contrastive_prompt.eval()
    if tassg is not None:
        tassg.eval()
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    # Compute metrics exactly matching definitions:
    # mIoU: Global Intersection / Global Union
    # nIoU: Mean of per-image IoUs
    f1s = []
    global_inter = 0.0
    global_union = 0.0
    niou_sum = 0.0
    niou_count = 0
    thr_sum = 0.0
    thr_count = 0
    pd_fa = PD_FA(distance_thresh=getattr(args, "pd_fa_dist", 3))
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        B, H, W = masks.shape
        clip_feat = None
        raw_clip_feat = None
        clip_token_feat = None
        clip_token_mask = None
        encoder_ms_embeddings = None
        semantic_source = _resolve_semantic_source(args, epoch=epoch)
        if semantic_source == "teacher":
            if "clip_text_feat" in batch:
                clip_feat = batch["clip_text_feat"].to(device, non_blocking=True)
                raw_clip_feat = clip_feat
            if "clip_text_token_feat" in batch:
                clip_token_feat = batch["clip_text_token_feat"].to(device, non_blocking=True)
                if "clip_text_attn_mask" in batch:
                    clip_token_mask = batch["clip_text_attn_mask"].to(device, non_blocking=True)
        elif semantic_source == "student" and tassg is None:
            raise RuntimeError("--semantic_source student requires --use_tassg.")

        use_teacher_backbone = (
            semantic_source == "teacher"
            and (not pgap_text_prior_only)
            and backbone_bifusion_adapter is not None
        )
        if use_teacher_backbone:
            img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                model=model,
                backbone_bifusion_adapter=backbone_bifusion_adapter,
                images=images,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        else:
            use_detail_outputs = bool(getattr(args, "use_hldf", False) or _uses_dsb_amgd(args))
            image_embedding_output = model.get_image_embeddings(images)
            img_emb, interms, detail_ms_embeddings = _unpack_image_embeddings_output(
                image_embedding_output,
                use_detail_outputs=use_detail_outputs,
            )
            encoder_ms_embeddings = _extract_encoder_ms_embeddings(
                image_embedding_output,
                use_detail_outputs=use_detail_outputs,
            )
        if tassg is not None and semantic_source == "student":
            tassg_out = tassg(img_emb)
            clip_feat = tassg_out["global"]
            raw_clip_feat = clip_feat
            clip_token_feat = tassg_out["tokens"]
            clip_token_mask = tassg_out["attn_mask"]
            if (
                bool(getattr(args, "tassg_two_pass_backbone", False))
                and (not pgap_text_prior_only)
                and backbone_bifusion_adapter is not None
            ):
                img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_backbone_bifusion_adapter(
                    model=model,
                    backbone_bifusion_adapter=backbone_bifusion_adapter,
                    images=images,
                    clip_feat=tassg_out["global"],
                    clip_token_feat=tassg_out["tokens"],
                    clip_token_mask=tassg_out["attn_mask"],
                )
                raw_clip_feat = tassg_out["global"]
        if (not pgap_text_prior_only) and semantic_source != "none" and bifusion_adapter is not None:
            img_emb, interms, detail_ms_embeddings, clip_feat, clip_token_feat, clip_token_mask = _apply_bifusion_adapter(
                bifusion_adapter=bifusion_adapter,
                img_emb=img_emb,
                interms=interms,
                detail_ms_embeddings=detail_ms_embeddings,
                clip_feat=clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
            )
        if (not pgap_text_prior_only) and semantic_source != "none" and text_conditioner is not None and clip_feat is not None:
            img_emb = text_conditioner(img_emb, clip_feat)
        pgap_text_prior = None
        if pgap is not None:
            pgap_text_prior = _build_pgap_text_prior(
                model,
                args,
                img_emb,
                clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_dense_prompt=text_dense_prompt,
                output_size=(H, W),
            )
        mask_prompt = None
        if pgap is not None:
            if getattr(args, "pgap_two_stage", False):
                pgap_pts, pgap_lbl, saliency = _run_pgap_with_text_prior(
                    pgap, images, args, text_prior=pgap_text_prior
                )
                if args.use_feature_mod:
                    img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pos_pts, pos_lbl = _select_topk_points(pgap_pts, pgap_lbl, args.pgap_stage1_top_k)
                pts1 = pos_pts.unsqueeze(1).to(device)
                lbl1 = pos_lbl.unsqueeze(1).to(device)
                use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
                if pgap_text_prior_only:
                    text_sparse_stage1, text_dense_stage1 = None, None
                else:
                    text_sparse_stage1, text_dense_stage1 = _build_text_prompt_inputs(
                        model, args, img_emb, clip_feat,
                        raw_clip_feat=raw_clip_feat,
                        clip_token_feat=clip_token_feat,
                        clip_token_mask=clip_token_mask,
                        text_sparse_prompt=text_sparse_prompt,
                        text_dense_prompt=text_dense_prompt,
                    )
                pred_masks1, _ = model.predict_masks(
                    img_emb,
                    interms,
                    pts1,
                    lbl1,
                    multi_scale_embeddings=detail_ms_embeddings,
                    detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                    batched_masks=_merge_dense_mask_prompts(
                        mask_prompt,
                        text_dense_stage1,
                        getattr(args, "text_dense_prompt_merge_alpha", 0.5),
                    ),
                    text_sparse_embeddings=text_sparse_stage1,
                    multimask_output=False,
                    input_h=H,
                    input_w=W,
                    output_h=H,
                    output_w=W,
                    hq_token_only=use_hq_only,
                )
                logits1 = pred_masks1[:, 0, 0, ...].unsqueeze(1)
                coarse = (torch.sigmoid(logits1) >= args.pgap_stage1_thr).float()
                neg_pts, neg_lbl = pgap.select_negatives_from_mask(
                    pgap_pts, pgap_lbl, saliency, coarse[:, 0], args.pgap_stage2_neg
                )
                pts = torch.cat([pos_pts, neg_pts], dim=1).unsqueeze(1)
                lbl = torch.cat([pos_lbl, neg_lbl], dim=1).unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
            else:
                pgap_pts, pgap_lbl, saliency = _build_pgap_prompts(
                    pgap, images, masks, args, text_prior=pgap_text_prior
                )
                if args.use_feature_mod:
                    img_emb = model.apply_saliency_modulation(img_emb, saliency)
                if args.use_mask_prompt:
                    target_size = getattr(model.prompt_encoder, "mask_input_size", saliency.shape[-2:])
                    mask_prompt = F.interpolate(
                        saliency, size=target_size, mode="bilinear", align_corners=False
                    )
                pts = pgap_pts.unsqueeze(1)
                lbl = pgap_lbl.unsqueeze(1)
                pts, lbl = pts.to(device), lbl.to(device)
        else:
            if self_prompt_head is not None:
                self_prompt_positive_guidance = build_self_prompt_positive_guidance(
                    args,
                    images,
                    (H, W),
                    prefix="self_prompt",
                )
                _, pts, lbl, _ = self_prompt_head(
                    img_emb,
                    output_size=(H, W),
                    positive_guidance=self_prompt_positive_guidance,
                )
                pts, lbl = pts.to(device), lbl.to(device)
            elif lca_prompt is not None:
                lca_prompt.eval()
                lca_pts, lca_lbl, _, _ = lca_prompt(
                    images,
                    neck_features=img_emb,
                    gt_mask=None,
                )
                pts = lca_pts.unsqueeze(1).to(device)
                lbl = lca_lbl.unsqueeze(1).to(device)
            else:
                if str(getattr(args, "prompt_mode", "gt_points")) == "assp_only":
                    pts, lbl = make_empty_point_prompt(int(masks.shape[0]), device)
                else:
                    pts, lbl = sample_points_from_mask(
                        masks,
                        n_pos=args.n_pos,
                        n_neg=args.n_neg,
                        boundary_prior=bool(args.boundary_prior_sampling),
                        boundary_ratio=float(args.boundary_ratio),
                    )
                    pts, lbl = pts.to(device), lbl.to(device)
        coarse_mask_prompt = None
        if coarse_mask_head is not None:
            target_size = getattr(model.prompt_encoder, "mask_input_size", (H, W))
            coarse_logits = coarse_mask_head(img_emb, output_size=target_size)
            coarse_mask_prompt = torch.sigmoid(coarse_logits)
        if pgap_text_prior_only:
            text_sparse_embeds, text_dense_mask = None, None
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                coarse_mask_prompt,
                getattr(args, "coarse_mask_prompt_merge_alpha", 0.5),
            )
        else:
            text_sparse_embeds, text_dense_mask = _build_text_prompt_inputs(
                model, args, img_emb, clip_feat,
                raw_clip_feat=raw_clip_feat,
                clip_token_feat=clip_token_feat,
                clip_token_mask=clip_token_mask,
                text_sparse_prompt=text_sparse_prompt,
                text_dense_prompt=text_dense_prompt,
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt,
                coarse_mask_prompt,
                getattr(args, "coarse_mask_prompt_merge_alpha", 0.5),
            )
            mask_prompt_eff = _merge_dense_mask_prompts(
                mask_prompt_eff,
                text_dense_mask,
                getattr(args, "text_dense_prompt_merge_alpha", 0.5),
            )
        contrastive_sparse_prompt, _ = _build_self_prompt_sparse_tokens(
            args,
            contrastive_prompt,
            pts,
            lbl,
            image_size=(H, W),
            training=False,
        )
        dynamic_sparse_tokens = None
        if dynamic_sparse_prompt_head is not None:
            if bool(getattr(args, "dynamic_sparse_multilevel", False)):
                dynamic_sparse_out = dynamic_sparse_prompt_head(img_emb, encoder_ms_embeddings)
                dynamic_sparse_tokens = dynamic_sparse_out[0] if isinstance(dynamic_sparse_out, (tuple, list)) else dynamic_sparse_out
            else:
                dynamic_sparse_tokens = dynamic_sparse_prompt_head(img_emb)
        sparse_prompt_eff = _merge_sparse_prompt_embeddings(
            text_sparse_embeds,
            contrastive_sparse_prompt,
        )
        sparse_prompt_eff = _merge_sparse_prompt_embeddings(
            sparse_prompt_eff,
            dynamic_sparse_tokens,
        )
        use_hq_only = bool(args.hq_token_only or (args.hq_warmup_epochs > 0 and epoch <= args.hq_warmup_epochs))
        if refine_self_prompt_head is not None and pgap is None:
            pred_masks_stage1, _ = model.predict_masks(
                img_emb,
                interms,
                pts,
                lbl,
                multi_scale_embeddings=detail_ms_embeddings,
                detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
                batched_masks=mask_prompt_eff,
                text_sparse_embeddings=sparse_prompt_eff,
                multimask_output=False,
                input_h=H,
                input_w=W,
                output_h=H,
                output_w=W,
                hq_token_only=use_hq_only,
            )
            stage1_logits = pred_masks_stage1[:, 0, 0, ...].unsqueeze(1)
            _, pts, lbl, _ = refine_self_prompt_head(
                img_emb,
                coarse_logits=stage1_logits,
                output_size=(H, W),
                positive_guidance=build_self_prompt_positive_guidance(
                    args,
                    images,
                    (H, W),
                    prefix="stage2_self_prompt",
                ),
            )
            pts, lbl = pts.to(device), lbl.to(device)
            if contrastive_prompt is not None:
                contrastive_sparse_prompt, _ = _build_self_prompt_sparse_tokens(
                    args,
                    contrastive_prompt,
                    pts,
                    lbl,
                    image_size=(H, W),
                    training=False,
                )
                sparse_prompt_eff = _merge_sparse_prompt_embeddings(
                    text_sparse_embeds,
                    contrastive_sparse_prompt,
                )
                sparse_prompt_eff = _merge_sparse_prompt_embeddings(
                    sparse_prompt_eff,
                    dynamic_sparse_tokens,
                )
        pred_masks, _ = model.predict_masks(
            img_emb,
            interms,
            pts,
            lbl,
            multi_scale_embeddings=detail_ms_embeddings,
            detail_branch_embeddings=_detail_branch_embeddings_arg(args, detail_ms_embeddings),
            batched_masks=mask_prompt_eff,
            text_sparse_embeddings=sparse_prompt_eff,
            multimask_output=False,
            input_h=H,
            input_w=W,
            output_h=H,
            output_w=W,
            hq_token_only=use_hq_only,
        )
        logits = pred_masks[:, 0, 0, ...].unsqueeze(1)
        if args.val_thr_search:
            best_iou, best_thr = -1.0, args.thr
            thr = args.val_thr_min
            while thr <= args.val_thr_max + 1e-6:
                miou_t, _ = compute_metrics(logits, masks.unsqueeze(1), thr=thr)
                if miou_t > best_iou:
                    best_iou, best_thr = miou_t, thr
                thr += args.val_thr_step
            thr_used = best_thr
        else:
            thr_used = args.thr

        prob = torch.sigmoid(logits)
        pred = (prob >= thr_used).float()
        target = masks.unsqueeze(1).float()
        inter = (pred * target).sum().item()
        union = (pred + target - pred * target).sum().item()
        global_inter += inter
        global_union += union
        inter_s = (pred * target).sum(dim=(1, 2, 3))
        union_s = (pred + target - pred * target).sum(dim=(1, 2, 3))
        iou_s = torch.where(union_s > 0, inter_s / union_s, torch.ones_like(union_s))
        niou_sum += iou_s.sum().item()
        niou_count += int(iou_s.numel())
        f1_batch = compute_metrics(logits, masks.unsqueeze(1), thr=thr_used)[1]
        f1s.append(f1_batch)

        pred_cpu = pred.detach().cpu()
        target_cpu = target.detach().cpu()
        for b in range(pred_cpu.shape[0]):
            pd_fa.update(pred_cpu[b, 0], target_cpu[b, 0], (H, W))

        thr_sum += float(thr_used)
        thr_count += 1

    miou_avg = (global_inter / global_union) if global_union > 0 else 1.0
    niou_avg = niou_sum / niou_count if niou_count > 0 else 0.0
    f1_avg = sum(f1s) / len(f1s) if f1s else 0.0
    pd_val, fa_val = pd_fa.get()
    thr_used = (thr_sum / thr_count) if thr_count > 0 else args.thr
    return miou_avg, niou_avg, f1_avg, pd_val, fa_val, thr_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--train_txt", type=str, default="train.txt")
    p.add_argument("--val_txt", type=str, default="test.txt")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--keep_ratio_pad", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--n_pos", type=int, default=4)
    p.add_argument("--n_neg", type=int, default=4)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_accum_steps", type=int, default=1,
                   help="Accumulate gradients for N micro-batches before an optimizer step.")
    p.add_argument("--grad_clip_norm", type=float, default=0.0,
                   help="Clip the global gradient norm after AMP unscaling; <=0 disables clipping.")
    p.add_argument("--out_dir", type=str, default="./outputs_sam_sirst_hq")
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--model", type=str, default="vitt", choices=["vitt", "vits"])  # kept for naming
    p.add_argument("--hq_token_only", action="store_true")
    p.add_argument("--hq_warmup_epochs", type=int, default=0,
                   help="If >0, use HQ token only for the first N epochs.")
    p.add_argument("--init_from_baseline", type=str, default=None,
                   help="Optional path to EfficientSAM baseline checkpoint to partially initialize from.")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Resume training from a checkpoint saved by this script, restoring model/optimizer/best_iou when possible.")
    p.add_argument("--resume_reset_optimizer", action="store_true",
                   help="Resume model/module weights but rebuild optimizer state with the current lr settings.")
    p.add_argument("--resume_reset_best", action="store_true",
                   help="When resuming, reset best_iou so the fine-tuning run saves its own best checkpoint.")
    p.add_argument("--use_fs_adapter", action="store_true",
                   help="Enable frequency-spatial adapter inside ViT blocks.")
    p.add_argument("--use_ms_fusion", action="store_true",
                   help="Enable multi-scale fusion from intermediate ViT blocks.")
    p.add_argument("--use_detail_enhancer", action="store_true",
                   help="Enable Sobel detail enhancer on shallow features.")
    p.add_argument("--early_exit_layer", type=int, default=0,
                   help="Exit after N transformer blocks (1-based). Use 0 to disable.")
    # Radial gate for HQ-SAM (optional)
    p.add_argument("--use_radial_gate_hq", action="store_true", help="Enable RadialFreqGate for HQ-SAM.")
    p.add_argument("--rgate_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply radial gate: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--freq_patch_size_hq", type=int, default=8)
    p.add_argument("--radial_bins_hq", type=int, default=6)
    p.add_argument("--radial_channel_shared_hq", action="store_true")
    p.add_argument("--rgate_strength_enc", type=float, default=0.5)
    p.add_argument("--rgate_strength_dec", type=float, default=0.5)
    p.add_argument("--rgate_edge_boost", type=float, default=0.5,
                   help="Edge-aware high-frequency boost factor for RadialFreqGate (set 0 to disable).")
    p.add_argument("--rgate_high_freq_thresh", type=float, default=0.6,
                   help="Normalized radial threshold above which frequencies are treated as high.")
    p.add_argument("--use_amgd", action="store_true", help="Enable AMGD multi-grained feature extraction")
    p.add_argument("--use_dog_amgd", action="store_true", help="Use DoG (Difference of Gaussians) differential fusion within AMGD")
    p.add_argument("--amgd_routing", type=str, default="prompt", choices=["prompt", "uniform"],
                   help="AMGD branch weighting: prompt uses the HQ-token router; uniform fixes all three weights to 1/3.")
    p.add_argument("--dog_amgd_mode", type=str, default="legacy", choices=["legacy", "residual"],
                   help="DoG fusion mode: legacy reproduces the original DoG-AMGD path, residual uses AMGD base + scaled DoG residual.")
    p.add_argument("--amgd_interm_layer", type=int, default=0,
                   help="0-based ViT block index used as the HQ detail source for AMGD/DoG (default 0 preserves legacy runs).")
    p.add_argument("--amgd_branch_design", type=str, default="legacy", choices=["legacy", "dsb_v1"],
                   help="AMGD branch design: legacy keeps the current fine/mid/coarse branches; dsb_v1 uses detail/structure/background branches.")
    p.add_argument("--amgd_detail_layer", type=int, default=None,
                   help="0-based ViT block index used as the detail branch source in dsb_v1. Defaults to --amgd_interm_layer.")
    p.add_argument("--amgd_structure_layer", type=int, default=None,
                   help="0-based ViT block index used as the structure branch source in dsb_v1. Defaults to --amgd_interm_layer.")
    p.add_argument("--amgd_background_layer", type=int, default=None,
                   help="0-based ViT block index used as the background branch source in dsb_v1. Defaults to --amgd_interm_layer.")
    p.add_argument("--dog_amgd_strength", type=float, default=0.25,
                   help="Residual strength of DoG enhancement added on top of AMGD base fusion (used only in residual mode).")
    p.add_argument("--use_hldf", action="store_true",
                   help="Enable Hierarchical Layer-wise Detail Fusion in decoder HQ branch.")
    p.add_argument("--hldf_layers", type=str, default="0,2,5",
                   help="Comma-separated 0-based ViT block indices used by HLDF, e.g. 0,2,5.")
    p.add_argument("--hldf_hidden_dim", type=int, default=96,
                   help="Hidden channel width used inside HLDF layer alignment and fusion.")
    p.add_argument("--hldf_use_hq_router", action=argparse.BooleanOptionalAction, default=True,
                   help="Use HQ-token router to predict HLDF layer weights and top-down gates.")
    p.add_argument("--hldf_router_temp", type=float, default=1.0,
                   help="Softmax temperature for the HLDF HQ-token router.")

    # ASG gate for HQ-SAM (optional)
    p.add_argument("--use_asg_hq", action="store_true", help="Enable AnisotropicSpectralGating for HQ-SAM.")
    p.add_argument("--asg_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply ASG: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--asg_radial_bins", type=int, default=64)
    p.add_argument("--asg_angular_bins", type=int, default=128)
    p.add_argument("--asg_variant", type=str, default="asg1", choices=["asg1", "asg2"],
                   help="Which ASG implementation to use.")
    p.add_argument("--asg_strength_enc", type=float, default=1.0)
    p.add_argument("--asg_strength_dec", type=float, default=1.0)
    p.add_argument("--asg_encoder_mode", type=str, default="neck", choices=["neck", "block_only"],
                   help="Encoder-side ASG placement: neck=apply on neck_out, block_only=apply only after selected ViT blocks.")
    p.add_argument("--asg_block_indices", type=str, default="11",
                   help="Comma-separated 0-based ViT block indices for block-only ASG, e.g. 11 or 5,11.")
    p.add_argument("--asg_block_strengths", type=str, default="0.25",
                   help="Comma-separated residual strengths for block-only ASG, matching asg_block_indices in length.")
    # AFD (Adaptive Frequency Decomposition) module for HQ-SAM
    p.add_argument("--use_afd_hq", action="store_true", help="Enable AdaptiveFrequencyDecomposition for HQ-SAM.")
    p.add_argument("--afd_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply AFD: encoder neck_out, decoder hq_features, or both.")
    p.add_argument("--afd_patch_size", type=int, default=8,
                   help="Patch size for AFD FFT processing.")
    p.add_argument("--afd_num_bins", type=int, default=16,
                   help="Number of discrete bins for cutoff prediction.")
    p.add_argument("--afd_low_ratio", type=float, default=1.0,
                   help="Low frequency gain initial value (learnable, default 1.0 = no suppression).")
    p.add_argument("--afd_high_ratio", type=float, default=1.0,
                   help="High frequency gain initial value (learnable, default 1.0 = no enhancement).")
    p.add_argument("--afd_learnable_gains", action="store_true", default=True,
                   help="Use learnable gains instead of fixed ratios (default True).")
    p.add_argument("--afd_fixed_gains", action="store_true",
                   help="Use fixed gains (disable learnable gains).")
    p.add_argument("--afd_channel_wise", action="store_true",
                   help="Use per-channel independent gains instead of global gains.")
    p.add_argument("--afd_strength_enc", type=float, default=0.5,
                   help="Residual strength for AFD at encoder.")
    p.add_argument("--afd_strength_dec", type=float, default=0.5,
                   help="Residual strength for AFD at decoder.")
    # MSFE (Multi-Scale Frequency Enhancement) module
    p.add_argument("--use_msfe_hq", action="store_true",
                   help="Enable MultiScaleFrequencyEnhancement for HQ-SAM.")
    p.add_argument("--msfe_loc", type=str, default="encoder", choices=["encoder", "decoder", "both"],
                   help="Where to apply MSFE: encoder, decoder, or both.")
    p.add_argument("--msfe_patch_sizes", type=str, default="4,8,16",
                   help="Comma-separated patch sizes, e.g., '4,8,16'.")
    p.add_argument("--msfe_num_bins", type=int, default=8,
                   help="Number of radial frequency bins per scale.")
    p.add_argument("--msfe_fusion", type=str, default="attention", choices=["attention", "concat", "average"],
                   help="Fusion method for multi-scale outputs.")
    p.add_argument("--msfe_strength_enc", type=float, default=0.5,
                   help="Residual strength for MSFE at encoder.")
    p.add_argument("--msfe_strength_dec", type=float, default=0.5,
                   help="Residual strength for MSFE at decoder.")
    p.add_argument("--freq_consistency_weight", type=float, default=0.0,
                   help="Weight for radial frequency consistency loss.")
    p.add_argument("--freq_consistency_bins", type=int, default=32,
                   help="Radial bins used for frequency consistency loss.")
    # FAB Loss (Frequency-Aware Boundary Loss)
    p.add_argument("--use_fab_loss", action="store_true",
                   help="Enable Frequency-Aware Boundary Loss for small target detection.")
    p.add_argument("--fab_weight", type=float, default=0.5,
                   help="Weight for FAB Loss.")
    p.add_argument("--fab_num_bins", type=int, default=16,
                   help="Number of radial frequency bins for FAB Loss.")
    p.add_argument("--fab_boundary_width", type=int, default=3,
                   help="Boundary extraction kernel size.")
    p.add_argument("--fab_high_freq_weight", type=float, default=2.0,
                   help="Weight multiplier for high frequency components.")
    # SCR Loss (Signal-to-Clutter Ratio Loss)
    p.add_argument("--use_scr_loss", action="store_true",
                   help="Enable Signal-to-Clutter Ratio Loss for IRSTD.")
    p.add_argument("--scr_weight", type=float, default=0.1,
                   help="Weight for SCR Loss.")
    p.add_argument("--scr_inner_k", type=int, default=5,
                   help="Inner annular dilation kernel size for SCR.")
    p.add_argument("--scr_outer_k", type=int, default=15,
                   help="Outer annular dilation kernel size for SCR.")
    # Prompt-Robust Consistency Loss
    p.add_argument("--use_prompt_robust_loss", action="store_true",
                   help="Enable Prompt-Robustness Consistency Loss.")
    p.add_argument("--prompt_robust_weight", type=float, default=0.1,
                   help="Weight for prompt robustness consistency loss.")
    p.add_argument("--prompt_robust_perturb_std", type=float, default=3.0,
                   help="Std of Gaussian noise added to prompt points (pixels).")
    p.add_argument("--use_self_prompting", action="store_true",
                   help="Enable a self-prompting head that predicts SAM prompt points from encoder features.")
    p.add_argument("--self_prompt_hidden_channels", type=int, default=64,
                   help="Hidden channels for the self-prompting heatmap head.")
    p.add_argument("--self_prompt_top_k_pos", type=int, default=None,
                   help="Number of positive self-prompt points (defaults to --n_pos).")
    p.add_argument("--self_prompt_top_k_neg", type=int, default=None,
                   help="Number of negative self-prompt points (defaults to --n_neg).")
    p.add_argument("--self_prompt_min_dist", type=int, default=8,
                   help="Minimum spacing between self-prompt peaks.")
    p.add_argument("--self_prompt_peak_thr", type=float, default=0.1,
                   help="Peak threshold used when extracting self-prompt candidates.")
    p.add_argument("--self_prompt_low_response_thr", type=float, default=0.3,
                   help="Fallback threshold for low-response negative sampling.")
    p.add_argument("--self_prompt_boundary_aware", action="store_true",
                   help="Predict foreground/inner-boundary/outer-boundary/background prompt maps instead of one object heatmap.")
    p.add_argument("--self_prompt_boundary_ratio", type=float, default=None,
                   help="Boundary prompt ratio used by boundary-aware self prompting; defaults to --boundary_ratio.")
    p.add_argument("--self_prompt_boundary_width", type=int, default=1,
                   help="GT boundary half-width used to supervise boundary-aware prompt maps.")
    p.add_argument("--self_prompt_target_mode", type=str, default="soft", choices=["hard", "soft"],
                   help="Target type for boundary-aware self-prompt supervision.")
    p.add_argument("--self_prompt_foreground_mode", type=str, default="mask", choices=["mask", "soft_mask"],
                   help="Foreground target used by boundary-aware self-prompt supervision.")
    p.add_argument("--self_prompt_soft_band_radius", type=int, default=3,
                   help="Radius in pixels for soft inner/outer boundary bands.")
    p.add_argument("--self_prompt_soft_sigma", type=float, default=1.5,
                   help="Exponential decay sigma for soft boundary bands.")
    p.add_argument("--self_prompt_background_margin", type=int, default=3,
                   help="Pixels around GT treated as unsafe for the safe-background negative map.")
    p.add_argument("--self_prompt_background_fade_radius", type=int, default=4,
                   help="Ramp-up radius after the safe-background no-sample margin.")
    p.add_argument("--self_prompt_boundary_pos_weight", type=float, default=10.0,
                   help="Positive-pixel BCE weight for inner/outer boundary prompt maps.")
    p.add_argument("--self_prompt_background_pos_weight", type=float, default=1.0,
                   help="Positive-pixel BCE weight for the safe-background prompt map.")
    p.add_argument("--self_prompt_background_loss_weight", type=float, default=0.25,
                   help="Relative loss weight for the safe-background prompt map.")
    p.add_argument("--self_prompt_loss_channels", type=str, default="all",
                   help="Boundary-aware prompt-map channels supervised by the dense loss: all, fg_bg, fg_bg_outer, or comma-separated names.")
    p.add_argument("--self_prompt_sup_weight", type=float, default=0.3,
                   help="Weight of the self-prompt heatmap supervision loss.")
    p.add_argument("--self_prompt_sampler_variant", type=str, default="legacy",
                   help="Sampler variant used by self-prompt head during training and validation.")
    p.add_argument("--self_prompt_positive_guidance", type=str, default="none", choices=["none", "dog_log"],
                   help="Optional positive-point guidance map. dog_log uses only multi-scale DoG/LoG band-pass saliency.")
    p.add_argument("--self_prompt_guidance_alpha", type=float, default=0.75,
                   help="Blend strength for positive guidance: fg*((1-alpha)+alpha*guidance).")
    p.add_argument("--self_prompt_guidance_bg_alpha", type=float, default=0.5,
                   help="Background-map gate strength for guided positive scores.")
    p.add_argument("--self_prompt_guidance_power", type=float, default=1.0,
                   help="Power applied to positive guidance before blending.")
    p.add_argument("--dog_log_dog_sigmas", type=str, default="0.7-1.4,1.0-2.0,1.5-3.0,2.0-4.0",
                   help="Comma-separated DoG sigma pairs for dog_log guidance, e.g. 0.7-1.4,1.0-2.0.")
    p.add_argument("--dog_log_log_sigmas", type=str, default="0.8,1.2,1.6,2.4",
                   help="Comma-separated LoG sigma values for dog_log guidance.")
    p.add_argument("--dog_log_truncate", type=float, default=3.0,
                   help="Gaussian kernel truncate radius in sigma units for dog_log guidance.")
    p.add_argument("--self_prompt_sampled_point_weight", type=float, default=0.0,
                   help="Weight for auxiliary loss on the actually sampled self-prompt points.")
    p.add_argument("--self_prompt_sampled_loss_mode", type=str, default="legacy", choices=["legacy", "channel_error"],
                   help="Sampled-point loss mode: legacy=max(pos/neg maps), channel_error=channel-aware error-only penalty.")
    p.add_argument("--self_prompt_channel_error_target_threshold", type=float, default=0.5,
                   help="Target threshold below which a sampled point is treated as channel-error in channel_error mode.")
    p.add_argument("--self_prompt_expected_hit_weight", type=float, default=0.0,
                   help="Weight for soft expected-hit loss over boundary-aware prompt maps.")
    p.add_argument("--self_prompt_component_peak_weight", type=float, default=0.0,
                   help="Weight for foreground component peak loss; encourages every GT component to contain a high foreground peak.")
    p.add_argument("--self_prompt_component_peak_temperature", type=float, default=0.5,
                   help="Temperature for smooth component peak pooling.")
    p.add_argument("--self_prompt_component_peak_min_area", type=int, default=1,
                   help="Minimum GT component area used by foreground component peak loss.")
    p.add_argument("--self_prompt_component_peak_max_components", type=int, default=256,
                   help="Maximum GT components per image used by foreground component peak loss.")
    p.add_argument("--use_two_stage_self_prompting", action="store_true",
                   help="Run decoder once with stage-1 self prompts, then predict refined stage-2 prompts from image embedding plus stage-1 mask.")
    p.add_argument("--stage2_self_prompt_hidden_channels", type=int, default=64,
                   help="Hidden channels for the mask-guided stage-2 self-prompt head.")
    p.add_argument("--stage2_self_prompt_mask_feature_channels", type=int, default=3,
                   help="Number of stage-1 mask evidence channels used by the stage-2 head: logit, prob, uncertainty.")
    p.add_argument("--stage2_self_prompt_detach_mask", action=argparse.BooleanOptionalAction, default=True,
                   help="Detach stage-1 mask before feeding it to the stage-2 prompt head.")
    p.add_argument("--stage2_self_prompt_sampler_variant", type=str, default="fgcomp_bg_r05",
                   help="Sampler variant used by the stage-2 mask-guided prompt head.")
    p.add_argument("--stage2_self_prompt_positive_guidance", type=str, default=None, choices=["none", "dog_log"],
                   help="Optional positive guidance for the stage-2 prompt head. Defaults to --self_prompt_positive_guidance.")
    p.add_argument("--stage2_self_prompt_guidance_alpha", type=float, default=None,
                   help="Stage-2 override for positive guidance blend strength.")
    p.add_argument("--stage2_self_prompt_guidance_bg_alpha", type=float, default=None,
                   help="Stage-2 override for background-map gate strength in guided positive scores.")
    p.add_argument("--stage2_self_prompt_guidance_power", type=float, default=None,
                   help="Stage-2 override for positive guidance power.")
    p.add_argument("--stage2_self_prompt_loss_channels", type=str, default="fg_bg",
                   help="Boundary-aware prompt-map channels supervised for stage-2 head.")
    p.add_argument("--stage2_self_prompt_sup_weight", type=float, default=0.3,
                   help="Weight of stage-2 prompt-map supervision loss.")
    p.add_argument("--stage2_self_prompt_sampled_point_weight", type=float, default=0.0,
                   help="Weight for auxiliary loss on stage-2 sampled prompt points.")
    p.add_argument("--stage2_self_prompt_sampled_loss_mode", type=str, default="legacy", choices=["legacy", "channel_error"],
                   help="Stage-2 sampled-point loss mode.")
    p.add_argument("--stage2_self_prompt_channel_error_target_threshold", type=float, default=0.5,
                   help="Target threshold below which a stage-2 sampled point is treated as channel-error.")
    p.add_argument("--stage2_self_prompt_expected_hit_weight", type=float, default=0.0,
                   help="Weight for stage-2 expected-hit loss.")
    p.add_argument("--stage2_self_prompt_component_peak_weight", type=float, default=0.05,
                   help="Weight for stage-2 foreground component peak supervision.")
    p.add_argument("--stage2_self_prompt_component_peak_temperature", type=float, default=0.5,
                   help="Temperature for stage-2 component peak pooling.")
    p.add_argument("--stage2_self_prompt_component_peak_min_area", type=int, default=1,
                   help="Minimum GT component area used by stage-2 component peak loss.")
    p.add_argument("--stage2_self_prompt_component_peak_max_components", type=int, default=256,
                   help="Maximum GT components per image used by stage-2 component peak loss.")
    p.add_argument("--two_stage_stage1_loss_weight", type=float, default=0.2,
                   help="Auxiliary BCE+Dice weight for the first-stage mask used as mask feedback.")
    p.add_argument("--self_prompt_sampled_pos_weight", type=float, default=1.0,
                   help="Positive-point term weight inside sampled-point loss.")
    p.add_argument("--self_prompt_sampled_neg_weight", type=float, default=1.0,
                   help="Negative-point term weight inside sampled-point loss.")
    p.add_argument("--self_prompt_pos_weight", type=float, default=10.0,
                   help="Positive-pixel weight used in the self-prompt heatmap loss.")
    p.add_argument("--self_prompt_mix_ratio", type=float, default=0.5,
                   help="Probability of replacing GT prompts with self-prompted points during training.")
    p.add_argument("--self_prompt_warmup", type=int, default=30,
                   help="Warmup epochs used by the self-prompt mix schedule.")
    p.add_argument("--self_prompt_mix_start_epoch", type=int, default=1,
                   help="First epoch where self-prompt points may replace GT prompts when mix scheduling is enabled.")
    p.add_argument("--self_prompt_mix_schedule", action="store_true",
                   help="Ramp the self-prompt mix ratio linearly during warmup.")
    p.add_argument("--self_prompt_cl_weight", type=float, default=0.03,
                   help="Weight of the contrastive prompt auxiliary loss.")
    p.add_argument("--self_prompt_cl_proj_dim", type=int, default=128,
                   help="Projection dimension used by the contrastive prompt module.")
    p.add_argument("--self_prompt_cl_temperature", type=float, default=0.07,
                   help="Temperature used by the contrastive prompt loss.")
    p.add_argument("--self_prompt_cl_loss", type=str, default="infonce", choices=["infonce", "ntxent", "triplet"],
                   help="Contrastive loss type for prompt embedding regularization.")
    p.add_argument("--self_prompt_inject_sparse_tokens", action="store_true",
                   help="Inject contrastive prompt embeddings as extra sparse tokens into the SAM prompt encoder.")
    p.add_argument("--use_coarse_mask_prompt", action="store_true",
                   help="Predict a dense coarse mask prompt from image embeddings and feed it as SAM mask prompt.")
    p.add_argument("--coarse_mask_prompt_hidden_channels", type=int, default=64,
                   help="Hidden channels for the coarse dense mask prompt head.")
    p.add_argument("--coarse_mask_prompt_weight", type=float, default=0.5,
                   help="Direct BCE+Dice supervision weight for the coarse dense mask prompt.")
    p.add_argument("--coarse_mask_prompt_pos_weight", type=float, default=10.0,
                   help="Positive-pixel BCE weight for coarse dense mask prompt supervision.")
    p.add_argument("--coarse_mask_prompt_bce_weight", type=float, default=1.0,
                   help="BCE term weight inside coarse dense mask prompt supervision.")
    p.add_argument("--coarse_mask_prompt_dice_weight", type=float, default=1.0,
                   help="Dice term weight inside coarse dense mask prompt supervision.")
    p.add_argument("--coarse_mask_prompt_merge_alpha", type=float, default=0.5,
                   help="Blend ratio when combining an existing dense prompt with the coarse mask prompt.")
    p.add_argument("--use_dynamic_sparse_prompt", action="store_true",
                   help="Predict dynamic sparse prompt tokens from image embeddings and append them to SAM sparse prompts.")
    p.add_argument("--dynamic_sparse_num_tokens", type=int, default=8,
                   help="Number of dynamic sparse prompt tokens.")
    p.add_argument("--dynamic_sparse_hidden_channels", type=int, default=64,
                   help="Hidden channels for the dynamic sparse prompt head.")
    p.add_argument("--dynamic_sparse_multilevel", action="store_true",
                   help="Use block 2/5/8/11 encoder features plus neck_out for dynamic sparse prompt tokens.")
    p.add_argument("--dynamic_sparse_loc_levels", type=int, default=3,
                   help="Number of projected levels used by the localization/targetness branch.")
    p.add_argument("--dynamic_sparse_temperature", type=float, default=1.0,
                   help="Softmax temperature for dynamic sparse token attention maps.")
    p.add_argument("--dynamic_sparse_init_scale", type=float, default=0.1,
                   help="Initial residual scale for dynamic sparse tokens.")
    p.add_argument("--dynamic_sparse_div_weight", type=float, default=0.0,
                   help="Weight for token diversity regularization.")
    p.add_argument("--dynamic_sparse_target_weight", type=float, default=0.0,
                   help="Weight for multilevel targetness-map supervision.")
    p.add_argument("--dynamic_sparse_target_pos_weight", type=float, default=20.0,
                   help="Positive-pixel BCE weight for multilevel targetness supervision.")
    p.add_argument("--dynamic_sparse_target_dice_weight", type=float, default=1.0,
                   help="Dice term weight inside multilevel targetness supervision.")
    p.add_argument("--dynamic_sparse_detach_image_embedding", action="store_true",
                   help="Detach image embeddings before feeding the dynamic sparse prompt head.")
    p.add_argument("--dynamic_sparse_train_head_only", action="store_true",
                   help="Freeze the SAM/HQ model and existing auxiliary heads; train only the dynamic sparse prompt head.")
    p.add_argument("--use_center_mask_decoder", action="store_true",
                   help="Enable center-mask joint decoder on the HQ branch.")
    p.add_argument("--center_gate_alpha", type=float, default=0.2,
                   help="Residual gate strength used by center_prob to modulate HQ mask features.")
    p.add_argument("--center_loss_weight", type=float, default=0.2,
                   help="Weight of the center heatmap supervision loss.")
    p.add_argument("--center_contain_weight", type=float, default=0.05,
                   help="Weight of the center-in-mask containment regularizer.")
    p.add_argument("--center_pos_weight", type=float, default=10.0,
                   help="Positive-pixel weight for center heatmap supervision.")
    p.add_argument("--center_gaussian_sigma", type=float, default=2.0,
                   help="Gaussian sigma used to build centroid heatmap targets.")
    # Proposed options (default OFF)
    p.add_argument("--use_point_loss", action="store_true",
                   help="Enable uncertainty-based point sampling BCE+Dice as auxiliary loss.")
    p.add_argument("--point_loss_points", type=int, default=4096,
                   help="Number of points for point loss.")
    p.add_argument("--point_loss_oversample", type=float, default=3.0,
                   help="Oversample ratio for uncertain point selection.")
    p.add_argument("--point_loss_importance", type=float, default=0.75,
                   help="Importance sample ratio for uncertain points.")
    p.add_argument("--point_loss_weight", type=float, default=0.3,
                   help="Weight for point loss term.")
    p.add_argument("--nwd_weight", type=float, default=0.0,
                   help="Weight for NWD loss (0 to disable).")
    p.add_argument("--nwd_constant", type=float, default=12.0,
                   help="Normalization constant for NWD loss.")
    p.add_argument("--boundary_prior_sampling", action="store_true",
                   help="Prefer sampling points near GT boundary.")
    p.add_argument("--boundary_ratio", type=float, default=0.5,
                   help="Fraction of pos/neg points sampled from boundary region.")
    p.add_argument("--prompt_mode", type=str, default="gt_points", choices=["gt_points", "assp_only"],
                   help="Prompt source for point prompts. 'assp_only' passes no valid point prompts and relies on text sparse prompts.")
    # LCA-Prompt (Local Contrast Attention-guided Prompt Generation)
    p.add_argument("--use_lca_prompt", action="store_true",
                   help="Enable LCA-Prompt: local contrast-guided auto prompt generation.")
    p.add_argument("--lca_scales", type=str, default="3,5,9",
                   help="Comma-separated LCM kernel sizes, e.g., '3,5,9'.")
    p.add_argument("--lca_top_k", type=int, default=5,
                   help="Number of prompt points to extract from contrast map.")
    p.add_argument("--lca_min_dist", type=int, default=8,
                   help="Minimum distance between extracted peaks.")
    p.add_argument("--lca_adaptive_ratio", type=float, default=0.5,
                   help="Adaptive point filtering: keep peaks with contrast >= max_peak * ratio (0-1). "
                        "Lower = more points, higher = fewer but more confident points.")
    p.add_argument("--lca_use_asg_bridge", action="store_true",
                   help="Enable ASG-LCA bridge to enhance contrast with ASG-filtered features.")
    p.add_argument("--lca_sup_weight", type=float, default=0.5,
                   help="Weight for LCA auxiliary supervision loss.")
    p.add_argument("--lca_sup_warmup", type=int, default=30,
                   help="LCA supervision warmup epochs (full weight after warmup).")
    p.add_argument("--lca_mix_ratio", type=float, default=0.5,
                   help="Probability of using LCA prompts vs GT-sampled prompts during training.")
    p.add_argument("--lca_mix_schedule", action="store_true",
                   help="Gradually increase LCA mix ratio from 0 to lca_mix_ratio over lca_sup_warmup epochs.")
    # Task Tokens (Learnable Prompt Tokens for IRSTD)
    p.add_argument("--use_task_tokens", action="store_true",
                   help="Enable learnable task tokens in PromptEncoder for IRSTD prior.")
    p.add_argument("--num_task_tokens", type=int, default=2,
                   help="Number of learnable task tokens (1-4 recommended).")
    p.add_argument("--task_token_init_scale", type=float, default=0.02,
                   help="Initialization scale for task tokens (small to avoid disrupting pretrain).")
    # Phase prompt generator (PGAP)
    p.add_argument("--use_pgap", action="store_true",
                   help="Use PhasePromptGenerator to auto-generate prompt points.")
    p.add_argument("--pgap_top_k", type=int, default=5)
    p.add_argument("--pgap_min_dist", type=int, default=10)
    p.add_argument("--pgap_saliency_thr", type=float, default=0.1)
    p.add_argument("--pgap_blur_kernel", type=int, default=5)
    p.add_argument("--pgap_blur_sigma", type=float, default=1.0)
    p.add_argument("--pgap_border_width", type=int, default=12)
    p.add_argument("--pgap_no_window", action="store_true")
    p.add_argument("--pgap_no_dynamic_thr", action="store_true")
    p.add_argument("--pgap_dyn_quantile", type=float, default=0.9)
    p.add_argument("--pgap_dyn_mode", type=str, default="max", choices=["max", "replace"])
    p.add_argument("--pgap_no_dynamic_topk", action="store_true")
    p.add_argument("--pgap_min_top_k", type=int, default=1)
    p.add_argument("--pgap_use_dct", action="store_true")
    p.add_argument("--pgap_text_fuse_internal", action="store_true",
                   help="Fuse text dense prior into PGAP saliency inside PGAP before point extraction.")
    p.add_argument("--pgap_text_fuse_weight", type=float, default=0.5,
                   help="Fusion strength for internal PGAP-text saliency fusion.")
    p.add_argument("--pgap_text_fuse_mode", type=str, default="mul", choices=["mul", "add"],
                   help="Internal PGAP-text saliency fusion mode.")
    p.add_argument("--pgap_text_prior_only", action="store_true",
                   help="Pure PGAP-text mode: text is only used to build PGAP internal fused saliency, not injected into SAM (no FiLM/sparse/dense prompt injection).")
    p.add_argument("--use_feature_mod", action="store_true",
                   help="Use PGAP saliency to modulate image embeddings.")
    p.add_argument("--pgap_label_by_gt", action="store_true",
                   help="Use GT to relabel PGAP points: inside=pos, outside=neg.")
    p.add_argument("--pgap_min_pos", type=int, default=1)
    p.add_argument("--pgap_max_neg", type=int, default=2)
    p.add_argument("--pgap_two_stage", action="store_true",
                   help="Two-stage prompting in validation: pos first, then add negatives outside coarse mask.")
    p.add_argument("--pgap_stage1_top_k", type=int, default=1)
    p.add_argument("--pgap_stage1_thr", type=float, default=0.5)
    p.add_argument("--pgap_stage2_neg", type=int, default=2)
    p.add_argument("--use_mask_prompt", action="store_true",
                   help="Use PGAP saliency map as dense mask prompt.")
    p.add_argument("--val_thr_search", action="store_true",
                   help="Enable validation threshold grid search.")
    p.add_argument("--val_thr_min", type=float, default=0.35)
    p.add_argument("--val_thr_max", type=float, default=0.55)
    p.add_argument("--val_thr_step", type=float, default=0.05)
    p.add_argument("--pd_fa_dist", type=int, default=3,
                   help="Distance threshold for PD/FA metrics (in pixels).")
    p.add_argument("--log_file", type=str, default=None,
                   help="Path to log file (default: <out_dir>/log.txt).")
    p.add_argument("--mask_suffix", type=str, default="",
                   help="Optional suffix for mask filenames before extension, e.g. '_pixels0'.")
    # SCTransNet-style preprocessing options (与备份版兼容)
    p.add_argument("--sctransnet_preproc", action="store_true",
                   help="Use SCTransNet-style preprocessing: 16-bit grayscale, dataset normalization, random crop, enhanced augmentation.")
    p.add_argument("--sc_use_noise", action="store_true",
                   help="Add Gaussian noise in SCTransNet augmentation.")
    p.add_argument("--sc_use_gamma", action="store_true",
                   help="Apply random gamma correction in SCTransNet augmentation.")
    p.add_argument("--sc_pos_prob", type=float, default=0.5,
                   help="Probability of cropping region containing target in SCTransNet mode.")
    p.add_argument("--sc_dataset_name", type=str, default=None,
                   help="Dataset name for SCTransNet normalization (auto-detected from data_root if not set).")
    p.add_argument("--sc_eval_crop", type=str, default="random", choices=["random", "center", "resize"],
                   help="SCTransNet validation crop mode. random preserves legacy behavior; center/resize are deterministic.")
    # MLLM text prompt (pre-computed CLIP features)
    p.add_argument("--use_mllm_prompt", action="store_true",
                   help="Enable MLLM-based text prompting with pre-computed CLIP features.")
    p.add_argument("--mllm_features_path", type=str, default="mllm_clip_features.pt",
                   help="Path to pre-computed CLIP text features file (.pt).")
    p.add_argument("--mllm_text_dim", type=int, default=512,
                   help="Dimension of CLIP text features (512 for ViT-B/32).")
    p.add_argument("--use_tassg", action="store_true",
                   help="Enable Targetness-aware Semantic Slot Generator.")
    p.add_argument("--semantic_source", type=str, default="teacher", choices=["teacher", "student", "none"],
                   help="Semantic source for text prompt/fusion modules.")
    p.add_argument("--tassg_num_slots", type=int, default=8)
    p.add_argument("--tassg_hidden_dim", type=int, default=256)
    p.add_argument("--tassg_num_heads", type=int, default=4)
    p.add_argument("--tassg_dropout", type=float, default=0.0)
    p.add_argument("--tassg_img_dim", type=int, default=256)
    p.add_argument("--tassg_text_dim", type=int, default=None,
                   help="Text feature dimension for TASSG. If omitted, uses --mllm_text_dim.")
    p.add_argument("--tassg_two_pass_backbone", action="store_true",
                   help="Run student TASSG once, then rerun the backbone with student tokens for CBGA.")
    p.add_argument("--student_only_start_epoch", type=int, default=-1,
                   help="If >=0, switch semantic_source teacher to student from this epoch onward.")
    p.add_argument("--lambda_tassg_global", type=float, default=0.1)
    p.add_argument("--lambda_tassg_token", type=float, default=0.1)
    p.add_argument("--lambda_tassg_prompt", type=float, default=0.5)
    p.add_argument("--lambda_tassg_targetness", type=float, default=0.2)
    p.add_argument("--disable_text_conditioner", action="store_true",
                   help="Disable FiLM-style text conditioning on image embeddings while keeping other text modules.")
    p.add_argument("--use_text_sparse_prompt", action="store_true",
                   help="Project selected text feature source into extra sparse prompt token(s).")
    p.add_argument("--text_sparse_num_tokens", type=int, default=1,
                   help="Number of text sparse prompt tokens.")
    p.add_argument("--text_sparse_init_scale", type=float, default=0.02,
                   help="Init scale for text sparse base tokens.")
    p.add_argument("--text_sparse_prompt_source", type=str, default="fused_tokens",
                   choices=["raw_global", "fused_global", "fused_tokens"],
                   help="Sparse prompt source: raw_global=original clip_text_feat, fused_global=post-fusion pooled text feature, fused_tokens=post-fusion token features (fallback to fused_global).")
    p.add_argument("--text_sparse_raw_global_gate", action="store_true",
                   help="Enable a small sigmoid gate on the enhanced raw_global sparse prompt delta path.")
    p.add_argument("--text_sparse_raw_global_gate_init_bias", type=float, default=-2.0,
                   help="Init bias for raw_global sparse prompt gate; more negative means weaker initial injection.")
    p.add_argument("--use_text_dense_prompt", action="store_true",
                   help="Generate a text-guided dense mask prompt from image embeddings + CLIP text feature.")
    p.add_argument("--text_dense_hidden_dim", type=int, default=128,
                   help="Hidden channels for text-guided dense mask prompt generator.")
    p.add_argument("--text_dense_prompt_type", type=str, default="global",
                   choices=["global", "token_xattn"],
                   help="Dense text prompt variant: global (v1) or token_xattn (v2 token-level cross-attn).")
    p.add_argument("--text_dense_num_heads", type=int, default=4,
                   help="Number of heads for token_xattn dense prompt variant.")
    p.add_argument("--text_dense_prompt_merge_alpha", type=float, default=0.5,
                   help="Blend ratio when combining PGAP mask prompt and text dense mask prompt.")
    p.add_argument("--text_dense_prompt_scale", type=float, default=1.0,
                   help="Scale factor applied to generated text dense mask prompt before merging.")
    p.add_argument("--use_bifusion_adapter", action="store_true",
                   help="Enable lightweight bidirectional text-vision fusion adapter at two levels (interms + img_emb).")
    p.add_argument("--bifusion_hidden_dim", type=int, default=128,
                   help="Hidden dim for BiFusion attention space.")
    p.add_argument("--bifusion_num_heads", type=int, default=4,
                   help="Number of heads for BiFusion cross-attention.")
    p.add_argument("--bifusion_interms_dim", type=int, default=192,
                   help="Fallback interms channel dim when auto-detection fails.")
    p.add_argument("--bifusion_disable_interms_level", action="store_true",
                   help="Disable interms-level fusion; keep img_emb level only.")
    p.add_argument("--bifusion_img_res_scale", type=float, default=1.0,
                   help="Residual scale for img_emb update in BiFusion.")
    p.add_argument("--bifusion_interms_res_scale", type=float, default=1.0,
                   help="Residual scale for interms update in BiFusion.")
    p.add_argument("--bifusion_text_res_scale", type=float, default=1.0,
                   help="Residual scale for text token update in BiFusion.")
    p.add_argument("--use_bifusion_backbone_blocks", action="store_true",
                   help="Enable bidirectional text-vision fusion inside image encoder blocks.")
    p.add_argument("--use_gated_bifusion_backbone_blocks", action="store_true",
                   help="Enable gated backbone BiFusion for ablation; keeps block-level bidirectional fusion but gates text/vision updates.")
    p.add_argument("--bifusion_block_apply_every", type=int, default=1,
                   help="Apply backbone BiFusion every K encoder blocks.")
    p.add_argument("--bifusion_block_vision_res_scale", type=float, default=1.0,
                   help="Residual scale for vision-token update in backbone BiFusion.")
    p.add_argument("--bifusion_block_text_res_scale", type=float, default=1.0,
                   help="Residual scale for text-token update in backbone BiFusion.")
    p.add_argument("--bifusion_gate_hidden_dim", type=int, default=0,
                   help="Hidden dim for gated backbone BiFusion gates (<=0 uses hidden_dim//4).")
    p.add_argument("--bifusion_gate_init_bias", type=float, default=-2.0,
                   help="Initial bias for gated backbone BiFusion sigmoid gates (negative keeps gates conservative at start).")
    p.add_argument("--bifusion_gate_delta_only", action="store_true",
                   help="Project only gated cross-modal deltas in gated backbone BiFusion; preserves identity when gates close and avoids legacy unimodal residual drift.")
    # Freeze/unfreeze strategy configs
    p.add_argument("--freeze_encoder_epochs", type=int, default=-1,
                   help="Freeze image encoder for N epochs first (<=0 to use epochs//4).")
    p.add_argument("--train_prompt_encoder_during_freeze", action="store_true",
                   help="Whether to train prompt encoder during initial freeze stage (default: False).")
    p.add_argument("--freeze_maskdecoder_to_hq", action="store_true", default=True,
                   help="Only train HQ-specific params in MaskDecoder during initial freeze stage.")
    p.add_argument("--unfreeze_all_when_encoder", action="store_true", default=True,
                   help="When unfreezing encoder, also unfreeze full mask decoder and prompt encoder.")
    args = p.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.exp_name is None:
        base = [f"model-{args.model}", f"bs-{args.batch_size}"]
        auto_name = "_".join(base)
        run_dir = os.path.join(args.out_dir, f"{auto_name}_{ts}")
    else:
        run_dir = os.path.join(args.out_dir, f"{args.exp_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    try:
        with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    args.out_dir = run_dir
    if args.log_file is None:
        args.log_file = os.path.join(run_dir, "log.txt")
    log_line(f"Run directory: {args.out_dir}", args.log_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hldf_layers = _parse_int_list_arg(getattr(args, "hldf_layers", "0,2,5"), default=[0, 2, 5])
    amgd_branch_design = str(getattr(args, "amgd_branch_design", "legacy")).lower()
    amgd_detail_layer_arg = getattr(args, "amgd_detail_layer", None)
    amgd_structure_layer_arg = getattr(args, "amgd_structure_layer", None)
    amgd_background_layer_arg = getattr(args, "amgd_background_layer", None)
    amgd_detail_layer = max(0, int(getattr(args, "amgd_interm_layer", 0) if amgd_detail_layer_arg is None else amgd_detail_layer_arg))
    amgd_structure_layer = max(0, int(getattr(args, "amgd_interm_layer", 0) if amgd_structure_layer_arg is None else amgd_structure_layer_arg))
    amgd_background_layer = max(0, int(getattr(args, "amgd_interm_layer", 0) if amgd_background_layer_arg is None else amgd_background_layer_arg))
    amgd_branch_layers = sorted({amgd_detail_layer, amgd_structure_layer, amgd_background_layer})
    asg_block_indices = _parse_int_list_raw_arg(getattr(args, "asg_block_indices", "11"), default=[11])
    asg_block_strengths = _parse_float_list_arg(getattr(args, "asg_block_strengths", "0.25"), default=[0.25])
    if getattr(args, "use_hldf", False):
        if getattr(args, "use_amgd", False):
            raise ValueError("--use_hldf and --use_amgd are mutually exclusive.")
        if getattr(args, "use_ms_fusion", False):
            raise ValueError("--use_hldf cannot be combined with --use_ms_fusion in the current implementation.")
        if len(hldf_layers) < 2:
            raise ValueError("--hldf_layers must specify at least two ViT block indices.")
    if getattr(args, "use_asg_hq", False) and args.asg_loc in ("encoder", "both") and args.asg_encoder_mode == "block_only":
        if len(asg_block_indices) == 0:
            raise ValueError("--asg_block_indices must specify at least one ViT block index in block_only mode.")
        if len(asg_block_indices) != len(asg_block_strengths):
            raise ValueError("--asg_block_indices and --asg_block_strengths must have the same length.")
        if len(set(asg_block_indices)) != len(asg_block_indices):
            raise ValueError("--asg_block_indices must not contain duplicates in block_only mode.")

    # Data
    train_loader = make_loader(
        args.data_root,
        args.train_txt,
        batch_size=args.batch_size,
        size=args.size,
        augment=True,
        keep_ratio_pad=args.keep_ratio_pad,
        workers=args.workers,
        shuffle=True,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=args.sctransnet_preproc,
        sc_use_noise=args.sc_use_noise,
        sc_use_gamma=args.sc_use_gamma,
        sc_pos_prob=args.sc_pos_prob,
        sc_dataset_name=args.sc_dataset_name,
        sc_eval_crop="random",
        mllm_features_path=getattr(args, "mllm_features_path", None)
        if (getattr(args, "use_mllm_prompt", False) or getattr(args, "use_tassg", False))
        else None,
    )
    val_loader = make_loader(
        args.data_root,
        args.val_txt,
        batch_size=max(1, args.batch_size // 2),
        size=args.size,
        augment=False,
        keep_ratio_pad=args.keep_ratio_pad,
        workers=args.workers,
        shuffle=False,
        mask_suffix=args.mask_suffix,
        sctransnet_preproc=args.sctransnet_preproc,
        sc_use_noise=False,  # No noise augmentation for validation
        sc_use_gamma=False,  # No gamma augmentation for validation
        sc_pos_prob=args.sc_pos_prob,
        sc_dataset_name=args.sc_dataset_name,
        sc_eval_crop=args.sc_eval_crop,
        mllm_features_path=getattr(args, "mllm_features_path", None)
        if (getattr(args, "use_mllm_prompt", False) or getattr(args, "use_tassg", False))
        else None,
    )

    # Model
    model = build_efficient_sam_hq(
        encoder_patch_embed_dim=192 if args.model == "vitt" else 384,
        encoder_num_heads=3 if args.model == "vitt" else 6,
        init_from_baseline=args.init_from_baseline,
        use_adapter=args.use_fs_adapter,
        use_ms_fusion=args.use_ms_fusion,
        use_detail_enhancer=args.use_detail_enhancer,
        early_exit_layer=getattr(args, "early_exit_layer", None),
        use_amgd=getattr(args, "use_amgd", False),
        use_dog_amgd=getattr(args, "use_dog_amgd", False),
        amgd_routing=str(getattr(args, "amgd_routing", "prompt")).lower(),
        amgd_interm_layer=max(0, int(getattr(args, "amgd_interm_layer", 0))),
        amgd_branch_design=amgd_branch_design,
        amgd_detail_layer=amgd_detail_layer,
        amgd_structure_layer=amgd_structure_layer,
        amgd_background_layer=amgd_background_layer,
        dog_amgd_mode=str(getattr(args, "dog_amgd_mode", "legacy")).lower(),
        dog_amgd_strength=max(0.0, float(getattr(args, "dog_amgd_strength", 0.25))),
        use_hldf=getattr(args, "use_hldf", False),
        hldf_layers=hldf_layers,
        hldf_hidden_dim=max(8, int(getattr(args, "hldf_hidden_dim", 96))),
        hldf_use_hq_router=bool(getattr(args, "hldf_use_hq_router", True)),
        hldf_router_temp=max(1e-3, float(getattr(args, "hldf_router_temp", 1.0))),
        use_center_mask_decoder=bool(getattr(args, "use_center_mask_decoder", False)),
        center_gate_alpha=float(getattr(args, "center_gate_alpha", 0.2)),
        return_encoder_multi_scale=bool(getattr(args, "dynamic_sparse_multilevel", False)),
    )
    if getattr(args, "use_asg_hq", False) and args.asg_loc in ("encoder", "both") and args.asg_encoder_mode == "block_only":
        encoder_depth = len(getattr(model.image_encoder, "blocks", []))
        invalid_block_indices = [idx for idx in asg_block_indices if idx < 0 or idx >= encoder_depth]
        if invalid_block_indices:
            raise ValueError(
                f"--asg_block_indices contains out-of-range indices {invalid_block_indices}; valid range is [0, {max(0, encoder_depth - 1)}]."
            )
    if getattr(args, "use_hldf", False):
        log_line(
            "HLDF enabled: "
            f"layers={tuple(hldf_layers)}, "
            f"hidden_dim={max(8, int(getattr(args, 'hldf_hidden_dim', 96)))}, "
            f"hq_router={bool(getattr(args, 'hldf_use_hq_router', True))}, "
            f"router_temp={max(1e-3, float(getattr(args, 'hldf_router_temp', 1.0))):.3f}",
            args.log_file,
        )
    if getattr(args, "use_dog_amgd", False) and not getattr(args, "use_amgd", False):
        log_line("[warn] --use_dog_amgd has no effect unless --use_amgd is enabled.", args.log_file)
    if getattr(args, "use_amgd", False):
        dog_mode = str(getattr(args, "dog_amgd_mode", "legacy")).lower()
        dog_strength = max(0.0, float(getattr(args, "dog_amgd_strength", 0.25)))
        amgd_routing = str(getattr(args, "amgd_routing", "prompt")).lower()
        if getattr(args, "use_dog_amgd", False) and dog_mode == "legacy":
            log_line("[info] --dog_amgd_strength is ignored when --dog_amgd_mode legacy.", args.log_file)
        dog_strength_desc = "n/a"
        if getattr(args, "use_dog_amgd", False) and dog_mode == "residual":
            dog_strength_desc = f"{dog_strength:.3f}"
        if amgd_branch_design == "dsb_v1":
            log_line(
                "AMGD enabled: "
                f"branch_design=dsb_v1, "
                f"detail_layer={amgd_detail_layer}, "
                f"structure_layer={amgd_structure_layer}, "
                f"background_layer={amgd_background_layer}, "
                f"detail_branch_indices={tuple(amgd_branch_layers)}, "
                f"routing={amgd_routing}, "
                f"dog_enabled={bool(getattr(args, 'use_dog_amgd', False))}, "
                f"dog_mode={dog_mode}, "
                f"dog_strength={dog_strength_desc}",
                args.log_file,
            )
        else:
            log_line(
                "AMGD enabled: "
                f"branch_design=legacy, "
                f"detail_source_block={max(0, int(getattr(args, 'amgd_interm_layer', 0)))}, "
                f"routing={amgd_routing}, "
                f"dog_enabled={bool(getattr(args, 'use_dog_amgd', False))}, "
                f"dog_mode={dog_mode}, "
                f"dog_strength={dog_strength_desc}",
                args.log_file,
            )
    if getattr(args, "use_asg_hq", False):
        if args.asg_loc in ("encoder", "both") and args.asg_encoder_mode == "block_only":
            log_line(
                "ASG config: "
                f"loc={args.asg_loc}, enc_mode=block_only, "
                f"enc_blocks={tuple(asg_block_indices)}, "
                f"enc_block_strengths={tuple(round(float(v), 4) for v in asg_block_strengths)}, "
                f"enc_strength=ignored, "
                f"dec_strength={float(args.asg_strength_dec):.3f}",
                args.log_file,
            )
        else:
            log_line(
                "ASG config: "
                f"loc={args.asg_loc}, enc_mode=neck, "
                f"enc_strength={float(args.asg_strength_enc):.3f}, "
                f"dec_strength={float(args.asg_strength_dec):.3f}",
                args.log_file,
            )
    if getattr(args, "use_center_mask_decoder", False):
        log_line(
            "Center-mask decoder enabled: "
            f"gate_alpha={float(getattr(args, 'center_gate_alpha', 0.2)):.3f}, "
            f"center_loss_weight={float(getattr(args, 'center_loss_weight', 0.2)):.3f}, "
            f"center_contain_weight={float(getattr(args, 'center_contain_weight', 0.05)):.3f}, "
            f"center_pos_weight={float(getattr(args, 'center_pos_weight', 10.0)):.3f}, "
            f"center_gaussian_sigma={float(getattr(args, 'center_gaussian_sigma', 2.0)):.3f}",
            args.log_file,
        )
    # Attach frequency gates if requested
    if args.use_asg_hq and args.use_radial_gate_hq:
        log_line("[warn] Both ASG and RadialFreqGate are enabled; using ASG and skipping RadialFreqGate.", args.log_file)
    if args.use_asg_hq:
        try:
            from efficient_sam.asg import AnisotropicSpectralGating, AnisotropicSpectralGating2

            class _ASGDelta(nn.Module):
                # Convert ASG residual output to a delta for x + gate(x) usage.
                def __init__(self, asg):
                    super().__init__()
                    self.asg = asg

                def forward(self, x):
                    return self.asg(x) - x

            if args.asg_variant == "asg2":
                asg_cls = AnisotropicSpectralGating2
                asg_kwargs = {
                    "r_bins": args.asg_radial_bins,
                    "theta_bins": args.asg_angular_bins,
                }
            else:
                asg_cls = AnisotropicSpectralGating
                asg_kwargs = {
                    "num_radial_bins": args.asg_radial_bins,
                    "num_angular_bins": args.asg_angular_bins,
                }

            enc_hw = getattr(model.image_encoder, "image_embedding_size", 64)
            model.image_encoder.configure_block_asg({}, {}, mode="neck")
            model.image_encoder.radial_gate = None
            if args.asg_loc in ("encoder", "both"):
                if args.asg_encoder_mode == "block_only":
                    try:
                        dim_block = model.image_encoder.patch_embed.proj.out_channels
                    except Exception:
                        dim_block = getattr(model.image_encoder.blocks[0].norm1, "normalized_shape", [192])[0]
                    block_modules = {}
                    block_strength_map = {}
                    for block_idx, block_strength in zip(asg_block_indices, asg_block_strengths):
                        block_modules[int(block_idx)] = _ASGDelta(asg_cls(dim_block, enc_hw, enc_hw, **asg_kwargs))
                        block_strength_map[int(block_idx)] = float(block_strength)
                    model.image_encoder.configure_block_asg(
                        block_modules,
                        block_strength_map,
                        mode="block_only",
                    )
                    if abs(float(args.asg_strength_enc)) > 1e-8:
                        log_line("[info] --asg_strength_enc is ignored when --asg_encoder_mode block_only.", args.log_file)
                else:
                    try:
                        dim_enc = model.image_encoder.neck[0].out_channels
                    except Exception:
                        dim_enc = 256
                    asg_enc = asg_cls(dim_enc, enc_hw, enc_hw, **asg_kwargs)
                    model.image_encoder.radial_gate = _ASGDelta(asg_enc)
                    model.image_encoder.rgate_strength = float(args.asg_strength_enc)
            if args.asg_loc in ("decoder", "both"):
                # hq_features channels = transformer_dim // 8, spatial size = 4x encoder grid
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                hq_hw = enc_hw * 4
                asg_dec = asg_cls(c_dec, hq_hw, hq_hw, **asg_kwargs)
                model.mask_decoder.radial_gate = _ASGDelta(asg_dec)
                model.mask_decoder.rgate_strength_dec = float(args.asg_strength_dec)
        except Exception as e:
            log_line(f"[warn] Failed to attach ASG: {e}", args.log_file)
    elif args.use_radial_gate_hq:
        try:
            from efficient_sam.freq_modules import RadialFreqGate
            if args.rgate_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                model.image_encoder.radial_gate = RadialFreqGate(
                    dim_enc,
                    patch_size=args.freq_patch_size_hq,
                    num_bins=args.radial_bins_hq,
                    channel_shared=args.radial_channel_shared_hq,
                    edge_boost=args.rgate_edge_boost,
                    high_freq_threshold=args.rgate_high_freq_thresh,
                )
                model.image_encoder.rgate_strength = float(args.rgate_strength_enc)
            if args.rgate_loc in ("decoder", "both"):
                # hq_features channels = transformer_dim // 8
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                model.mask_decoder.radial_gate = RadialFreqGate(
                    c_dec,
                    patch_size=args.freq_patch_size_hq,
                    num_bins=args.radial_bins_hq,
                    channel_shared=args.radial_channel_shared_hq,
                    edge_boost=args.rgate_edge_boost,
                    high_freq_threshold=args.rgate_high_freq_thresh,
                )
                model.mask_decoder.rgate_strength_dec = float(args.rgate_strength_dec)
        except Exception as e:
            log_line(f"[warn] Failed to attach RadialFreqGate: {e}", args.log_file)
    # Attach AFD (Adaptive Frequency Decomposition) if requested
    if args.use_afd_hq:
        try:
            from efficient_sam.freq_modules import AdaptiveFrequencyDecomposition
            if args.afd_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                afd_enc = AdaptiveFrequencyDecomposition(
                    dim=dim_enc,
                    patch_size=args.afd_patch_size,
                    num_cutoff_bins=args.afd_num_bins,
                    low_enhance_ratio=args.afd_low_ratio,
                    high_enhance_ratio=args.afd_high_ratio,
                    learnable_gains=not getattr(args, 'afd_fixed_gains', False),
                    channel_wise_gains=getattr(args, 'afd_channel_wise', False),
                )
                model.image_encoder.afd_gate = afd_enc
                model.image_encoder.afd_strength = float(args.afd_strength_enc)
                log_line(f"Attached AFD to encoder (dim={dim_enc}, patch={args.afd_patch_size}, bins={args.afd_num_bins})", args.log_file)
            if args.afd_loc in ("decoder", "both"):
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                afd_dec = AdaptiveFrequencyDecomposition(
                    dim=c_dec,
                    patch_size=args.afd_patch_size,
                    num_cutoff_bins=args.afd_num_bins,
                    low_enhance_ratio=args.afd_low_ratio,
                    high_enhance_ratio=args.afd_high_ratio,
                    learnable_gains=not getattr(args, 'afd_fixed_gains', False),
                    channel_wise_gains=getattr(args, 'afd_channel_wise', False),
                )
                model.mask_decoder.afd_gate = afd_dec
                model.mask_decoder.afd_strength_dec = float(args.afd_strength_dec)
                log_line(f"Attached AFD to decoder (dim={c_dec}, patch={args.afd_patch_size}, bins={args.afd_num_bins})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach AFD: {e}", args.log_file)
    # Attach MSFE (Multi-Scale Frequency Enhancement) if requested
    if getattr(args, "use_msfe_hq", False):
        try:
            from efficient_sam.freq_modules import MultiScaleFrequencyEnhancement
            # Parse patch sizes from comma-separated string
            patch_sizes = tuple(int(x) for x in args.msfe_patch_sizes.split(","))
            if args.msfe_loc in ("encoder", "both"):
                try:
                    dim_enc = model.image_encoder.neck[0].out_channels
                except Exception:
                    dim_enc = 256
                msfe_enc = MultiScaleFrequencyEnhancement(
                    dim=dim_enc,
                    patch_sizes=patch_sizes,
                    num_radial_bins=args.msfe_num_bins,
                    fusion_method=args.msfe_fusion,
                )
                model.image_encoder.msfe_gate = msfe_enc
                model.image_encoder.msfe_strength = float(args.msfe_strength_enc)
                log_line(f"Attached MSFE to encoder (dim={dim_enc}, patches={patch_sizes}, bins={args.msfe_num_bins}, fusion={args.msfe_fusion})", args.log_file)
            if args.msfe_loc in ("decoder", "both"):
                c_dec = getattr(model.mask_decoder, "transformer_dim", 256) // 8
                msfe_dec = MultiScaleFrequencyEnhancement(
                    dim=c_dec,
                    patch_sizes=patch_sizes,
                    num_radial_bins=args.msfe_num_bins,
                    fusion_method=args.msfe_fusion,
                )
                model.mask_decoder.msfe_gate = msfe_dec
                model.mask_decoder.msfe_strength_dec = float(args.msfe_strength_dec)
                log_line(f"Attached MSFE to decoder (dim={c_dec}, patches={patch_sizes}, bins={args.msfe_num_bins}, fusion={args.msfe_fusion})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach MSFE: {e}", args.log_file)
    # Attach Task Tokens (Learnable Prompt Tokens) if requested
    if getattr(args, "use_task_tokens", False):
        try:
            embed_dim = model.prompt_encoder.embed_dim
            num_tokens = args.num_task_tokens
            init_scale = args.task_token_init_scale
            # Create learnable task tokens and attach to prompt_encoder
            task_tokens = torch.nn.Parameter(
                torch.randn(1, num_tokens, embed_dim) * init_scale
            )
            model.prompt_encoder.task_tokens = task_tokens
            log_line(f"Attached Task Tokens to PromptEncoder (num_tokens={num_tokens}, embed_dim={embed_dim}, init_scale={init_scale})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to attach Task Tokens: {e}", args.log_file)
    model.to(device)

    pgap = None
    if args.use_pgap:
        from efficient_sam.PGAP import PhasePromptGenerator
        pgap = PhasePromptGenerator(
            top_k=args.pgap_top_k,
            input_size=(args.size, args.size),
            min_dist=args.pgap_min_dist,
            saliency_thr=args.pgap_saliency_thr,
            blur_kernel_size=args.pgap_blur_kernel,
            blur_sigma=args.pgap_blur_sigma,
            use_window=not args.pgap_no_window,
            border_width=args.pgap_border_width,
            dynamic_thr=not args.pgap_no_dynamic_thr,
            dynamic_thr_quantile=args.pgap_dyn_quantile,
            dynamic_thr_mode=args.pgap_dyn_mode,
            dynamic_top_k=not args.pgap_no_dynamic_topk,
            min_top_k=args.pgap_min_top_k,
            use_dct=args.pgap_use_dct,
        ).to(device)
        pgap.eval()

    # Initialize LCA-Prompt if requested
    lca_prompt = None
    if getattr(args, "use_lca_prompt", False) and not getattr(args, "use_self_prompting", False):
        try:
            from efficient_sam.lca_prompt import LCAPromptGenerator
            lca_scales = tuple(int(x) for x in args.lca_scales.split(","))
            try:
                neck_dim = model.image_encoder.neck[0].out_channels
            except Exception:
                neck_dim = 256
            lca_prompt = LCAPromptGenerator(
                scales=lca_scales,
                top_k=args.lca_top_k,
                min_dist=args.lca_min_dist,
                adaptive_ratio=args.lca_adaptive_ratio,
                use_asg_bridge=args.lca_use_asg_bridge,
                neck_dim=neck_dim,
            ).to(device)
            n_lca = sum(p.numel() for p in lca_prompt.parameters())
            log_line(
                f"LCA-Prompt enabled: scales={lca_scales}, top_k={args.lca_top_k}, "
                f"min_dist={args.lca_min_dist}, asg_bridge={args.lca_use_asg_bridge}, "
                f"sup_weight={args.lca_sup_weight}, params={n_lca}",
                args.log_file,
            )
        except Exception as e:
            log_line(f"[warn] Failed to initialize LCA-Prompt: {e}", args.log_file)
    elif getattr(args, "use_lca_prompt", False) and getattr(args, "use_self_prompting", False):
        log_line("[warn] Ignoring --use_lca_prompt because --use_self_prompting is enabled.", args.log_file)

    # Initialize FAB Loss (Frequency-Aware Boundary Loss) if requested
    fab_criterion = None
    if getattr(args, "use_fab_loss", False):
        try:
            from efficient_sam.fab_loss import build_fab_loss
            fab_criterion = build_fab_loss(
                num_bins=args.fab_num_bins,
                boundary_width=args.fab_boundary_width,
                high_freq_weight=args.fab_high_freq_weight,
                use_multiscale=True,
            ).to(device)
            log_line(f"Initialized FAB Loss (bins={args.fab_num_bins}, boundary_width={args.fab_boundary_width}, high_freq_weight={args.fab_high_freq_weight})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to initialize FAB Loss: {e}", args.log_file)

    # Initialize SCR Loss (Signal-to-Clutter Ratio Loss) if requested
    scr_criterion = None
    if getattr(args, "use_scr_loss", False):
        try:
            from efficient_sam.scr_loss import build_scr_loss
            scr_criterion = build_scr_loss(
                annular_inner_k=args.scr_inner_k,
                annular_outer_k=args.scr_outer_k,
            ).to(device)
            log_line(f"Initialized SCR Loss (inner_k={args.scr_inner_k}, outer_k={args.scr_outer_k}, weight={args.scr_weight})", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to initialize SCR Loss: {e}", args.log_file)

    # Log Prompt-Robust Loss config
    if getattr(args, "use_prompt_robust_loss", False):
        log_line(f"Enabled Prompt-Robust Consistency Loss (weight={args.prompt_robust_weight}, perturb_std={args.prompt_robust_perturb_std})", args.log_file)

    # Stage-1: freeze image encoder
    for p_ in model.image_encoder.parameters():
        p_.requires_grad = False
    if args.use_fs_adapter:
        try:
            from efficient_sam.efficient_sam_encoder_hq import FSAdapter
            fs_tensors = 0
            for m in model.image_encoder.modules():
                if isinstance(m, FSAdapter):
                    for p in m.parameters():
                        p.requires_grad = True
                        fs_tensors += 1
            if fs_tensors > 0:
                log_line("Enabled FSAdapter params during encoder freeze.", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to enable FSAdapter during freeze: {e}", args.log_file)

    if (
        getattr(args, "use_asg_hq", False)
        and args.asg_loc in ("encoder", "both")
        and getattr(model.image_encoder, "asg_encoder_mode", "neck") == "block_only"
    ):
        block_asg_modules = getattr(model.image_encoder, "block_asg_modules", None)
        if block_asg_modules is not None and len(block_asg_modules) > 0:
            enabled_blocks = []
            for key, mod in block_asg_modules.items():
                for p in mod.parameters():
                    p.requires_grad = True
                enabled_blocks.append(str(key))
            log_line(
                "Enabled block-only encoder ASG during freeze "
                f"(head-lr path): blocks={','.join(enabled_blocks)}",
                args.log_file,
            )
    if (
        getattr(args, "use_asg_hq", False)
        and args.asg_loc in ("encoder", "both")
        and getattr(model.image_encoder, "asg_encoder_mode", "neck") != "block_only"
        and getattr(model.image_encoder, "radial_gate", None) is not None
    ):
        for p in model.image_encoder.radial_gate.parameters():
            p.requires_grad = True
        log_line("Enabled encoder neck_out ASG/radial_gate during freeze (head-lr path).", args.log_file)

    # Configure which head params are trainable initially
    # Follow HQ-SAM: only train HQ-specific layers by default
    def mark_maskdecoder_stage1(md):
        for n, p in md.named_parameters():
            p.requires_grad = False
        allow_keys = [
            "hf_token", "hf_mlp", "compress_vit_feat", "embedding_encoder", "embedding_maskfeature",
            "center_head",
            # AMGD components need to be trained from scratch initially
            "amgd_fine", "amgd_mid", "amgd_coarse", "amgd_router",
            "amgd_shared_stem", "amgd_detail_head", "amgd_structure_head", "amgd_background_head",
            # HLDF replaces AMGD as the decoder-side HQ detail branch.
            "hldf",
            # Decoder-side ASG / radial gate should start learning from epoch 1.
            "radial_gate",
        ]
        for key in allow_keys:
            mod = getattr(md, key, None)
            if mod is None:
                continue
            for p in mod.parameters():
                p.requires_grad = True

    if args.freeze_maskdecoder_to_hq:
        mark_maskdecoder_stage1(model.mask_decoder)
        if getattr(model.mask_decoder, "radial_gate", None) is not None:
            log_line("Enabled decoder ASG/radial_gate during freeze.", args.log_file)
    else:
        for p_ in model.mask_decoder.parameters():
            p_.requires_grad = True

    # Prompt encoder trainable during freeze?
    for p_ in model.prompt_encoder.parameters():
        p_.requires_grad = bool(args.train_prompt_encoder_during_freeze)

    # Collect params for optimizer
    head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
    if (
        getattr(args, "use_asg_hq", False)
        and args.asg_loc in ("encoder", "both")
        and getattr(model.image_encoder, "asg_encoder_mode", "neck") != "block_only"
        and getattr(model.image_encoder, "radial_gate", None) is not None
    ):
        head_params += [p for p in model.image_encoder.radial_gate.parameters() if p.requires_grad]
    if (
        getattr(args, "use_asg_hq", False)
        and args.asg_loc in ("encoder", "both")
        and getattr(model.image_encoder, "asg_encoder_mode", "neck") == "block_only"
    ):
        block_asg_modules = getattr(model.image_encoder, "block_asg_modules", None)
        if block_asg_modules is not None:
            head_params += [p for p in block_asg_modules.parameters() if p.requires_grad]
    if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
        head_params += list(model.saliency_adapter.parameters())
    if hasattr(model, "ms_aggregator") and model.ms_aggregator is not None:
        head_params += list(model.ms_aggregator.parameters())
    if hasattr(model, "detail_enhancer") and model.detail_enhancer is not None:
        head_params += list(model.detail_enhancer.parameters())

    self_prompt_head = None
    refine_self_prompt_head = None
    coarse_mask_head = None
    dynamic_sparse_prompt_head = None
    contrastive_prompt = None
    if getattr(args, "use_self_prompting", False):
        if getattr(args, "use_pgap", False):
            log_line("[warn] --use_self_prompting only applies when --use_pgap is disabled; PGAP will remain the active prompt generator.", args.log_file)
        try:
            try:
                neck_dim = model.image_encoder.neck[0].out_channels
            except Exception:
                neck_dim = 256
            sp_top_k_pos = int(args.self_prompt_top_k_pos) if args.self_prompt_top_k_pos is not None else int(args.n_pos)
            sp_top_k_neg = int(args.self_prompt_top_k_neg) if args.self_prompt_top_k_neg is not None else int(args.n_neg)
            sp_boundary_ratio = (
                float(args.self_prompt_boundary_ratio)
                if args.self_prompt_boundary_ratio is not None
                else float(getattr(args, "boundary_ratio", 0.5))
            )
            self_prompt_head = build_self_prompting_head(
                in_channels=neck_dim,
                hidden_channels=max(8, int(args.self_prompt_hidden_channels)),
                top_k_pos=max(1, sp_top_k_pos),
                top_k_neg=max(0, sp_top_k_neg),
                min_dist=max(1, int(args.self_prompt_min_dist)),
                peak_thr=float(getattr(args, "self_prompt_peak_thr", 0.1)),
                low_response_thr=float(getattr(args, "self_prompt_low_response_thr", 0.3)),
                boundary_aware=bool(getattr(args, "self_prompt_boundary_aware", False)),
                boundary_ratio=sp_boundary_ratio,
            ).to(device)
            configure_self_prompt_sampler(self_prompt_head, str(getattr(args, "self_prompt_sampler_variant", "legacy")))
            configure_self_prompt_guidance(self_prompt_head, args, prefix="self_prompt")
            head_params += list(self_prompt_head.parameters())
            n_sp = sum(p.numel() for p in self_prompt_head.parameters())
            sp_mode = "boundary-aware" if getattr(args, "self_prompt_boundary_aware", False) else "heatmap"
            log_line(
                f"Self-prompting enabled ({sp_mode}): pos={max(1, sp_top_k_pos)}, neg={max(0, sp_top_k_neg)}, "
                f"boundary_ratio={sp_boundary_ratio:.2f}, "
                f"sampler={str(getattr(args, 'self_prompt_sampler_variant', 'legacy'))}, "
                f"positive_guidance={str(getattr(args, 'self_prompt_positive_guidance', 'none'))}, "
                f"guidance_alpha={float(getattr(args, 'self_prompt_guidance_alpha', 0.75)):.2f}, "
                f"guidance_bg_alpha={float(getattr(args, 'self_prompt_guidance_bg_alpha', 0.5)):.2f}, "
                f"loss_channels={str(getattr(args, 'self_prompt_loss_channels', 'all'))}, "
                f"mix_ratio={float(getattr(args, 'self_prompt_mix_ratio', 0.5)):.2f}, "
                f"mix_start={int(getattr(args, 'self_prompt_mix_start_epoch', 1))}, "
                f"warmup={int(getattr(args, 'self_prompt_warmup', 30))}, "
                f"sup_weight={float(getattr(args, 'self_prompt_sup_weight', 0.3)):.3f}, "
                f"sampled_weight={float(getattr(args, 'self_prompt_sampled_point_weight', 0.0)):.3f}, "
                f"sampled_mode={str(getattr(args, 'self_prompt_sampled_loss_mode', 'legacy'))}, "
                f"expected_weight={float(getattr(args, 'self_prompt_expected_hit_weight', 0.0)):.3f}, "
                f"component_peak_weight={float(getattr(args, 'self_prompt_component_peak_weight', 0.0)):.3f}, params={n_sp}",
                args.log_file,
            )
        except Exception as e:
            log_line(f"[warn] Failed to initialize self-prompt head: {e}", args.log_file)
            self_prompt_head = None
        if self_prompt_head is not None and (
            float(getattr(args, "self_prompt_cl_weight", 0.0)) > 0.0
            or bool(getattr(args, "self_prompt_inject_sparse_tokens", False))
        ):
            try:
                contrastive_prompt = build_contrastive_prompt_learning(
                    embed_dim=getattr(model.prompt_encoder, "embed_dim", 256),
                    proj_dim=max(8, int(getattr(args, "self_prompt_cl_proj_dim", 128))),
                    temperature=float(getattr(args, "self_prompt_cl_temperature", 0.07)),
                    loss_type=str(getattr(args, "self_prompt_cl_loss", "infonce")),
                ).to(device)
                head_params += list(contrastive_prompt.parameters())
                n_cp = sum(p.numel() for p in contrastive_prompt.parameters())
                log_line(
                    f"Contrastive prompt enabled: loss={getattr(args, 'self_prompt_cl_loss', 'infonce')}, "
                    f"proj_dim={int(getattr(args, 'self_prompt_cl_proj_dim', 128))}, temp={float(getattr(args, 'self_prompt_cl_temperature', 0.07)):.3f}, "
                    f"inject_tokens={bool(getattr(args, 'self_prompt_inject_sparse_tokens', False))}, params={n_cp}",
                    args.log_file,
                )
            except Exception as e:
                log_line(f"[warn] Failed to initialize contrastive prompt module: {e}", args.log_file)
                contrastive_prompt = None

    if getattr(args, "use_two_stage_self_prompting", False):
        if self_prompt_head is None:
            log_line("[warn] --use_two_stage_self_prompting requires --use_self_prompting; stage-2 prompt head disabled.", args.log_file)
            refine_self_prompt_head = None
        else:
            try:
                try:
                    neck_dim = model.image_encoder.neck[0].out_channels
                except Exception:
                    neck_dim = 256
                sp_top_k_pos = int(args.self_prompt_top_k_pos) if args.self_prompt_top_k_pos is not None else int(args.n_pos)
                sp_top_k_neg = int(args.self_prompt_top_k_neg) if args.self_prompt_top_k_neg is not None else int(args.n_neg)
                sp_boundary_ratio = (
                    float(args.self_prompt_boundary_ratio)
                    if args.self_prompt_boundary_ratio is not None
                    else float(getattr(args, "boundary_ratio", 0.5))
                )
                refine_self_prompt_head = build_mask_guided_self_prompting_head(
                    in_channels=neck_dim,
                    hidden_channels=max(8, int(getattr(args, "stage2_self_prompt_hidden_channels", 64))),
                    top_k_pos=max(1, sp_top_k_pos),
                    top_k_neg=max(0, sp_top_k_neg),
                    min_dist=max(1, int(args.self_prompt_min_dist)),
                    peak_thr=float(getattr(args, "self_prompt_peak_thr", 0.1)),
                    low_response_thr=float(getattr(args, "self_prompt_low_response_thr", 0.3)),
                    boundary_ratio=sp_boundary_ratio,
                    mask_feature_channels=int(getattr(args, "stage2_self_prompt_mask_feature_channels", 3)),
                    detach_mask=bool(getattr(args, "stage2_self_prompt_detach_mask", True)),
                ).to(device)
                configure_self_prompt_sampler(
                    refine_self_prompt_head,
                    str(getattr(args, "stage2_self_prompt_sampler_variant", "fgcomp_bg_r05")),
                )
                configure_self_prompt_guidance(refine_self_prompt_head, args, prefix="stage2_self_prompt")
                head_params += list(refine_self_prompt_head.parameters())
                n_refine = sum(p.numel() for p in refine_self_prompt_head.parameters())
                log_line(
                    "Two-stage self-prompting enabled: "
                    f"stage1_sampler={str(getattr(args, 'self_prompt_sampler_variant', 'legacy'))}, "
                    f"stage2_sampler={str(getattr(args, 'stage2_self_prompt_sampler_variant', 'fgcomp_bg_r05'))}, "
                    f"stage2_positive_guidance={str(getattr(args, 'stage2_self_prompt_positive_guidance', None) or getattr(args, 'self_prompt_positive_guidance', 'none'))}, "
                    f"stage2_loss_channels={str(getattr(args, 'stage2_self_prompt_loss_channels', 'fg_bg'))}, "
                    f"stage2_sup_weight={float(getattr(args, 'stage2_self_prompt_sup_weight', 0.3)):.3f}, "
                    f"stage2_component_peak_weight={float(getattr(args, 'stage2_self_prompt_component_peak_weight', 0.05)):.3f}, "
                    f"stage1_mask_weight={float(getattr(args, 'two_stage_stage1_loss_weight', 0.2)):.3f}, "
                    f"detach_mask={bool(getattr(args, 'stage2_self_prompt_detach_mask', True))}, params={n_refine}",
                    args.log_file,
                )
            except Exception as e:
                log_line(f"[warn] Failed to initialize two-stage self-prompt head: {e}", args.log_file)
                refine_self_prompt_head = None

    if getattr(args, "use_coarse_mask_prompt", False):
        try:
            try:
                neck_dim = model.image_encoder.neck[0].out_channels
            except Exception:
                neck_dim = 256
            coarse_mask_head = build_coarse_mask_prompt_head(
                in_channels=neck_dim,
                hidden_channels=max(8, int(getattr(args, "coarse_mask_prompt_hidden_channels", 64))),
            ).to(device)
            head_params += list(coarse_mask_head.parameters())
            n_cmp = sum(p.numel() for p in coarse_mask_head.parameters())
            log_line(
                "Coarse mask prompt enabled: "
                f"hidden={int(getattr(args, 'coarse_mask_prompt_hidden_channels', 64))}, "
                f"weight={float(getattr(args, 'coarse_mask_prompt_weight', 0.0)):.3f}, "
                f"bce={float(getattr(args, 'coarse_mask_prompt_bce_weight', 1.0)):.3f}, "
                f"dice={float(getattr(args, 'coarse_mask_prompt_dice_weight', 1.0)):.3f}, "
                f"pos_weight={float(getattr(args, 'coarse_mask_prompt_pos_weight', 10.0)):.3f}, "
                f"merge_alpha={float(getattr(args, 'coarse_mask_prompt_merge_alpha', 0.5)):.3f}, "
                f"params={n_cmp}",
                args.log_file,
            )
        except Exception as e:
            log_line(f"[warn] Failed to initialize coarse mask prompt head: {e}", args.log_file)
            coarse_mask_head = None

    if getattr(args, "use_dynamic_sparse_prompt", False):
        try:
            try:
                neck_dim = model.image_encoder.neck[0].out_channels
            except Exception:
                neck_dim = 256
            embed_dim = int(getattr(model.prompt_encoder, "embed_dim", 256))
            if bool(getattr(args, "dynamic_sparse_multilevel", False)):
                try:
                    level_dim = int(model.image_encoder.patch_embed.proj.out_channels)
                except Exception:
                    level_dim = 192 if args.model == "vitt" else 384
                dynamic_sparse_prompt_head = build_multilevel_dynamic_sparse_prompt_head(
                    level_channels=[level_dim, level_dim, level_dim, level_dim],
                    neck_channels=neck_dim,
                    embed_dim=embed_dim,
                    num_tokens=max(1, int(getattr(args, "dynamic_sparse_num_tokens", 8))),
                    hidden_channels=max(8, int(getattr(args, "dynamic_sparse_hidden_channels", 128))),
                    temperature=float(getattr(args, "dynamic_sparse_temperature", 1.0)),
                    init_scale=float(getattr(args, "dynamic_sparse_init_scale", 0.02)),
                    loc_levels=max(1, int(getattr(args, "dynamic_sparse_loc_levels", 3))),
                ).to(device)
            else:
                dynamic_sparse_prompt_head = build_dynamic_sparse_prompt_head(
                    in_channels=neck_dim,
                    embed_dim=embed_dim,
                    num_tokens=max(1, int(getattr(args, "dynamic_sparse_num_tokens", 8))),
                    hidden_channels=max(8, int(getattr(args, "dynamic_sparse_hidden_channels", 64))),
                    temperature=float(getattr(args, "dynamic_sparse_temperature", 1.0)),
                    init_scale=float(getattr(args, "dynamic_sparse_init_scale", 0.1)),
                ).to(device)
            head_params += list(dynamic_sparse_prompt_head.parameters())
            n_dyn = sum(p.numel() for p in dynamic_sparse_prompt_head.parameters())
            log_line(
                "Dynamic sparse prompt enabled: "
                f"multilevel={bool(getattr(args, 'dynamic_sparse_multilevel', False))}, "
                f"tokens={int(getattr(args, 'dynamic_sparse_num_tokens', 8))}, "
                f"hidden={int(getattr(args, 'dynamic_sparse_hidden_channels', 64))}, "
                f"embed_dim={embed_dim}, "
                f"temperature={float(getattr(args, 'dynamic_sparse_temperature', 1.0)):.3f}, "
                f"init_scale={float(getattr(args, 'dynamic_sparse_init_scale', 0.1)):.3f}, "
                f"div_weight={float(getattr(args, 'dynamic_sparse_div_weight', 0.0)):.4f}, "
                f"target_weight={float(getattr(args, 'dynamic_sparse_target_weight', 0.0)):.4f}, "
                f"head_only={bool(getattr(args, 'dynamic_sparse_train_head_only', False))}, "
                f"params={n_dyn}",
                args.log_file,
            )
        except Exception as e:
            log_line(f"[warn] Failed to initialize dynamic sparse prompt head: {e}", args.log_file)
            dynamic_sparse_prompt_head = None

    # MLLM text modules (global CLIP feature -> FiLM / sparse prompt / dense mask prompt)
    text_conditioner = None
    text_sparse_prompt = None
    text_dense_prompt = None
    tassg = None
    bifusion_adapter = None
    backbone_bifusion_adapter = None
    pgap_text_prior_only = bool(getattr(args, "pgap_text_prior_only", False))
    if getattr(args, "use_mllm_prompt", False):
        img_dim = 256  # EfficientSAM-ViTT image embedding dim
        if not pgap_text_prior_only and not getattr(args, "disable_text_conditioner", False):
            text_conditioner = build_text_conditioner(
                img_dim=img_dim,
                text_dim=args.mllm_text_dim,
            ).to(device)
            head_params += list(text_conditioner.parameters())
            n_tc = sum(p.numel() for p in text_conditioner.parameters())
            log_line(f"MLLM TextConditioner enabled: text_dim={args.mllm_text_dim}, params={n_tc}", args.log_file)
        elif getattr(args, "disable_text_conditioner", False):
            log_line("MLLM TextConditioner disabled by --disable_text_conditioner.", args.log_file)
        else:
            log_line("MLLM TextConditioner disabled by --pgap_text_prior_only.", args.log_file)
        if getattr(args, "use_text_sparse_prompt", False) and not pgap_text_prior_only:
            text_sparse_prompt = build_text_sparse_prompt_projector(
                text_dim=args.mllm_text_dim,
                embed_dim=getattr(model.prompt_encoder, "embed_dim", 256),
                num_tokens=max(1, int(args.text_sparse_num_tokens)),
                init_scale=float(args.text_sparse_init_scale),
                use_raw_global_gate=bool(getattr(args, "text_sparse_raw_global_gate", False)),
                raw_global_gate_init_bias=float(getattr(args, "text_sparse_raw_global_gate_init_bias", -2.0)),
            ).to(device)
            head_params += list(text_sparse_prompt.parameters())
            n_tsp = sum(p.numel() for p in text_sparse_prompt.parameters())
            log_line(
                f"Text sparse prompt enabled: source={args.text_sparse_prompt_source}, tokens={args.text_sparse_num_tokens}, gate={bool(getattr(args, 'text_sparse_raw_global_gate', False))}, params={n_tsp}",
                args.log_file,
            )
        elif getattr(args, "use_text_sparse_prompt", False) and pgap_text_prior_only:
            log_line("[info] Ignoring --use_text_sparse_prompt because --pgap_text_prior_only is enabled.", args.log_file)
        if getattr(args, "use_text_dense_prompt", False):
            dense_variant = getattr(args, "text_dense_prompt_type", "global")
            if dense_variant == "token_xattn":
                text_dense_prompt = build_text_dense_mask_prompt_generator_v2(
                    img_dim=img_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                    num_heads=max(1, int(args.text_dense_num_heads)),
                ).to(device)
            else:
                text_dense_prompt = build_text_dense_mask_prompt_generator(
                    img_dim=img_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.text_dense_hidden_dim)),
                ).to(device)
            head_params += list(text_dense_prompt.parameters())
            n_tdp = sum(p.numel() for p in text_dense_prompt.parameters())
            log_line(
                f"Text dense mask prompt enabled: type={dense_variant}, hidden={args.text_dense_hidden_dim}, "
                f"heads={getattr(args, 'text_dense_num_heads', 4)}, alpha={args.text_dense_prompt_merge_alpha}, params={n_tdp}",
                args.log_file,
            )
        if getattr(args, "use_bifusion_adapter", False):
            if pgap_text_prior_only:
                log_line("[info] Disabling BiFusion due to --pgap_text_prior_only.", args.log_file)
            else:
                try:
                    interms_dim = int(model.image_encoder.patch_embed.proj.out_channels)
                except Exception:
                    interms_dim = int(getattr(args, "bifusion_interms_dim", 192))
                bifusion_adapter = build_bifusion_adapter_lite(
                    img_dim=img_dim,
                    interms_dim=interms_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.bifusion_hidden_dim)),
                    num_heads=max(1, int(args.bifusion_num_heads)),
                    use_interms_level=not bool(getattr(args, "bifusion_disable_interms_level", False)),
                    img_res_scale=float(getattr(args, "bifusion_img_res_scale", 1.0)),
                    interms_res_scale=float(getattr(args, "bifusion_interms_res_scale", 1.0)),
                    text_res_scale=float(getattr(args, "bifusion_text_res_scale", 1.0)),
                ).to(device)
                head_params += list(bifusion_adapter.parameters())
                n_bf = sum(p.numel() for p in bifusion_adapter.parameters())
                log_line(
                    f"BiFusion adapter enabled: interms+img levels, hidden={args.bifusion_hidden_dim}, "
                    f"heads={args.bifusion_num_heads}, params={n_bf}",
                    args.log_file,
                )
        use_plain_backbone_bifusion = bool(getattr(args, "use_bifusion_backbone_blocks", False))
        use_gated_backbone_bifusion = bool(getattr(args, "use_gated_bifusion_backbone_blocks", False))
        if use_plain_backbone_bifusion and use_gated_backbone_bifusion:
            log_line("[warn] Both --use_bifusion_backbone_blocks and --use_gated_bifusion_backbone_blocks are set; using gated backbone BiFusion.", args.log_file)
            use_plain_backbone_bifusion = False
        if use_plain_backbone_bifusion or use_gated_backbone_bifusion:
            if pgap_text_prior_only:
                log_line("[info] Disabling backbone BiFusion due to --pgap_text_prior_only.", args.log_file)
            else:
                try:
                    vision_dim = int(model.image_encoder.patch_embed.proj.out_channels)
                except Exception:
                    vision_dim = int(getattr(args, "bifusion_interms_dim", 192))
                num_layers = len(getattr(model.image_encoder, "blocks", []))
                common_kwargs = dict(
                    num_layers=max(1, int(num_layers)),
                    vision_dim=vision_dim,
                    text_dim=args.mllm_text_dim,
                    hidden_dim=max(8, int(args.bifusion_hidden_dim)),
                    num_heads=max(1, int(args.bifusion_num_heads)),
                    apply_every=max(1, int(getattr(args, "bifusion_block_apply_every", 1))),
                    vision_res_scale=float(getattr(args, "bifusion_block_vision_res_scale", 1.0)),
                    text_res_scale=float(getattr(args, "bifusion_block_text_res_scale", 1.0)),
                )
                if use_gated_backbone_bifusion:
                    backbone_bifusion_adapter = build_gated_backbone_bifusion_block_adapter(
                        gate_hidden_dim=int(getattr(args, "bifusion_gate_hidden_dim", 0)),
                        gate_init_bias=float(getattr(args, "bifusion_gate_init_bias", -2.0)),
                        delta_only=bool(getattr(args, "bifusion_gate_delta_only", False)),
                        **common_kwargs,
                    ).to(device)
                else:
                    backbone_bifusion_adapter = build_backbone_bifusion_block_adapter(
                        **common_kwargs,
                    ).to(device)
                head_params += list(backbone_bifusion_adapter.parameters())
                if hasattr(model.image_encoder, "set_text_block_fuser"):
                    model.image_encoder.set_text_block_fuser(backbone_bifusion_adapter)
                else:
                    model.image_encoder.block_text_fuser = backbone_bifusion_adapter
                n_bfb = sum(p.numel() for p in backbone_bifusion_adapter.parameters())
                if use_gated_backbone_bifusion:
                    log_line(
                        f"Gated Backbone BiFusion enabled: layers={num_layers}, hidden={args.bifusion_hidden_dim}, "
                        f"heads={args.bifusion_num_heads}, every={getattr(args, 'bifusion_block_apply_every', 1)}, "
                        f"gate_hidden={getattr(args, 'bifusion_gate_hidden_dim', 0)}, gate_bias={getattr(args, 'bifusion_gate_init_bias', -2.0)}, "
                        f"delta_only={getattr(args, 'bifusion_gate_delta_only', False)}, params={n_bfb}",
                        args.log_file,
                    )
                else:
                    log_line(
                        f"Backbone BiFusion enabled: layers={num_layers}, hidden={args.bifusion_hidden_dim}, "
                        f"heads={args.bifusion_num_heads}, every={getattr(args, 'bifusion_block_apply_every', 1)}, params={n_bfb}",
                        args.log_file,
                    )
    elif (
        getattr(args, "use_text_sparse_prompt", False)
        or getattr(args, "use_text_dense_prompt", False)
        or getattr(args, "use_bifusion_backbone_blocks", False)
        or getattr(args, "use_gated_bifusion_backbone_blocks", False)
    ):
        log_line("[warn] Text sparse/dense/backbone-bifusion flags require --use_mllm_prompt; ignoring.", args.log_file)
    elif getattr(args, "use_bifusion_adapter", False):
        log_line("[warn] --use_bifusion_adapter requires --use_mllm_prompt; ignoring.", args.log_file)

    if getattr(args, "use_tassg", False):
        tassg_text_dim = getattr(args, "tassg_text_dim", None)
        if tassg_text_dim is None:
            tassg_text_dim = int(getattr(args, "mllm_text_dim", 512))
        tassg = build_targetness_aware_semantic_slot_generator(
            img_dim=int(getattr(args, "tassg_img_dim", 256)),
            text_dim=int(tassg_text_dim),
            num_slots=max(1, int(getattr(args, "tassg_num_slots", 8))),
            hidden_dim=max(8, int(getattr(args, "tassg_hidden_dim", 256))),
            num_heads=max(1, int(getattr(args, "tassg_num_heads", 4))),
            dropout=float(getattr(args, "tassg_dropout", 0.0)),
        ).to(device)
        head_params += list(tassg.parameters())
        n_tassg = sum(p.numel() for p in tassg.parameters())
        log_line(
            "TASSG enabled: "
            f"semantic_source={getattr(args, 'semantic_source', 'teacher')}, "
            f"slots={int(getattr(args, 'tassg_num_slots', 8))}, "
            f"img_dim={int(getattr(args, 'tassg_img_dim', 256))}, "
            f"text_dim={int(tassg_text_dim)}, "
            f"hidden={int(getattr(args, 'tassg_hidden_dim', 256))}, "
            f"heads={int(getattr(args, 'tassg_num_heads', 4))}, "
            f"two_pass={bool(getattr(args, 'tassg_two_pass_backbone', False))}, "
            f"lambda_global={float(getattr(args, 'lambda_tassg_global', 0.0)):.4f}, "
            f"lambda_token={float(getattr(args, 'lambda_tassg_token', 0.0)):.4f}, "
            f"lambda_prompt={float(getattr(args, 'lambda_tassg_prompt', 0.0)):.4f}, "
            f"lambda_targetness={float(getattr(args, 'lambda_tassg_targetness', 0.0)):.4f}, "
            f"params={n_tassg}",
            args.log_file,
        )
        log_line(
            "TASSG deployment contract: "
            "Qwen3-VL used during training=teacher distillation only; "
            f"Qwen3-VL used during inference={str(getattr(args, 'semantic_source', 'teacher')).lower() == 'teacher'}; "
            f"CLIP cached feature required during inference={str(getattr(args, 'semantic_source', 'teacher')).lower() == 'teacher'}; "
            f"tassg_two_pass_backbone={bool(getattr(args, 'tassg_two_pass_backbone', False))}; "
            f"text_sparse_prompt_source={getattr(args, 'text_sparse_prompt_source', 'fused_tokens')}; "
            f"text_sparse_num_tokens={getattr(args, 'text_sparse_num_tokens', 1)}",
            args.log_file,
        )
        if str(getattr(args, "semantic_source", "teacher")) == "student" and text_sparse_prompt is None and bifusion_adapter is None and backbone_bifusion_adapter is None and text_conditioner is None:
            log_line(
                "[warn] --semantic_source student is enabled but no text prompt/fusion module was constructed; TASSG will only add distillation losses.",
                args.log_file,
            )
    elif str(getattr(args, "semantic_source", "teacher")) == "student":
        raise RuntimeError("--semantic_source student requires --use_tassg.")

    if getattr(args, "pgap_text_fuse_internal", False):
        if not getattr(args, "use_pgap", False):
            log_line("[warn] --pgap_text_fuse_internal is set but --use_pgap is disabled; internal fusion will not run.", args.log_file)
        if not (getattr(args, "use_mllm_prompt", False) and getattr(args, "use_text_dense_prompt", False)):
            log_line("[warn] --pgap_text_fuse_internal requires --use_mllm_prompt and --use_text_dense_prompt for text prior; falling back to PGAP-only prompts.", args.log_file)
    if getattr(args, "pgap_text_prior_only", False):
        if not getattr(args, "use_pgap", False):
            log_line("[warn] --pgap_text_prior_only has no effect because --use_pgap is disabled.", args.log_file)
        if not getattr(args, "pgap_text_fuse_internal", False):
            log_line("[warn] --pgap_text_prior_only is set without --pgap_text_fuse_internal; text will not affect PGAP prompts.", args.log_file)
    if lca_prompt is not None:
        head_params += list(lca_prompt.parameters())

    unfreeze_epoch = (args.epochs // 4) if (args.freeze_encoder_epochs is None or args.freeze_encoder_epochs <= 0) else args.freeze_encoder_epochs
    resume_ckpt = None
    resume_start_epoch = 1
    best_iou = -1.0
    resume_path = getattr(args, "resume_ckpt", None)
    if resume_path:
        if not os.path.isfile(resume_path):
            log_line(f"[warn] --resume_ckpt not found: {resume_path}", args.log_file)
        else:
            try:
                resume_ckpt = torch.load(resume_path, map_location="cpu")
                model_state = resume_ckpt.get("model", resume_ckpt) if isinstance(resume_ckpt, dict) else resume_ckpt
                missing, unexpected = model.load_state_dict(model_state, strict=False)
                if isinstance(resume_ckpt, dict):
                    def _restore_aux_module(module, key):
                        if module is None or key not in resume_ckpt:
                            return
                        try:
                            aux_missing, aux_unexpected = module.load_state_dict(resume_ckpt[key], strict=False)
                            log_line(
                                f"Restored {key}: missing={len(aux_missing)}, unexpected={len(aux_unexpected)}",
                                args.log_file,
                            )
                        except Exception as e:
                            log_line(f"[warn] Failed to restore {key}: {e}", args.log_file)

                    _restore_aux_module(self_prompt_head, "self_prompt_head")
                    _restore_aux_module(refine_self_prompt_head, "refine_self_prompt_head")
                    _restore_aux_module(coarse_mask_head, "coarse_mask_head")
                    _restore_aux_module(dynamic_sparse_prompt_head, "dynamic_sparse_prompt_head")
                    _restore_aux_module(contrastive_prompt, "contrastive_prompt")
                    _restore_aux_module(text_conditioner, "text_conditioner")
                    _restore_aux_module(text_sparse_prompt, "text_sparse_prompt")
                    _restore_aux_module(text_dense_prompt, "text_dense_prompt")
                    _restore_aux_module(tassg, "tassg")
                    _restore_aux_module(bifusion_adapter, "bifusion_adapter")
                    _restore_aux_module(backbone_bifusion_adapter, "backbone_bifusion_adapter")
                    _restore_aux_module(lca_prompt, "lca_prompt")
                resume_epoch = int(resume_ckpt.get("epoch", 0)) if isinstance(resume_ckpt, dict) else 0
                resume_start_epoch = min(args.epochs + 1, max(1, resume_epoch + 1))
                best_iou = float(resume_ckpt.get("best_iou", -1.0)) if isinstance(resume_ckpt, dict) else -1.0
                if bool(getattr(args, "resume_reset_best", False)):
                    best_iou = -1.0
                log_line(
                    f"Resumed checkpoint: path={resume_path}, checkpoint_epoch={resume_epoch}, "
                    f"start_epoch={resume_start_epoch}, target_epochs={args.epochs}, "
                    f"actual_extra_epochs={max(0, int(args.epochs) - int(resume_epoch))}, "
                    f"resume_reset_optimizer={bool(getattr(args, 'resume_reset_optimizer', False))}, "
                    f"resume_reset_best={bool(getattr(args, 'resume_reset_best', False))}, "
                    f"best_iou={best_iou:.4f}, missing={len(missing)}, unexpected={len(unexpected)}",
                    args.log_file,
                )
                if resume_start_epoch > max(1, unfreeze_epoch):
                    for p_ in model.image_encoder.parameters():
                        p_.requires_grad = True
                    for p_ in model.mask_decoder.parameters():
                        p_.requires_grad = True
                    for p_ in model.prompt_encoder.parameters():
                        p_.requires_grad = True
                    head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
                    if (
                        getattr(args, "use_asg_hq", False)
                        and args.asg_loc in ("encoder", "both")
                        and getattr(model.image_encoder, "asg_encoder_mode", "neck") != "block_only"
                        and getattr(model.image_encoder, "radial_gate", None) is not None
                    ):
                        head_params += [p for p in model.image_encoder.radial_gate.parameters() if p.requires_grad]
                    if (
                        getattr(args, "use_asg_hq", False)
                        and args.asg_loc in ("encoder", "both")
                        and getattr(model.image_encoder, "asg_encoder_mode", "neck") == "block_only"
                    ):
                        block_asg_modules = getattr(model.image_encoder, "block_asg_modules", None)
                        if block_asg_modules is not None:
                            head_params += [p for p in block_asg_modules.parameters() if p.requires_grad]
                    if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
                        head_params += list(model.saliency_adapter.parameters())
                    if hasattr(model, "ms_aggregator") and model.ms_aggregator is not None:
                        head_params += list(model.ms_aggregator.parameters())
                    if hasattr(model, "detail_enhancer") and model.detail_enhancer is not None:
                        head_params += list(model.detail_enhancer.parameters())
                    if self_prompt_head is not None:
                        head_params += list(self_prompt_head.parameters())
                    if refine_self_prompt_head is not None:
                        head_params += list(refine_self_prompt_head.parameters())
                    if coarse_mask_head is not None:
                        head_params += list(coarse_mask_head.parameters())
                    if dynamic_sparse_prompt_head is not None:
                        head_params += list(dynamic_sparse_prompt_head.parameters())
                    if contrastive_prompt is not None:
                        head_params += list(contrastive_prompt.parameters())
                    if text_conditioner is not None:
                        head_params += list(text_conditioner.parameters())
                    if text_sparse_prompt is not None:
                        head_params += list(text_sparse_prompt.parameters())
                    if text_dense_prompt is not None:
                        head_params += list(text_dense_prompt.parameters())
                    if tassg is not None:
                        head_params += list(tassg.parameters())
                    if bifusion_adapter is not None:
                        head_params += list(bifusion_adapter.parameters())
                    if backbone_bifusion_adapter is not None:
                        head_params += list(backbone_bifusion_adapter.parameters())
                    if lca_prompt is not None:
                        head_params += list(lca_prompt.parameters())
                    log_line("Resume starts after unfreeze point; restored full trainable parameter set before optimizer creation.", args.log_file)
            except Exception as e:
                log_line(f"[warn] Failed to resume checkpoint {resume_path}: {e}", args.log_file)
                resume_ckpt = None
                resume_start_epoch = 1
                best_iou = -1.0
    if getattr(args, "dynamic_sparse_train_head_only", False):
        if dynamic_sparse_prompt_head is None:
            raise RuntimeError("--dynamic_sparse_train_head_only requires --use_dynamic_sparse_prompt.")
        for p_ in model.parameters():
            p_.requires_grad = False
        for module in (
            self_prompt_head,
            refine_self_prompt_head,
            coarse_mask_head,
            contrastive_prompt,
            text_conditioner,
            text_sparse_prompt,
            text_dense_prompt,
            tassg,
            bifusion_adapter,
            backbone_bifusion_adapter,
            lca_prompt,
        ):
            if module is not None:
                for p_ in module.parameters():
                    p_.requires_grad = False
        for p_ in dynamic_sparse_prompt_head.parameters():
            p_.requires_grad = True
        head_params = list(dynamic_sparse_prompt_head.parameters())
        log_line(
            "Dynamic sparse head-only training: froze model and existing auxiliary heads; optimizer will train only dynamic_sparse_prompt_head.",
            args.log_file,
        )
    head_params = _dedup_trainable_params(head_params)
    enc_params = _exclude_params(model.image_encoder.parameters(), head_params)

    optimizer = _make_optimizer(head_params, enc_params, args)
    if (
        resume_ckpt is not None
        and isinstance(resume_ckpt, dict)
        and "optimizer" in resume_ckpt
        and not bool(getattr(args, "resume_reset_optimizer", False))
    ):
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            log_line("Restored optimizer state from resume checkpoint.", args.log_file)
        except Exception as e:
            log_line(f"[warn] Failed to restore optimizer state from resume checkpoint: {e}", args.log_file)
    elif resume_ckpt is not None and bool(getattr(args, "resume_reset_optimizer", False)):
        log_line("Resume checkpoint loaded; optimizer state reset by --resume_reset_optimizer.", args.log_file)
    scaler = make_scaler(device)

    if resume_start_epoch > args.epochs:
        log_line(f"Resume checkpoint is already at or beyond target epochs={args.epochs}; nothing to train.", args.log_file)
        return
    for epoch in range(resume_start_epoch, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args,
            pgap=pgap, fab_criterion=fab_criterion, scr_criterion=scr_criterion,
            text_conditioner=text_conditioner,
            text_sparse_prompt=text_sparse_prompt,
            text_dense_prompt=text_dense_prompt,
            tassg=tassg,
            bifusion_adapter=bifusion_adapter,
            backbone_bifusion_adapter=backbone_bifusion_adapter,
            self_prompt_head=self_prompt_head,
            refine_self_prompt_head=refine_self_prompt_head,
            coarse_mask_head=coarse_mask_head,
            dynamic_sparse_prompt_head=dynamic_sparse_prompt_head,
            contrastive_prompt=contrastive_prompt,
            lca_prompt=lca_prompt,
        )
        miou, niou, mf1, pd_val, fa_val, thr_used = validate(
            model, val_loader, device, args, epoch, pgap=pgap,
            text_conditioner=text_conditioner,
            text_sparse_prompt=text_sparse_prompt,
            text_dense_prompt=text_dense_prompt,
            tassg=tassg,
            bifusion_adapter=bifusion_adapter,
            backbone_bifusion_adapter=backbone_bifusion_adapter,
            self_prompt_head=self_prompt_head,
            refine_self_prompt_head=refine_self_prompt_head,
            coarse_mask_head=coarse_mask_head,
            dynamic_sparse_prompt_head=dynamic_sparse_prompt_head,
            contrastive_prompt=contrastive_prompt,
            lca_prompt=lca_prompt,
        )
        dt = time.time() - t0
        log_line(
            f"[Epoch {epoch:03d}] loss={train_stats['loss']:.4f} miou={miou:.4f} niou={niou:.4f} f1={mf1:.4f} "
            f"pd={pd_val:.4f} fa={fa_val:.6f} thr={thr_used:.2f} time={dt:.1f}s "
            f"center_loss={train_stats['center_loss']:.4f} center_contain={train_stats['center_contain']:.4f} "
            f"sp_sampled={train_stats.get('self_prompt_sampled', 0.0):.4f} "
            f"sp_expected={train_stats.get('self_prompt_expected', 0.0):.4f} "
            f"sp_component={train_stats.get('self_prompt_component', 0.0):.4f} "
            f"stage2_sp={train_stats.get('stage2_self_prompt', 0.0):.4f} "
            f"stage2_sampled={train_stats.get('stage2_self_prompt_sampled', 0.0):.4f} "
            f"stage2_expected={train_stats.get('stage2_self_prompt_expected', 0.0):.4f} "
            f"stage2_component={train_stats.get('stage2_self_prompt_component', 0.0):.4f} "
            f"stage1_mask={train_stats.get('stage1_mask', 0.0):.4f} "
            f"coarse_mask={train_stats.get('coarse_mask', 0.0):.4f} "
            f"dyn_sparse={train_stats.get('dynamic_sparse', 0.0):.4f} "
            f"dyn_target={train_stats.get('dynamic_sparse_target', 0.0):.4f} "
            f"tassg_global={train_stats.get('tassg_global', 0.0):.4f} "
            f"tassg_token={train_stats.get('tassg_token', 0.0):.4f} "
            f"tassg_prompt={train_stats.get('tassg_prompt', 0.0):.4f} "
            f"tassg_targetness={train_stats.get('tassg_targetness', 0.0):.4f} "
            f"semantic_source={train_stats.get('semantic_source', getattr(args, 'semantic_source', 'teacher'))}",
            args.log_file,
        )

        # Unfreeze schedule
        if (not getattr(args, "dynamic_sparse_train_head_only", False)) and epoch == max(1, unfreeze_epoch):
            for p_ in model.image_encoder.parameters():
                p_.requires_grad = True
            if args.unfreeze_all_when_encoder:
                for p_ in model.mask_decoder.parameters():
                    p_.requires_grad = True
                for p_ in model.prompt_encoder.parameters():
                    p_.requires_grad = True
            # Rebuild optimizer to include newly trainable params
            head_params = [p for p in list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()) if p.requires_grad]
            if (
                getattr(args, "use_asg_hq", False)
                and args.asg_loc in ("encoder", "both")
                and getattr(model.image_encoder, "asg_encoder_mode", "neck") != "block_only"
                and getattr(model.image_encoder, "radial_gate", None) is not None
            ):
                head_params += [p for p in model.image_encoder.radial_gate.parameters() if p.requires_grad]
            if (
                getattr(args, "use_asg_hq", False)
                and args.asg_loc in ("encoder", "both")
                and getattr(model.image_encoder, "asg_encoder_mode", "neck") == "block_only"
            ):
                block_asg_modules = getattr(model.image_encoder, "block_asg_modules", None)
                if block_asg_modules is not None:
                    head_params += [p for p in block_asg_modules.parameters() if p.requires_grad]
            if hasattr(model, "saliency_adapter") and model.saliency_adapter is not None:
                head_params += list(model.saliency_adapter.parameters())
            if hasattr(model, "ms_aggregator") and model.ms_aggregator is not None:
                head_params += list(model.ms_aggregator.parameters())
            if hasattr(model, "detail_enhancer") and model.detail_enhancer is not None:
                head_params += list(model.detail_enhancer.parameters())
            if self_prompt_head is not None:
                head_params += list(self_prompt_head.parameters())
            if refine_self_prompt_head is not None:
                head_params += list(refine_self_prompt_head.parameters())
            if coarse_mask_head is not None:
                head_params += list(coarse_mask_head.parameters())
            if dynamic_sparse_prompt_head is not None:
                head_params += list(dynamic_sparse_prompt_head.parameters())
            if contrastive_prompt is not None:
                head_params += list(contrastive_prompt.parameters())
            if text_conditioner is not None:
                head_params += list(text_conditioner.parameters())
            if text_sparse_prompt is not None:
                head_params += list(text_sparse_prompt.parameters())
            if text_dense_prompt is not None:
                head_params += list(text_dense_prompt.parameters())
            if tassg is not None:
                head_params += list(tassg.parameters())
            if bifusion_adapter is not None:
                head_params += list(bifusion_adapter.parameters())
            if backbone_bifusion_adapter is not None:
                head_params += list(backbone_bifusion_adapter.parameters())
            if lca_prompt is not None:
                head_params += list(lca_prompt.parameters())
            head_params = _dedup_trainable_params(head_params)
            enc_params = _exclude_params(model.image_encoder.parameters(), head_params)
            optimizer = _make_optimizer(head_params, enc_params, args)
            log_line(
                f"Unfroze at epoch {epoch}: encoder + {'all heads' if args.unfreeze_all_when_encoder else 'keep current head mask' }.",
                args.log_file,
            )

        is_best = miou > best_iou
        if is_best:
            best_iou = miou
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_iou": best_iou,
            "args": vars(args),
        }
        if self_prompt_head is not None:
            ckpt["self_prompt_head"] = self_prompt_head.state_dict()
        if refine_self_prompt_head is not None:
            ckpt["refine_self_prompt_head"] = refine_self_prompt_head.state_dict()
        if coarse_mask_head is not None:
            ckpt["coarse_mask_head"] = coarse_mask_head.state_dict()
        if dynamic_sparse_prompt_head is not None:
            ckpt["dynamic_sparse_prompt_head"] = dynamic_sparse_prompt_head.state_dict()
        if contrastive_prompt is not None:
            ckpt["contrastive_prompt"] = contrastive_prompt.state_dict()
        if text_conditioner is not None:
            ckpt["text_conditioner"] = text_conditioner.state_dict()
        if text_sparse_prompt is not None:
            ckpt["text_sparse_prompt"] = text_sparse_prompt.state_dict()
        if text_dense_prompt is not None:
            ckpt["text_dense_prompt"] = text_dense_prompt.state_dict()
        if tassg is not None:
            ckpt["tassg"] = tassg.state_dict()
        if bifusion_adapter is not None:
            ckpt["bifusion_adapter"] = bifusion_adapter.state_dict()
        if backbone_bifusion_adapter is not None:
            ckpt["backbone_bifusion_adapter"] = backbone_bifusion_adapter.state_dict()
        if lca_prompt is not None:
            ckpt["lca_prompt"] = lca_prompt.state_dict()
        if is_best:
            metric_tag = format_metric_tag(epoch, miou, niou, mf1, pd_val, fa_val)
            torch.save(ckpt, os.path.join(args.out_dir, f"best_{metric_tag}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))


if __name__ == "__main__":
    main()

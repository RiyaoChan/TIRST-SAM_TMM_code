"""
Self-Prompting Head for EfficientSAM IRSTD.

Replaces manual/GT point prompts with a learnable detection head
that predicts target heatmaps from encoder features, enabling
fully automatic end-to-end infrared small target detection.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_sigma_pairs(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((float(item[0]), float(item[1])))
            else:
                parts = str(item).replace(":", ",").split(",")
                if len(parts) == 2:
                    out.append((float(parts[0]), float(parts[1])))
        return out or list(default)
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.replace(":", "-").split("-")
        if len(parts) != 2:
            continue
        out.append((float(parts[0]), float(parts[1])))
    return out or list(default)


def _parse_sigmas(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value] or list(default)
    out = [float(v.strip()) for v in str(value).split(",") if v.strip()]
    return out or list(default)


def _gaussian_kernel2d(sigma: float, device, dtype, truncate: float = 3.0):
    sigma = max(float(sigma), 1e-3)
    radius = max(1, int(float(truncate) * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-((xx * xx + yy * yy) / (2.0 * sigma * sigma)))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)


def _conv2d_reflect(x: torch.Tensor, kernel: torch.Tensor):
    pad_h = kernel.shape[-2] // 2
    pad_w = kernel.shape[-1] // 2
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    return F.conv2d(x_pad, kernel)


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-6):
    bsz = x.shape[0]
    flat = x.flatten(1)
    mn = flat.min(dim=1).values.view(bsz, 1, 1, 1)
    mx = flat.max(dim=1).values.view(bsz, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)


def build_dog_log_saliency(
    images: torch.Tensor,
    dog_sigmas=None,
    log_sigmas=None,
    truncate: float = 3.0,
):
    """Build a bright-small-target saliency map with DoG/LoG band-pass filters.

    This function intentionally implements only multi-scale DoG and LoG. It
    does not include local contrast/CFAR/top-hat terms, so experiments can
    isolate the frequency/band-pass prior.
    """
    default_dog = ((0.7, 1.4), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0))
    default_log = (0.8, 1.2, 1.6, 2.4)
    dog_pairs = _parse_sigma_pairs(dog_sigmas, default_dog)
    log_scales = _parse_sigmas(log_sigmas, default_log)

    x = images.detach().float()
    if x.dim() != 4:
        raise ValueError("build_dog_log_saliency expects images with shape [B,C,H,W].")
    if x.shape[1] > 1:
        x = x.mean(dim=1, keepdim=True)
    x = _normalize_per_image(x)

    responses = []
    for sigma_small, sigma_large in dog_pairs:
        if sigma_large <= sigma_small:
            continue
        k_small = _gaussian_kernel2d(sigma_small, x.device, x.dtype, truncate=truncate)
        k_large = _gaussian_kernel2d(sigma_large, x.device, x.dtype, truncate=truncate)
        dog = _conv2d_reflect(x, k_small) - _conv2d_reflect(x, k_large)
        responses.append(F.relu(dog))

    lap = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    for sigma in log_scales:
        k = _gaussian_kernel2d(float(sigma), x.device, x.dtype, truncate=truncate)
        blur = _conv2d_reflect(x, k)
        log_resp = -(float(sigma) ** 2) * _conv2d_reflect(blur, lap)
        responses.append(F.relu(log_resp))

    if not responses:
        return x.new_zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]))
    saliency = torch.stack(responses, dim=0).amax(dim=0)
    return _normalize_per_image(saliency)


class SelfPromptingHead(nn.Module):
    """
    Lightweight detection head that predicts target heatmaps from
    the image encoder's feature map, then samples prompt points.

    Input:  encoder features  [B, C, h, w]  (typically [B, 256, 64, 64])
    Output: heatmap           [B, 1, H, W]  (original image resolution)
            point_coords      [B, 1, K, 2]  (x, y in pixel coords)
            point_labels      [B, 1, K]     (1=positive, 0=negative)
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        top_k_pos: int = 3,
        top_k_neg: int = 2,
        min_dist: int = 8,
        peak_thr: float = 0.1,
        low_response_thr: float = 0.3,
    ):
        super().__init__()
        self.top_k_pos = int(top_k_pos)
        self.top_k_neg = int(top_k_neg)
        self.min_dist = int(min_dist)
        self.peak_thr = float(peak_thr)
        self.low_response_thr = float(low_response_thr)

        # Lightweight segmentation head: 3-layer CNN.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """Small initialization to avoid disrupting pretrained encoder."""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Start with a conservative heatmap to limit early false alarms.
                    nn.init.constant_(m.bias, -2.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        output_size: tuple = None,
        gt_mask: torch.Tensor = None,
    ):
        """
        Args:
            encoder_features: [B, C, h, w] from image encoder neck
            output_size: (H, W) target resolution for heatmap, e.g. (256, 256)
            gt_mask: optional GT mask used during training for hard-negative mining
        Returns:
            heatmap:      [B, 1, H, W] sigmoid-activated probability map
            point_coords: [B, 1, K, 2] sampled prompt point coordinates (x, y)
            point_labels: [B, 1, K] point labels (1=pos, 0=neg)
        """
        logits = self.head(encoder_features)

        if output_size is not None:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )

        heatmap = torch.sigmoid(logits)
        point_coords, point_labels = self._sample_points(heatmap, gt_mask=gt_mask)
        return heatmap, point_coords, point_labels, logits

    @torch.no_grad()
    def _sample_points(self, heatmap: torch.Tensor, gt_mask: torch.Tensor = None):
        """
        Sample top-K positive and negative points from heatmap.

        Positive points: top-K peaks with NMS-like spacing.
        Negative points:
          - Training: high-response false-alarm peaks outside GT first.
          - Fallback / inference: low-response regions.
        """
        bsz, _, h, w = heatmap.shape
        device = heatmap.device
        total_k = self.top_k_pos + self.top_k_neg

        if gt_mask is not None:
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            if gt_mask.shape[-2:] != (h, w):
                gt_mask = F.interpolate(gt_mask.float(), size=(h, w), mode="nearest")
            gt_mask = gt_mask.float()

        all_coords = []
        all_labels = []
        for b in range(bsz):
            smap = heatmap[b, 0]
            gt_mask_b = gt_mask[b, 0] if gt_mask is not None else None

            peak_coords, _ = self._extract_peaks(
                smap,
                max_k=max(self.top_k_pos + self.top_k_neg * 4, self.top_k_pos),
            )
            pos_coords = peak_coords[: min(self.top_k_pos, peak_coords.shape[0])]
            neg_coords = self._sample_negatives(
                smap,
                self.top_k_neg,
                peak_coords=peak_coords,
                gt_mask=gt_mask_b,
                exclude_coords=pos_coords,
            )

            coords = torch.cat([pos_coords, neg_coords], dim=0)
            labels = torch.cat(
                [
                    torch.ones(pos_coords.shape[0], device=device, dtype=torch.long),
                    torch.zeros(neg_coords.shape[0], device=device, dtype=torch.long),
                ],
                dim=0,
            )

            if coords.shape[0] < total_k:
                pad_len = total_k - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
                labels = F.pad(labels, (0, pad_len), value=-1)
            coords = coords[:total_k]
            labels = labels[:total_k]

            all_coords.append(coords)
            all_labels.append(labels)

        point_coords = torch.stack(all_coords, dim=0).unsqueeze(1)
        point_labels = torch.stack(all_labels, dim=0).unsqueeze(1)
        return point_coords.float(), point_labels.int()

    def _extract_peaks(self, smap: torch.Tensor, max_k: int = None):
        """Extract sorted local maxima with NMS-like spacing."""
        _, w = smap.shape

        min_d = max(1, self.min_dist)
        if min_d % 2 == 0:
            min_d += 1
        padding = min_d // 2
        local_max = F.max_pool2d(
            smap.unsqueeze(0).unsqueeze(0),
            kernel_size=min_d,
            stride=1,
            padding=padding,
        ).squeeze(0).squeeze(0)

        is_peak = (smap == local_max) & (smap > self.peak_thr)
        peak_ys, peak_xs = torch.where(is_peak)
        if peak_xs.numel() == 0:
            flat_idx = torch.argmax(smap)
            y = flat_idx // w
            x = flat_idx % w
            coords = torch.stack([x, y], dim=0).unsqueeze(0).float()
            scores = smap[y, x].unsqueeze(0)
            return coords, scores

        vals = smap[peak_ys, peak_xs]
        n_select = vals.numel() if max_k is None else min(max_k, vals.numel())
        _, topk_idx = torch.topk(vals, n_select)
        coords = torch.stack([peak_xs[topk_idx].float(), peak_ys[topk_idx].float()], dim=1)
        scores = vals[topk_idx]
        return coords, scores

    def _sample_negatives(
        self,
        smap: torch.Tensor,
        k: int,
        peak_coords: torch.Tensor = None,
        gt_mask: torch.Tensor = None,
        exclude_coords: torch.Tensor = None,
    ):
        """Prefer hard negatives from high-response false alarms outside GT."""
        device = smap.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        if peak_coords is None:
            peak_coords, _ = self._extract_peaks(
                smap,
                max_k=max(k * 4, self.top_k_pos + k),
            )

        selected = torch.zeros((0, 2), device=device, dtype=torch.float32)
        if gt_mask is not None and peak_coords.numel() > 0:
            peak_x = peak_coords[:, 0].long().clamp(min=0, max=smap.shape[1] - 1)
            peak_y = peak_coords[:, 1].long().clamp(min=0, max=smap.shape[0] - 1)
            outside_gt = gt_mask[peak_y, peak_x] <= 0.5
            if exclude_coords is not None and exclude_coords.numel() > 0:
                same_x = peak_coords[:, None, 0] == exclude_coords[None, :, 0]
                same_y = peak_coords[:, None, 1] == exclude_coords[None, :, 1]
                outside_gt = outside_gt & (~(same_x & same_y).any(dim=1))
            hard_neg = peak_coords[outside_gt]
            if hard_neg.shape[0] > 0:
                selected = hard_neg[: min(k, hard_neg.shape[0])]

        if selected.shape[0] >= k:
            return selected[:k]

        if exclude_coords is not None and exclude_coords.numel() > 0:
            exclude_all = torch.cat([exclude_coords, selected], dim=0)
        else:
            exclude_all = selected
        fallback = self._sample_low_response_negatives(
            smap,
            k - selected.shape[0],
            exclude_coords=exclude_all,
        )
        if selected.numel() == 0:
            return fallback
        return torch.cat([selected, fallback], dim=0)

    def _sample_low_response_negatives(
        self,
        smap: torch.Tensor,
        k: int,
        exclude_coords: torch.Tensor = None,
    ):
        device = smap.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        low_mask = smap < self.low_response_thr
        if exclude_coords is not None and exclude_coords.numel() > 0:
            ex = exclude_coords.long()
            ex_x = ex[:, 0].clamp(min=0, max=smap.shape[1] - 1)
            ex_y = ex[:, 1].clamp(min=0, max=smap.shape[0] - 1)
            low_mask[ex_y, ex_x] = False
        low_ys, low_xs = torch.where(low_mask)

        if low_xs.numel() == 0:
            low_ys, low_xs = torch.where(smap < smap.max())
            if exclude_coords is not None and exclude_coords.numel() > 0 and low_xs.numel() > 0:
                keep = torch.ones_like(low_xs, dtype=torch.bool)
                ex = exclude_coords.long()
                for idx in range(ex.shape[0]):
                    keep = keep & ~((low_xs == ex[idx, 0]) & (low_ys == ex[idx, 1]))
                low_xs = low_xs[keep]
                low_ys = low_ys[keep]
            if low_xs.numel() == 0:
                return torch.zeros((0, 2), device=device, dtype=torch.float32)

        n_select = min(k, low_xs.numel())
        indices = torch.randperm(low_xs.numel(), device=device)[:n_select]
        return torch.stack([low_xs[indices].float(), low_ys[indices].float()], dim=1)


class BoundaryAwareSelfPromptingHead(nn.Module):
    """
    Predict prompt distributions distilled from GT boundary-prior sampling.

    The head predicts four heatmaps:
      - foreground: generic positive point distribution.
      - inner_boundary: positive points near the object-side boundary.
      - outer_boundary: negative points near the background-side boundary.
      - background: safe negative point distribution.

    During inference, point prompts are sampled from these predicted maps with
    the same positive/negative split implied by boundary_ratio.
    """

    map_names = ("foreground", "inner_boundary", "outer_boundary", "background")

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        top_k_pos: int = 4,
        top_k_neg: int = 4,
        boundary_ratio: float = 0.5,
        min_dist: int = 8,
        peak_thr: float = 0.05,
        low_response_thr: float = 0.3,
    ):
        super().__init__()
        self.top_k_pos = int(top_k_pos)
        self.top_k_neg = int(top_k_neg)
        self.boundary_ratio = float(boundary_ratio)
        self.min_dist = int(min_dist)
        self.peak_thr = float(peak_thr)
        self.low_response_thr = float(low_response_thr)
        self.pos_boundary_ratio = None
        self.neg_boundary_ratio = None
        self.fill_shortfall = False
        self.fill_pos_shortfall = None
        self.fill_neg_shortfall = None
        self.suppress_fg_from_inner = True
        self.positive_objectness_thr = 0.0
        self.pos_min_dist = None
        self.neg_min_dist = None
        self.neg_suppress_radius = None
        self.neg_fill_from_background = False
        self.component_aware_pos = False
        self.component_positive_mode = "full"
        self.component_score_mode = "foreground"
        self.component_rel_thr = 0.5
        self.component_abs_thr = 0.05
        self.component_min_area = 1
        self.positive_guidance_alpha = 0.75
        self.positive_guidance_bg_alpha = 0.5
        self.positive_guidance_power = 1.0
        self.last_point_types = None

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, len(self.map_names), kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, -2.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        final = self.head[-1]
        if isinstance(final, nn.Conv2d) and final.bias is not None:
            nn.init.constant_(final.bias, -2.0)
            with torch.no_grad():
                final.bias[3].fill_(0.0)

    def forward(
        self,
        encoder_features: torch.Tensor,
        output_size: tuple = None,
        gt_mask: torch.Tensor = None,
        positive_guidance: torch.Tensor = None,
    ):
        logits = self.head(encoder_features)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        prompt_maps = torch.sigmoid(logits)
        if positive_guidance is not None:
            if positive_guidance.dim() == 3:
                positive_guidance = positive_guidance.unsqueeze(1)
            if positive_guidance.shape[-2:] != prompt_maps.shape[-2:]:
                positive_guidance = F.interpolate(
                    positive_guidance.float(),
                    size=prompt_maps.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            positive_guidance = _normalize_per_image(positive_guidance.float()).to(
                device=prompt_maps.device,
                dtype=prompt_maps.dtype,
            )
        point_coords, point_labels = self._sample_points(prompt_maps, positive_guidance=positive_guidance)
        prompt_maps_dict = {
            name: prompt_maps[:, idx : idx + 1] for idx, name in enumerate(self.map_names)
        }
        logits_dict = {
            name: logits[:, idx : idx + 1] for idx, name in enumerate(self.map_names)
        }
        return prompt_maps_dict, point_coords, point_labels, logits_dict

    @torch.no_grad()
    def _sample_points(self, prompt_maps: torch.Tensor, positive_guidance: torch.Tensor = None):
        bsz, _, h, w = prompt_maps.shape
        device = prompt_maps.device
        total_k = self.top_k_pos + self.top_k_neg

        pos_ratio = self.boundary_ratio if self.pos_boundary_ratio is None else float(self.pos_boundary_ratio)
        neg_ratio = self.boundary_ratio if self.neg_boundary_ratio is None else float(self.neg_boundary_ratio)
        pos_boundary_k = int(max(0, min(self.top_k_pos, self.top_k_pos * pos_ratio)))
        neg_boundary_k = int(max(0, min(self.top_k_neg, self.top_k_neg * neg_ratio)))
        pos_fg_k = max(0, self.top_k_pos - pos_boundary_k)
        neg_bg_k = max(0, self.top_k_neg - neg_boundary_k)

        all_coords = []
        all_labels = []
        all_types = []
        for b in range(bsz):
            fg = prompt_maps[b, 0]
            inner = prompt_maps[b, 1]
            outer = prompt_maps[b, 2]
            bg = prompt_maps[b, 3]
            if positive_guidance is not None:
                fg = self._apply_positive_guidance(fg, bg, positive_guidance[b, 0])

            selected_parts = []
            exclude = None
            objectness_thr = float(getattr(self, "positive_objectness_thr", 0.0))
            suppress_fg = bool(getattr(self, "suppress_fg_from_inner", True))
            pos_min_dist = getattr(self, "pos_min_dist", None)
            neg_min_dist = getattr(self, "neg_min_dist", None)
            neg_suppress_radius = getattr(self, "neg_suppress_radius", None)
            fill_pos = getattr(self, "fill_pos_shortfall", None)
            fill_neg = getattr(self, "fill_neg_shortfall", None)
            allow_positive = True
            if objectness_thr > 0.0:
                allow_positive = bool(torch.maximum(fg.max(), inner.max()).item() >= objectness_thr)

            if allow_positive:
                if bool(getattr(self, "component_aware_pos", False)):
                    component_positive_mode = str(getattr(self, "component_positive_mode", "full")).lower()
                    if component_positive_mode in ("foreground_slots", "fg_slots", "fg_only"):
                        pos_inner = self._select_points(
                            inner,
                            pos_boundary_k,
                            exclude_coords=exclude,
                            min_dist=pos_min_dist,
                            fill_shortfall=fill_pos,
                        )
                        if pos_inner.numel() > 0:
                            selected_parts.append(pos_inner)
                            exclude = pos_inner
                        fg_for_components = (
                            self._suppress_coords(fg, pos_inner, radius=0)
                            if pos_inner.numel() > 0
                            else fg
                        )
                        pos_fg = self._select_component_points(fg_for_components, inner, pos_fg_k)
                        if pos_fg.shape[0] < pos_fg_k:
                            fill = self._select_points(
                                fg,
                                pos_fg_k - pos_fg.shape[0],
                                exclude_coords=pos_fg if pos_fg.numel() > 0 else None,
                                min_dist=pos_min_dist,
                                fill_shortfall=True if fill_pos is None else fill_pos,
                            )
                            pos_fg = self._cat_nonempty([pos_fg, fill], device=device)
                        pos_coords = self._cat_nonempty([pos_inner, pos_fg[:pos_fg_k]], device=device)
                    else:
                        pos_inner = torch.zeros((0, 2), device=device, dtype=torch.float32)
                        pos_fg = self._select_component_points(fg, inner, self.top_k_pos)
                        if pos_fg.shape[0] < self.top_k_pos:
                            fill = self._select_points(
                                fg,
                                self.top_k_pos - pos_fg.shape[0],
                                exclude_coords=pos_fg if pos_fg.numel() > 0 else None,
                                min_dist=pos_min_dist,
                                fill_shortfall=True if fill_pos is None else fill_pos,
                            )
                            pos_fg = self._cat_nonempty([pos_fg, fill], device=device)
                        pos_coords = pos_fg[: self.top_k_pos]
                    if pos_fg.numel() > 0:
                        selected_parts.append(pos_fg[: max(0, self.top_k_pos - pos_inner.shape[0])])
                    if pos_coords.shape[0] < self.top_k_pos:
                        fill = self._select_points(
                            fg,
                            self.top_k_pos - pos_coords.shape[0],
                            exclude_coords=pos_coords if pos_coords.numel() > 0 else None,
                            min_dist=pos_min_dist,
                            fill_shortfall=True if fill_pos is None else fill_pos,
                        )
                        pos_coords = self._cat_nonempty([pos_coords, fill], device=device)
                    pos_coords = pos_coords[: self.top_k_pos]
                    if pos_coords.numel() > 0:
                        selected_parts = [pos_coords]
                        exclude = pos_coords
                else:
                    pos_inner = self._select_points(
                        inner,
                        pos_boundary_k,
                        exclude_coords=exclude,
                        min_dist=pos_min_dist,
                        fill_shortfall=fill_pos,
                    )
                    if pos_inner.numel() > 0:
                        selected_parts.append(pos_inner)
                        exclude = pos_inner

                    fg_exclude = exclude if suppress_fg else None
                    pos_fg = self._select_points(
                        fg,
                        pos_fg_k,
                        exclude_coords=fg_exclude,
                        min_dist=pos_min_dist,
                        fill_shortfall=fill_pos,
                    )
                    pos_coords = self._cat_nonempty([pos_inner, pos_fg], device=device)
                    if pos_fg.numel() > 0:
                        selected_parts.append(pos_fg)
                    if selected_parts:
                        exclude = self._cat_nonempty(selected_parts, device=device)
            else:
                pos_inner = torch.zeros((0, 2), device=device, dtype=torch.float32)
                pos_fg = torch.zeros((0, 2), device=device, dtype=torch.float32)
                pos_coords = torch.zeros((0, 2), device=device, dtype=torch.float32)

            inner_count = min(int(pos_inner.shape[0]), int(pos_coords.shape[0]))
            fg_count = max(0, int(pos_coords.shape[0]) - inner_count)
            pos_types = torch.cat(
                [
                    torch.full((inner_count,), 1, device=device, dtype=torch.long),
                    torch.full((fg_count,), 0, device=device, dtype=torch.long),
                ],
                dim=0,
            )

            neg_outer = self._select_points(
                outer,
                neg_boundary_k,
                exclude_coords=exclude,
                min_dist=neg_min_dist,
                suppress_radius=neg_suppress_radius,
                fill_shortfall=fill_neg,
            )
            if neg_outer.numel() > 0:
                selected_parts.append(neg_outer)
                exclude = self._cat_nonempty(selected_parts, device=device)

            neg_bg = self._select_points(
                bg,
                neg_bg_k,
                exclude_coords=exclude,
                min_dist=neg_min_dist,
                suppress_radius=neg_suppress_radius,
                fill_shortfall=fill_neg,
            )
            neg_coords = self._cat_nonempty([neg_outer, neg_bg], device=device)
            if bool(getattr(self, "neg_fill_from_background", False)) and neg_coords.shape[0] < self.top_k_neg:
                bg_exclude = self._cat_nonempty(selected_parts + [neg_bg], device=device)
                extra_bg = self._select_points(
                    bg,
                    self.top_k_neg - neg_coords.shape[0],
                    exclude_coords=bg_exclude,
                    min_dist=neg_min_dist,
                    suppress_radius=neg_suppress_radius,
                    fill_shortfall=True,
                )
                neg_coords = self._cat_nonempty([neg_coords, extra_bg], device=device)

            outer_count = min(int(neg_outer.shape[0]), int(neg_coords.shape[0]))
            bg_count = max(0, int(neg_coords.shape[0]) - outer_count)
            neg_types = torch.cat(
                [
                    torch.full((outer_count,), 2, device=device, dtype=torch.long),
                    torch.full((bg_count,), 3, device=device, dtype=torch.long),
                ],
                dim=0,
            )

            coords = self._cat_nonempty([pos_coords, neg_coords], device=device)
            labels = torch.cat(
                [
                    torch.ones(pos_coords.shape[0], device=device, dtype=torch.long),
                    torch.zeros(neg_coords.shape[0], device=device, dtype=torch.long),
                ],
                dim=0,
            )
            point_types = torch.cat([pos_types, neg_types], dim=0)

            if coords.shape[0] < total_k:
                pad_len = total_k - coords.shape[0]
                coords = F.pad(coords, (0, 0, 0, pad_len), value=0.0)
                labels = F.pad(labels, (0, pad_len), value=-1)
                point_types = F.pad(point_types, (0, pad_len), value=-1)
            coords = coords[:total_k]
            labels = labels[:total_k]
            point_types = point_types[:total_k]
            all_coords.append(coords)
            all_labels.append(labels)
            all_types.append(point_types)

        point_coords = torch.stack(all_coords, dim=0).unsqueeze(1)
        point_labels = torch.stack(all_labels, dim=0).unsqueeze(1)
        point_types = torch.stack(all_types, dim=0).unsqueeze(1)
        self.last_point_types = point_types.int()
        return point_coords.float(), point_labels.int()

    def _cat_nonempty(self, tensors, device):
        nonempty = [t for t in tensors if t is not None and t.numel() > 0]
        if not nonempty:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)
        return torch.cat(nonempty, dim=0)

    def _apply_positive_guidance(self, fg: torch.Tensor, bg: torch.Tensor, guidance: torch.Tensor):
        guidance = guidance.to(device=fg.device, dtype=fg.dtype).clamp(0.0, 1.0)
        alpha = max(0.0, min(1.0, float(getattr(self, "positive_guidance_alpha", 0.75))))
        bg_alpha = max(0.0, min(1.0, float(getattr(self, "positive_guidance_bg_alpha", 0.5))))
        power = max(1e-3, float(getattr(self, "positive_guidance_power", 1.0)))
        guided = fg * ((1.0 - alpha) + alpha * guidance.pow(power))
        if bg_alpha > 0.0:
            guided = guided * (1.0 - bg_alpha * bg.clamp(0.0, 1.0)).clamp_min(0.0)
        return guided

    def _select_component_points(self, fg: torch.Tensor, inner: torch.Tensor, k: int):
        device = fg.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        mode = str(getattr(self, "component_score_mode", "foreground")).lower()
        if mode in ("fg_inner", "foreground_inner", "max"):
            comp_score = torch.maximum(fg, inner)
            pick_score = fg
        elif mode in ("inner", "inner_boundary"):
            comp_score = inner
            pick_score = inner
        else:
            comp_score = fg
            pick_score = fg

        max_score = float(comp_score.max().item())
        if max_score <= self.peak_thr:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        rel_thr = float(getattr(self, "component_rel_thr", 0.5))
        abs_thr = float(getattr(self, "component_abs_thr", self.peak_thr))
        threshold = max(abs_thr, max_score * rel_thr)
        mask_np = (comp_score.detach().float().cpu().numpy() >= threshold)
        if not mask_np.any():
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        pick_np = pick_score.detach().float().cpu().numpy()
        comp_np = comp_score.detach().float().cpu().numpy()
        visited = np.zeros(mask_np.shape, dtype=bool)
        h, w = mask_np.shape
        min_area = max(1, int(getattr(self, "component_min_area", 1)))
        candidates = []
        for y0, x0 in np.argwhere(mask_np):
            y0 = int(y0)
            x0 = int(x0)
            if visited[y0, x0]:
                continue
            stack = [(y0, x0)]
            visited[y0, x0] = True
            pixels = []
            while stack:
                y, x = stack.pop()
                pixels.append((y, x))
                for yy in range(max(0, y - 1), min(h, y + 2)):
                    for xx in range(max(0, x - 1), min(w, x + 2)):
                        if yy == y and xx == x:
                            continue
                        if mask_np[yy, xx] and not visited[yy, xx]:
                            visited[yy, xx] = True
                            stack.append((yy, xx))

            if len(pixels) < min_area:
                continue
            ys = np.fromiter((p[0] for p in pixels), dtype=np.int64)
            xs = np.fromiter((p[1] for p in pixels), dtype=np.int64)
            values = pick_np[ys, xs]
            best_idx = int(values.argmax())
            best_y = int(ys[best_idx])
            best_x = int(xs[best_idx])
            comp_peak = float(comp_np[ys, xs].max())
            candidates.append((comp_peak, float(values[best_idx]), len(pixels), best_x, best_y))

        if not candidates:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        coords = [[float(x), float(y)] for _, _, _, x, y in candidates[:k]]
        return torch.tensor(coords, device=device, dtype=torch.float32)

    def _select_points(
        self,
        score_map: torch.Tensor,
        k: int,
        exclude_coords: torch.Tensor = None,
        min_dist: int = None,
        suppress_radius: int = None,
        fill_shortfall: bool = None,
    ):
        device = score_map.device
        if k <= 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)

        scores = score_map.clone()
        min_d = max(1, int(self.min_dist if min_dist is None else min_dist))
        if exclude_coords is not None and exclude_coords.numel() > 0:
            radius = max(1, min_d // 2) if suppress_radius is None else max(0, int(suppress_radius))
            scores = self._suppress_coords(scores, exclude_coords, radius=radius)

        if min_d % 2 == 0:
            min_d += 1
        padding = min_d // 2
        local_max = F.max_pool2d(scores.unsqueeze(0).unsqueeze(0), kernel_size=min_d, stride=1, padding=padding).squeeze(0).squeeze(0)
        valid = (scores == local_max) & (scores > self.peak_thr)
        ys, xs = torch.where(valid)

        if xs.numel() == 0:
            return self._topk_from_flat(scores, k)

        vals = scores[ys, xs]
        n_select = min(k, vals.numel())
        _, order = torch.topk(vals, n_select)
        selected = torch.stack([xs[order].float(), ys[order].float()], dim=1)
        do_fill = bool(getattr(self, "fill_shortfall", False) if fill_shortfall is None else fill_shortfall)
        if do_fill and selected.shape[0] < k:
            fill_scores = self._suppress_coords(scores, selected, radius=0)
            fill = self._topk_from_flat(fill_scores, k - selected.shape[0])
            selected = self._cat_nonempty([selected, fill], device=device)
        return selected

    def _topk_from_flat(self, scores: torch.Tensor, k: int):
        device = scores.device
        flat = scores.flatten()
        finite = torch.isfinite(flat)
        if not finite.any():
            return torch.zeros((0, 2), device=device, dtype=torch.float32)
        n_select = min(k, int(finite.sum().item()))
        vals, idx = torch.topk(flat, n_select)
        keep = torch.isfinite(vals)
        idx = idx[keep]
        if idx.numel() == 0:
            return torch.zeros((0, 2), device=device, dtype=torch.float32)
        h, w = scores.shape
        ys = idx // w
        xs = idx % w
        return torch.stack([xs.float(), ys.float()], dim=1)

    def _suppress_coords(self, scores: torch.Tensor, coords: torch.Tensor, radius: int):
        h, w = scores.shape
        out = scores.clone()
        for xy in coords.long():
            x = int(xy[0].clamp(min=0, max=w - 1).item())
            y = int(xy[1].clamp(min=0, max=h - 1).item())
            x1, x2 = max(0, x - radius), min(w, x + radius + 1)
            y1, y2 = max(0, y - radius), min(h, y + radius + 1)
            out[y1:y2, x1:x2] = -float("inf")
        return out


def _ensure_mask4d(gt_mask: torch.Tensor) -> torch.Tensor:
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    return gt_mask.float()


def _dilate_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(1, int(radius))
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def _erode_mask(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    radius = max(1, int(radius))
    k = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius)


def _soft_inner_boundary_band(gt: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
    """Soft object-side boundary band by peeling the mask inward."""
    radius = max(1, int(radius))
    sigma = max(float(sigma), 1e-6)
    remaining = gt
    band = torch.zeros_like(gt)
    for dist in range(radius):
        eroded = _erode_mask(remaining, radius=1)
        ring = (remaining - eroded).clamp(min=0.0, max=1.0)
        weight = float(torch.exp(torch.tensor(-float(dist) / sigma, device=gt.device, dtype=gt.dtype)).item())
        band = torch.maximum(band, ring * weight)
        remaining = eroded
    return band * gt


def _soft_outer_boundary_band(gt: torch.Tensor, radius: int, sigma: float) -> torch.Tensor:
    """Soft background-side boundary band by expanding the mask outward."""
    radius = max(1, int(radius))
    sigma = max(float(sigma), 1e-6)
    prev = gt
    band = torch.zeros_like(gt)
    for dist in range(radius):
        dil = _dilate_mask(prev, radius=1)
        ring = (dil - prev).clamp(min=0.0, max=1.0)
        weight = float(torch.exp(torch.tensor(-float(dist) / sigma, device=gt.device, dtype=gt.dtype)).item())
        band = torch.maximum(band, ring * weight)
        prev = dil
    return band * (1.0 - gt)


def _soft_safe_background(gt: torch.Tensor, margin: int, fade_radius: int) -> torch.Tensor:
    """Safe-background target that ramps up after a no-sample margin."""
    margin = max(0, int(margin))
    fade_radius = max(1, int(fade_radius))
    unsafe = _dilate_mask(gt, radius=margin) if margin > 0 else gt
    bg = torch.zeros_like(gt)
    prev = unsafe
    for dist in range(fade_radius):
        dil = _dilate_mask(prev, radius=1)
        ring = (dil - prev).clamp(min=0.0, max=1.0)
        weight = float(dist + 1) / float(fade_radius)
        bg = torch.maximum(bg, ring * weight)
        prev = dil
    far_bg = (1.0 - prev).clamp(min=0.0, max=1.0)
    bg = torch.maximum(bg, far_bg)
    return bg * (1.0 - gt)


def build_boundary_prompt_targets(
    gt_mask: torch.Tensor,
    boundary_width: int = 1,
    background_margin: int = 3,
    target_mode: str = "soft",
    foreground_mode: str = "mask",
    soft_band_radius: int = 3,
    soft_sigma: float = 1.5,
    background_fade_radius: int = 4,
):
    gt = (_ensure_mask4d(gt_mask) > 0.5).float()
    mode = str(target_mode).lower()
    if mode == "soft":
        if str(foreground_mode).lower() == "soft_mask":
            foreground = _dilate_mask(gt, radius=1) * 0.25
            foreground = torch.maximum(foreground, gt)
        else:
            foreground = gt
        return {
            "foreground": foreground.clamp(min=0.0, max=1.0),
            "inner_boundary": _soft_inner_boundary_band(gt, radius=soft_band_radius, sigma=soft_sigma),
            "outer_boundary": _soft_outer_boundary_band(gt, radius=soft_band_radius, sigma=soft_sigma),
            "background": _soft_safe_background(
                gt,
                margin=background_margin,
                fade_radius=background_fade_radius,
            ).clamp(min=0.0, max=1.0),
        }

    bw = max(1, int(boundary_width))
    k = 2 * bw + 1
    dil = F.max_pool2d(gt, kernel_size=k, stride=1, padding=bw)
    erode = 1.0 - F.max_pool2d(1.0 - gt, kernel_size=k, stride=1, padding=bw)
    boundary = (dil - erode).clamp(min=0.0, max=1.0)
    inner_boundary = boundary * gt
    outer_boundary = boundary * (1.0 - gt)

    margin = max(0, int(background_margin))
    if margin > 0:
        mk = 2 * margin + 1
        unsafe_bg = F.max_pool2d(gt, kernel_size=mk, stride=1, padding=margin)
        background = 1.0 - unsafe_bg
    else:
        background = 1.0 - gt

    return {
        "foreground": gt,
        "inner_boundary": inner_boundary,
        "outer_boundary": outer_boundary,
        "background": background.clamp(min=0.0, max=1.0),
    }


def _normalize_boundary_prompt_channels(active_channels):
    if active_channels is None:
        return set(BoundaryAwareSelfPromptingHead.map_names)
    if isinstance(active_channels, str):
        spec = active_channels.strip().lower()
        aliases = {
            "all": BoundaryAwareSelfPromptingHead.map_names,
            "fg_bg": ("foreground", "background"),
            "fgbg": ("foreground", "background"),
            "foreground_background": ("foreground", "background"),
            "fg_bg_outer": ("foreground", "outer_boundary", "background"),
            "fgbg_outer": ("foreground", "outer_boundary", "background"),
            "foreground_background_outer": ("foreground", "outer_boundary", "background"),
        }
        if spec in aliases:
            return set(aliases[spec])
        tokens = [tok.strip().lower() for tok in spec.replace(";", ",").split(",") if tok.strip()]
    else:
        tokens = [str(tok).strip().lower() for tok in active_channels]

    name_alias = {
        "fg": "foreground",
        "foreground": "foreground",
        "inner": "inner_boundary",
        "inner_boundary": "inner_boundary",
        "outer": "outer_boundary",
        "outer_boundary": "outer_boundary",
        "bg": "background",
        "background": "background",
    }
    channels = set()
    for token in tokens:
        if token not in name_alias:
            raise ValueError(f"Unknown boundary prompt channel: {token}")
        channels.add(name_alias[token])
    if not channels:
        return set(BoundaryAwareSelfPromptingHead.map_names)
    return channels


def boundary_aware_self_prompt_loss(
    prompt_logits,
    gt_mask: torch.Tensor,
    pos_weight: float = 10.0,
    boundary_pos_weight: float = None,
    background_pos_weight: float = 1.0,
    boundary_width: int = 1,
    background_margin: int = 3,
    background_loss_weight: float = 0.25,
    target_mode: str = "soft",
    foreground_mode: str = "mask",
    soft_band_radius: int = 3,
    soft_sigma: float = 1.5,
    background_fade_radius: int = 4,
    active_channels=None,
):
    """Supervise boundary-aware prompt maps derived from a GT mask."""
    if isinstance(prompt_logits, dict):
        logits_by_name = prompt_logits
        ref = next(iter(prompt_logits.values()))
    else:
        ref = prompt_logits
        logits_by_name = {
            name: prompt_logits[:, idx : idx + 1]
            for idx, name in enumerate(BoundaryAwareSelfPromptingHead.map_names)
        }

    gt = _ensure_mask4d(gt_mask)
    if gt.shape[-2:] != ref.shape[-2:]:
        gt = F.interpolate(gt, size=ref.shape[-2:], mode="nearest")

    targets = build_boundary_prompt_targets(
        gt,
        boundary_width=boundary_width,
        background_margin=background_margin,
        target_mode=target_mode,
        foreground_mode=foreground_mode,
        soft_band_radius=soft_band_radius,
        soft_sigma=soft_sigma,
        background_fade_radius=background_fade_radius,
    )
    boundary_pos_weight = float(pos_weight if boundary_pos_weight is None else boundary_pos_weight)
    specs = [
        ("foreground", float(pos_weight), 1.0),
        ("inner_boundary", boundary_pos_weight, 1.0),
        ("outer_boundary", boundary_pos_weight, 1.0),
        ("background", float(background_pos_weight), float(background_loss_weight)),
    ]
    enabled_channels = _normalize_boundary_prompt_channels(active_channels)
    total = ref.new_tensor(0.0)
    denom = 0.0
    for name, ch_pos_weight, ch_loss_weight in specs:
        if name not in enabled_channels:
            continue
        logit = logits_by_name[name]
        target = targets[name].to(device=logit.device, dtype=logit.dtype)
        weight = torch.ones_like(target)
        weight[target > 0.5] = ch_pos_weight
        total = total + ch_loss_weight * F.binary_cross_entropy_with_logits(
            logit,
            target,
            weight=weight,
            reduction="mean",
        )
        denom += ch_loss_weight
    return total / max(denom, 1e-6)


def _gt_connected_components(mask_np: np.ndarray, min_area: int = 1, max_components: int = 256):
    mask_np = np.asarray(mask_np, dtype=bool)
    visited = np.zeros(mask_np.shape, dtype=bool)
    h, w = mask_np.shape
    min_area = max(1, int(min_area))
    max_components = max(1, int(max_components))
    comps = []
    for y0, x0 in np.argwhere(mask_np):
        y0 = int(y0)
        x0 = int(x0)
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        ys = []
        xs = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    if yy == y and xx == x:
                        continue
                    if mask_np[yy, xx] and not visited[yy, xx]:
                        visited[yy, xx] = True
                        stack.append((yy, xx))
        if len(ys) >= min_area:
            comps.append((np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)))
            if len(comps) >= max_components:
                break
    return comps


def foreground_component_peak_loss(
    prompt_logits,
    gt_mask: torch.Tensor,
    temperature: float = 0.5,
    min_area: int = 1,
    max_components: int = 256,
):
    """
    Encourage every GT component to contain at least one high foreground peak.

    This is a differentiable proxy for component-aware positive prompt coverage:
    connected components are computed from the GT mask only, while the smooth
    peak inside each component is computed from foreground logits and receives
    gradients.
    """
    logits_by_name = _boundary_logits_dict(prompt_logits)
    fg_logits = logits_by_name["foreground"]
    gt = (_ensure_mask4d(gt_mask) > 0.5).float()
    if gt.shape[-2:] != fg_logits.shape[-2:]:
        gt = F.interpolate(gt, size=fg_logits.shape[-2:], mode="nearest")

    tau = max(float(temperature), 1e-6)
    total = fg_logits.new_tensor(0.0)
    count = 0
    for b in range(fg_logits.shape[0]):
        comps = _gt_connected_components(
            gt[b, 0].detach().cpu().numpy() > 0.5,
            min_area=min_area,
            max_components=max_components,
        )
        for ys_np, xs_np in comps:
            ys = torch.as_tensor(ys_np, device=fg_logits.device, dtype=torch.long)
            xs = torch.as_tensor(xs_np, device=fg_logits.device, dtype=torch.long)
            vals = fg_logits[b, 0, ys, xs].float()
            if vals.numel() == 0:
                continue
            if vals.numel() == 1:
                peak_logit = vals[0]
            else:
                peak_logit = tau * torch.logsumexp(vals / tau, dim=0) - tau * torch.log(
                    vals.new_tensor(float(vals.numel()))
                )
            target = torch.ones_like(peak_logit)
            total = total + F.binary_cross_entropy_with_logits(peak_logit, target, reduction="mean")
            count += 1
    if count <= 0:
        return fg_logits.new_tensor(0.0)
    return total / float(count)


def _boundary_logits_dict(prompt_logits):
    if isinstance(prompt_logits, dict):
        return prompt_logits
    return {
        name: prompt_logits[:, idx : idx + 1]
        for idx, name in enumerate(BoundaryAwareSelfPromptingHead.map_names)
    }


def _gather_map_at_points(value_map: torch.Tensor, point_coords: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor gather from [B,1,H,W] maps at [B,1,K,2] xy coords."""
    if point_coords.dim() == 4:
        point_coords = point_coords.squeeze(1)
    bsz, _, h, w = value_map.shape
    coords = point_coords.detach()
    x = coords[..., 0].round().long().clamp(0, w - 1)
    y = coords[..., 1].round().long().clamp(0, h - 1)
    flat_idx = y * w + x
    flat = value_map[:, 0].reshape(bsz, h * w)
    return torch.gather(flat, dim=1, index=flat_idx)


def boundary_aware_sampled_point_loss(
    prompt_logits,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
):
    """
    Auxiliary loss on the actual sampled prompts.

    The top-k coordinates are detached by the sampler, but the logits at those
    coordinates still receive gradients. Positive prompts are rewarded only when
    they fall inside GT; negative prompts are rewarded only when they fall outside.
    """
    logits_by_name = _boundary_logits_dict(prompt_logits)
    ref = next(iter(logits_by_name.values()))
    gt = (_ensure_mask4d(gt_mask) > 0.5).float()
    if gt.shape[-2:] != ref.shape[-2:]:
        gt = F.interpolate(gt, size=ref.shape[-2:], mode="nearest")

    if point_labels.dim() == 3:
        labels = point_labels.squeeze(1)
    else:
        labels = point_labels
    labels = labels.to(device=ref.device)
    coords = point_coords.to(device=ref.device, dtype=ref.dtype)

    pos_logit_map = torch.maximum(logits_by_name["foreground"], logits_by_name["inner_boundary"])
    neg_logit_map = torch.maximum(logits_by_name["outer_boundary"], logits_by_name["background"])
    gt_at_points = _gather_map_at_points(gt, coords)
    pos_logits = _gather_map_at_points(pos_logit_map, coords)
    neg_logits = _gather_map_at_points(neg_logit_map, coords)

    valid = labels >= 0
    pos_valid = valid & (labels == 1)
    neg_valid = valid & (labels == 0)
    total = ref.new_tensor(0.0)
    denom = 0.0
    if pos_valid.any():
        pos_target = gt_at_points[pos_valid]
        total = total + float(pos_weight) * F.binary_cross_entropy_with_logits(
            pos_logits[pos_valid],
            pos_target.to(dtype=pos_logits.dtype),
            reduction="mean",
        )
        denom += float(pos_weight)
    if neg_valid.any():
        neg_target = 1.0 - gt_at_points[neg_valid]
        total = total + float(neg_weight) * F.binary_cross_entropy_with_logits(
            neg_logits[neg_valid],
            neg_target.to(dtype=neg_logits.dtype),
            reduction="mean",
        )
        denom += float(neg_weight)
    if denom <= 0.0:
        return ref.new_tensor(0.0)
    return total / denom


def boundary_aware_channel_sampled_point_loss(
    prompt_logits,
    point_coords: torch.Tensor,
    point_labels: torch.Tensor,
    point_types: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    boundary_width: int = 1,
    background_margin: int = 3,
    target_mode: str = "hard",
    foreground_mode: str = "mask",
    soft_band_radius: int = 3,
    soft_sigma: float = 1.5,
    background_fade_radius: int = 4,
    error_only: bool = True,
    target_threshold: float = 0.5,
):
    """
    Channel-aware loss on sampled prompt points.

    Point labels only encode SAM semantics, but point_types encode where the
    point was sampled from: 0=foreground, 1=inner_boundary, 2=outer_boundary,
    3=background. In error-only mode, already-correct points do not keep
    pushing logits upward; only points outside their channel target are penalized.
    """
    logits_by_name = _boundary_logits_dict(prompt_logits)
    ref = next(iter(logits_by_name.values()))
    gt = (_ensure_mask4d(gt_mask) > 0.5).float()
    if gt.shape[-2:] != ref.shape[-2:]:
        gt = F.interpolate(gt, size=ref.shape[-2:], mode="nearest")

    targets = build_boundary_prompt_targets(
        gt,
        boundary_width=boundary_width,
        background_margin=background_margin,
        target_mode=target_mode,
        foreground_mode=foreground_mode,
        soft_band_radius=soft_band_radius,
        soft_sigma=soft_sigma,
        background_fade_radius=background_fade_radius,
    )

    if point_labels.dim() == 3:
        labels = point_labels.squeeze(1)
    else:
        labels = point_labels
    if point_types.dim() == 3:
        types = point_types.squeeze(1)
    else:
        types = point_types
    labels = labels.to(device=ref.device)
    types = types.to(device=ref.device)
    coords = point_coords.to(device=ref.device, dtype=ref.dtype)

    type_specs = [
        (0, "foreground", float(pos_weight)),
        (1, "inner_boundary", float(pos_weight)),
        (2, "outer_boundary", float(neg_weight)),
        (3, "background", float(neg_weight)),
    ]
    valid = labels >= 0
    total = ref.new_tensor(0.0)
    denom = ref.new_tensor(0.0)
    thr = float(target_threshold)

    for type_id, name, channel_weight in type_specs:
        selected = valid & (types == int(type_id))
        if not selected.any() or channel_weight <= 0.0:
            continue
        channel_logits = _gather_map_at_points(logits_by_name[name], coords)
        channel_target = _gather_map_at_points(
            targets[name].to(device=ref.device, dtype=ref.dtype),
            coords,
        )
        logits_sel = channel_logits[selected]
        target_sel = channel_target[selected]
        if bool(error_only):
            wrong = target_sel <= thr
            if not wrong.any():
                continue
            logits_sel = logits_sel[wrong]
            target_sel = target_sel[wrong]
        loss_vec = F.binary_cross_entropy_with_logits(
            logits_sel,
            target_sel.to(dtype=logits_sel.dtype),
            reduction="none",
        )
        total = total + channel_weight * loss_vec.sum()
        denom = denom + ref.new_tensor(channel_weight * max(1, int(loss_vec.numel())))

    if float(denom.item()) <= 0.0:
        return ref.new_tensor(0.0)
    return total / denom


def boundary_aware_expected_hit_loss(
    prompt_logits,
    gt_mask: torch.Tensor,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    eps: float = 1e-6,
):
    """
    Soft distribution loss for prompt maps.

    Positive-map probability mass should concentrate inside GT. Negative-map
    probability mass should concentrate outside GT. Empty-GT patches suppress
    positive-map responses instead of taking an undefined inside ratio.
    """
    logits_by_name = _boundary_logits_dict(prompt_logits)
    ref = next(iter(logits_by_name.values()))
    gt = (_ensure_mask4d(gt_mask) > 0.5).float()
    if gt.shape[-2:] != ref.shape[-2:]:
        gt = F.interpolate(gt, size=ref.shape[-2:], mode="nearest")

    pos_score = torch.sigmoid(logits_by_name["foreground"]) + torch.sigmoid(logits_by_name["inner_boundary"])
    neg_score = torch.sigmoid(logits_by_name["outer_boundary"]) + torch.sigmoid(logits_by_name["background"])
    gt = gt.to(device=ref.device, dtype=ref.dtype)
    bg = 1.0 - gt

    dims = (1, 2, 3)
    gt_area = gt.sum(dim=dims)
    nonempty = gt_area > 0.5
    total = ref.new_tensor(0.0)
    denom = 0.0

    if nonempty.any():
        pos_score_ne = pos_score[nonempty]
        gt_ne = gt[nonempty]
        pos_mass = (pos_score_ne * gt_ne).sum(dim=dims) / (pos_score_ne.sum(dim=dims) + eps)
        total = total + float(pos_weight) * (-torch.log(pos_mass.clamp_min(eps))).mean()
        denom += float(pos_weight)
    if (~nonempty).any():
        total = total + float(pos_weight) * pos_score[(~nonempty)].mean()
        denom += float(pos_weight)

    neg_mass = (neg_score * bg).sum(dim=dims) / (neg_score.sum(dim=dims) + eps)
    total = total + float(neg_weight) * (-torch.log(neg_mass.clamp_min(eps))).mean()
    denom += float(neg_weight)

    return total / max(denom, 1e-6)


class MaskGuidedBoundarySelfPromptingHead(BoundaryAwareSelfPromptingHead):
    """Boundary-aware self-prompt head conditioned on a first-stage mask.

    Stage 1 produces a coarse mask with ordinary sparse prompts. This head
    receives the image embedding plus compact mask evidence and predicts the
    second-round prompt maps. The prompt encoder/mask decoder remain unchanged;
    only the prompt generator receives mask feedback.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        top_k_pos: int = 4,
        top_k_neg: int = 4,
        boundary_ratio: float = 0.5,
        min_dist: int = 8,
        peak_thr: float = 0.05,
        low_response_thr: float = 0.3,
        mask_feature_channels: int = 3,
        detach_mask: bool = True,
    ):
        self.image_channels = int(in_channels)
        self.mask_feature_channels = max(1, min(3, int(mask_feature_channels)))
        self.detach_mask = bool(detach_mask)
        super().__init__(
            in_channels=self.image_channels + self.mask_feature_channels,
            hidden_channels=hidden_channels,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
            boundary_ratio=boundary_ratio,
            min_dist=min_dist,
            peak_thr=peak_thr,
            low_response_thr=low_response_thr,
        )

    def _mask_features(self, coarse_logits: torch.Tensor, target_size: tuple, ref: torch.Tensor) -> torch.Tensor:
        if coarse_logits is None:
            return ref.new_zeros((ref.shape[0], self.mask_feature_channels, target_size[0], target_size[1]))
        if coarse_logits.dim() == 3:
            coarse_logits = coarse_logits.unsqueeze(1)
        if coarse_logits.shape[-2:] != target_size:
            coarse_logits = F.interpolate(coarse_logits, size=target_size, mode="bilinear", align_corners=False)
        if self.detach_mask:
            coarse_logits = coarse_logits.detach()
        coarse_logits = coarse_logits.to(device=ref.device, dtype=ref.dtype)
        logit_feat = coarse_logits.clamp(-10.0, 10.0) / 5.0
        prob_feat = torch.sigmoid(coarse_logits)
        uncertainty = 1.0 - torch.abs(2.0 * prob_feat - 1.0)
        feats = torch.cat([logit_feat, prob_feat, uncertainty], dim=1)
        return feats[:, : self.mask_feature_channels]

    def forward(
        self,
        encoder_features: torch.Tensor,
        coarse_logits: torch.Tensor = None,
        output_size: tuple = None,
        gt_mask: torch.Tensor = None,
        positive_guidance: torch.Tensor = None,
    ):
        target_size = encoder_features.shape[-2:]
        mask_features = self._mask_features(coarse_logits, target_size, encoder_features)
        guided_features = torch.cat([encoder_features, mask_features], dim=1)
        return super().forward(
            guided_features,
            output_size=output_size,
            gt_mask=gt_mask,
            positive_guidance=positive_guidance,
        )


class CoarseMaskPromptHead(nn.Module):
    """Lightweight dense mask prompt head from image embeddings."""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, -2.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        final = self.head[-1]
        if isinstance(final, nn.Conv2d) and final.bias is not None:
            nn.init.constant_(final.bias, -2.0)

    def forward(self, encoder_features: torch.Tensor, output_size: tuple = None):
        logits = self.head(encoder_features)
        if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


def coarse_mask_prompt_loss(
    mask_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 10.0,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
):
    """BCE+Dice supervision for the dense coarse mask prompt."""
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    if gt_mask.shape[-2:] != mask_logits.shape[-2:]:
        gt_mask = F.interpolate(gt_mask.float(), size=mask_logits.shape[-2:], mode="nearest")
    gt_mask = gt_mask.float()

    weight = torch.ones_like(gt_mask)
    weight[gt_mask > 0.5] = float(pos_weight)
    bce = F.binary_cross_entropy_with_logits(
        mask_logits,
        gt_mask,
        weight=weight,
        reduction="mean",
    )

    prob = torch.sigmoid(mask_logits)
    inter = (prob * gt_mask).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return float(bce_weight) * bce + float(dice_weight) * dice


class DynamicSparsePromptHead(nn.Module):
    """Predict extra sparse prompt tokens from image embeddings.

    The head uses K spatial attention maps over encoder features and pools a
    token per map. The resulting [B, K, C] tokens can be concatenated with SAM
    sparse prompt embeddings.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embed_dim: int = 256,
        num_tokens: int = 8,
        hidden_channels: int = 64,
        temperature: float = 1.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.temperature = float(max(1e-4, temperature))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(hidden_channels, self.num_tokens, kernel_size=1)
        self.value = nn.Conv2d(hidden_channels, self.embed_dim, kernel_size=1)
        self.token_norm = nn.LayerNorm(self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.base_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        self.token_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        feat = self.stem(encoder_features)
        bsz = feat.shape[0]
        scores = self.score(feat).flatten(2) / self.temperature
        attn = torch.softmax(scores, dim=-1)
        values = self.value(feat).flatten(2).transpose(1, 2)
        tokens = torch.bmm(attn, values)
        tokens = self.token_norm(tokens)
        tokens = self.out_proj(tokens)
        return self.base_tokens.expand(bsz, -1, -1) + self.token_scale * tokens


class MultiLevelDynamicSparsePromptHead(nn.Module):
    """Generate sparse prompt tokens from multiple image encoder levels.

    Each token learns two selections:
      - a spatial attention map over the 64x64 encoder grid.
      - a per-token level gate over block-level features plus neck_out.

    Shallow/mid layers drive localization, while deep/neck features provide
    decoder-aligned token values.
    """

    def __init__(
        self,
        level_channels=None,
        neck_channels: int = 256,
        embed_dim: int = 256,
        num_tokens: int = 8,
        hidden_channels: int = 128,
        temperature: float = 1.0,
        init_scale: float = 0.02,
        loc_levels: int = 3,
    ):
        super().__init__()
        if level_channels is None:
            level_channels = [192, 192, 192, 192]
        self.level_channels = [int(c) for c in level_channels]
        self.num_encoder_levels = len(self.level_channels)
        self.num_levels = self.num_encoder_levels + 1
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.temperature = float(max(1e-4, temperature))
        self.loc_levels = int(max(1, min(loc_levels, self.num_levels)))

        def make_proj(in_ch: int):
            return nn.Sequential(
                nn.Conv2d(in_ch, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, embed_dim),
                nn.GELU(),
            )

        self.level_projs = nn.ModuleList([make_proj(c) for c in self.level_channels])
        self.neck_proj = make_proj(int(neck_channels))
        self.loc_level_logits = nn.Parameter(torch.zeros(self.loc_levels))
        self.targetness_head = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        self.token_score = nn.Conv2d(embed_dim, self.num_tokens, kernel_size=1)
        self.targetness_bias = nn.Parameter(torch.tensor(1.0))
        self.level_gate = nn.Sequential(
            nn.Linear(embed_dim * self.num_levels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.num_tokens * self.num_levels),
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.base_tokens = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.token_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def _normalize_encoder_levels(self, encoder_levels, neck_features):
        if encoder_levels is None:
            encoder_levels = []
        levels = []
        for feat in list(encoder_levels)[: self.num_encoder_levels]:
            if feat is None:
                continue
            if feat.dim() != 4:
                continue
            # Encoder block features may be [B,H,W,C] in some paths.
            if feat.shape[1] != self.level_channels[min(len(levels), self.num_encoder_levels - 1)]:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            levels.append(feat)

        while len(levels) < self.num_encoder_levels:
            levels.append(torch.zeros(
                neck_features.shape[0],
                self.level_channels[len(levels)],
                neck_features.shape[-2],
                neck_features.shape[-1],
                device=neck_features.device,
                dtype=neck_features.dtype,
            ))
        return levels

    def forward(self, neck_features: torch.Tensor, encoder_levels=None):
        encoder_levels = self._normalize_encoder_levels(encoder_levels, neck_features)
        projected = []
        target_size = neck_features.shape[-2:]
        for feat, proj in zip(encoder_levels, self.level_projs):
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(proj(feat))
        projected.append(self.neck_proj(neck_features))

        loc_feats = projected[: self.loc_levels]
        loc_weights = torch.softmax(self.loc_level_logits, dim=0)
        loc_feat = sum(w * f for w, f in zip(loc_weights, loc_feats))
        targetness_logits = self.targetness_head(loc_feat)

        spatial_scores = self.token_score(loc_feat) + self.targetness_bias * targetness_logits
        attn = torch.softmax(spatial_scores.flatten(2) / self.temperature, dim=-1)

        pooled_per_level = []
        global_per_level = []
        for feat in projected:
            values = feat.flatten(2).transpose(1, 2)
            pooled_per_level.append(torch.bmm(attn, values))
            global_per_level.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        pooled = torch.stack(pooled_per_level, dim=2)  # [B,K,L,C]

        gate_input = torch.cat(global_per_level, dim=1)
        gate_logits = self.level_gate(gate_input).view(
            neck_features.shape[0], self.num_tokens, self.num_levels
        )
        gate = torch.softmax(gate_logits, dim=-1)
        tokens = (pooled * gate.unsqueeze(-1)).sum(dim=2)
        tokens = self.token_norm(tokens)
        tokens = self.out_proj(tokens)
        tokens = self.base_tokens.expand(neck_features.shape[0], -1, -1) + self.token_scale * tokens
        aux = {
            "targetness_logits": targetness_logits,
            "targetness_prob": torch.sigmoid(targetness_logits),
            "level_gate": gate,
            "loc_level_weights": loc_weights,
        }
        return tokens, aux


def dynamic_sparse_token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    """Penalize duplicate dynamic sparse tokens."""
    if tokens is None or tokens.dim() != 3 or tokens.shape[1] <= 1:
        if tokens is None:
            return torch.tensor(0.0)
        return tokens.sum() * 0.0
    normed = F.normalize(tokens.float(), dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))
    eye = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool).unsqueeze(0).expand_as(sim)
    off_diag = sim.masked_select(~eye)
    return (off_diag ** 2).mean()


def dynamic_sparse_targetness_loss(
    targetness_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 20.0,
    dice_weight: float = 1.0,
):
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)
    gt_mask = gt_mask.float()
    th, tw = targetness_logits.shape[-2:]
    gh, gw = gt_mask.shape[-2:]
    if gh >= th and gw >= tw and gh % th == 0 and gw % tw == 0:
        target = F.max_pool2d(gt_mask, kernel_size=(gh // th, gw // tw), stride=(gh // th, gw // tw))
    else:
        target = F.interpolate(gt_mask, size=(th, tw), mode="nearest")
    target = target.clamp(0.0, 1.0)
    weight = torch.ones_like(target)
    weight[target > 0.5] = float(pos_weight)
    bce = F.binary_cross_entropy_with_logits(targetness_logits, target, weight=weight, reduction="mean")
    prob = torch.sigmoid(targetness_logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
    return bce + float(dice_weight) * dice


def build_dynamic_sparse_prompt_head(
    in_channels: int = 256,
    embed_dim: int = 256,
    num_tokens: int = 8,
    hidden_channels: int = 64,
    temperature: float = 1.0,
    init_scale: float = 0.1,
):
    return DynamicSparsePromptHead(
        in_channels=in_channels,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        hidden_channels=hidden_channels,
        temperature=temperature,
        init_scale=init_scale,
    )


def build_multilevel_dynamic_sparse_prompt_head(
    level_channels=None,
    neck_channels: int = 256,
    embed_dim: int = 256,
    num_tokens: int = 8,
    hidden_channels: int = 128,
    temperature: float = 1.0,
    init_scale: float = 0.02,
    loc_levels: int = 3,
):
    return MultiLevelDynamicSparsePromptHead(
        level_channels=level_channels,
        neck_channels=neck_channels,
        embed_dim=embed_dim,
        num_tokens=num_tokens,
        hidden_channels=hidden_channels,
        temperature=temperature,
        init_scale=init_scale,
        loc_levels=loc_levels,
    )


def build_self_prompting_head(
    in_channels: int = 256,
    hidden_channels: int = 64,
    top_k_pos: int = 3,
    top_k_neg: int = 2,
    min_dist: int = 8,
    peak_thr: float = 0.1,
    low_response_thr: float = 0.3,
    boundary_aware: bool = False,
    boundary_ratio: float = 0.5,
):
    """Factory function for SelfPromptingHead."""
    if boundary_aware:
        return BoundaryAwareSelfPromptingHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
            boundary_ratio=boundary_ratio,
            min_dist=min_dist,
            peak_thr=peak_thr,
            low_response_thr=low_response_thr,
        )
    return SelfPromptingHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        top_k_pos=top_k_pos,
        top_k_neg=top_k_neg,
        min_dist=min_dist,
        peak_thr=peak_thr,
        low_response_thr=low_response_thr,
    )


def build_coarse_mask_prompt_head(
    in_channels: int = 256,
    hidden_channels: int = 64,
):
    return CoarseMaskPromptHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
    )


def build_mask_guided_self_prompting_head(
    in_channels: int = 256,
    hidden_channels: int = 64,
    top_k_pos: int = 4,
    top_k_neg: int = 4,
    min_dist: int = 8,
    peak_thr: float = 0.05,
    low_response_thr: float = 0.3,
    boundary_ratio: float = 0.5,
    mask_feature_channels: int = 3,
    detach_mask: bool = True,
):
    return MaskGuidedBoundarySelfPromptingHead(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        top_k_pos=top_k_pos,
        top_k_neg=top_k_neg,
        boundary_ratio=boundary_ratio,
        min_dist=min_dist,
        peak_thr=peak_thr,
        low_response_thr=low_response_thr,
        mask_feature_channels=mask_feature_channels,
        detach_mask=detach_mask,
    )


def self_prompt_heatmap_loss(
    heatmap_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    pos_weight: float = 10.0,
):
    """
    Weighted BCE loss for heatmap supervision (AMP-safe).

    Uses binary_cross_entropy_with_logits which is safe under autocast.
    Because targets are tiny (< 1% of pixels), positive pixels
    are weighted much higher than negatives.
    """
    if gt_mask.dim() == 3:
        gt_mask = gt_mask.unsqueeze(1)

    if heatmap_logits.shape[-2:] != gt_mask.shape[-2:]:
        gt_mask = F.interpolate(
            gt_mask.float(), size=heatmap_logits.shape[-2:], mode="nearest"
        )

    gt_mask = gt_mask.float()
    weight = torch.ones_like(gt_mask)
    weight[gt_mask > 0.5] = pos_weight

    loss = F.binary_cross_entropy_with_logits(
        heatmap_logits,
        gt_mask,
        weight=weight,
        reduction="mean",
    )
    return loss

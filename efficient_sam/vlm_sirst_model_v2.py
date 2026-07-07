from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .efficient_sam_hq import build_efficient_sam_hq
from .self_prompting_head import SelfPromptingHead


class SceneConditionalNorm(nn.Module):
    """Conditional instance normalization driven by scene labels."""

    def __init__(self, num_scenes: int = 7, channels: int = 256) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False, eps=1e-5)
        self.gamma_bank = nn.Embedding(num_scenes, channels)
        self.beta_bank = nn.Embedding(num_scenes, channels)
        nn.init.ones_(self.gamma_bank.weight)
        nn.init.zeros_(self.beta_bank.weight)

    def forward(self, x: torch.Tensor, scene_id: torch.Tensor) -> torch.Tensor:
        scene_id = scene_id.to(device=x.device, dtype=torch.long).clamp(min=0, max=self.gamma_bank.num_embeddings - 1)
        gamma = self.gamma_bank(scene_id).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_bank(scene_id).unsqueeze(-1).unsqueeze(-1)
        x_norm = self.norm(x.float()).to(dtype=x.dtype)
        return x_norm * gamma.to(dtype=x.dtype) + beta.to(dtype=x.dtype)


class SceneClassifier(nn.Module):
    """Predict scene labels from image embeddings for VLM-free inference."""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 128, num_scenes: int = 7) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_scenes),
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(image_embeddings).flatten(1)
        return self.mlp(pooled.float())


class VLMGuidedSelfPromptingSAMV2(nn.Module):
    """Scheme 1 v2: VLM meta-guided training and VLM-free inference."""

    def __init__(
        self,
        model_type: str = "vitt",
        init_from_baseline: Optional[str] = None,
        self_prompt_top_k_pos: int = 3,
        self_prompt_top_k_neg: int = 2,
        self_prompt_hidden_channels: int = 64,
        self_prompt_min_dist: int = 8,
        self_prompt_peak_thr: float = 0.1,
        self_prompt_low_response_thr: float = 0.3,
        use_scn: bool = True,
        hq_token_only: bool = False,
    ) -> None:
        super().__init__()
        model_type = str(model_type).lower()
        if model_type not in {"vitt", "vits"}:
            raise ValueError("model_type must be 'vitt' or 'vits'.")
        patch_dim = 192 if model_type == "vitt" else 384
        num_heads = 3 if model_type == "vitt" else 6
        self.sam = build_efficient_sam_hq(
            encoder_patch_embed_dim=patch_dim,
            encoder_num_heads=num_heads,
            init_from_baseline=init_from_baseline,
            use_adapter=False,
        )
        self.hq_token_only = bool(hq_token_only)
        self.use_scn = bool(use_scn)
        self.scene_norm = SceneConditionalNorm(num_scenes=7, channels=256) if self.use_scn else None
        self.scene_classifier = SceneClassifier(in_channels=256, hidden_dim=128, num_scenes=7) if self.use_scn else None
        self.self_prompt_head = SelfPromptingHead(
            in_channels=256,
            hidden_channels=self_prompt_hidden_channels,
            top_k_pos=self_prompt_top_k_pos,
            top_k_neg=self_prompt_top_k_neg,
            min_dist=self_prompt_min_dist,
            peak_thr=self_prompt_peak_thr,
            low_response_thr=self_prompt_low_response_thr,
        )

    def _get_embeddings(self, images: torch.Tensor):
        out = self.sam.get_image_embeddings(images)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError("EfficientSamHQ.get_image_embeddings must return at least image and intermediate embeddings.")
        return out[0], out[1]

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        scene_id: Optional[torch.Tensor] = None,
        use_oracle_scene: bool = False,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, _, image_h, image_w = images.shape
        if output_size is None:
            output_size = (image_h, image_w)

        image_embeddings, interm_embeddings = self._get_embeddings(images)
        raw_embeddings = image_embeddings
        scene_logits = None
        applied_scene_id = torch.zeros((bsz,), device=images.device, dtype=torch.long)

        if self.use_scn and self.scene_classifier is not None and self.scene_norm is not None:
            scene_logits = self.scene_classifier(raw_embeddings)
            if use_oracle_scene and scene_id is not None:
                applied_scene_id = scene_id.to(device=images.device, dtype=torch.long)
            else:
                applied_scene_id = scene_logits.argmax(dim=1).to(dtype=torch.long)
            image_embeddings = self.scene_norm(image_embeddings, applied_scene_id)

        heatmap, point_coords, point_labels, heatmap_logits = self.self_prompt_head(
            image_embeddings,
            output_size=output_size,
            gt_mask=masks,
        )
        pred_masks, decoder_iou = self.sam.predict_masks(
            image_embeddings,
            interm_embeddings,
            point_coords,
            point_labels,
            multimask_output=False,
            input_h=image_h,
            input_w=image_w,
            output_h=output_size[0],
            output_w=output_size[1],
            hq_token_only=self.hq_token_only,
            text_sparse_embeddings=None,
        )
        mask_logits = pred_masks[:, 0, 0].unsqueeze(1)
        return {
            "mask_logits": mask_logits,
            "decoder_iou": decoder_iou,
            "heatmap": heatmap,
            "heatmap_logits": heatmap_logits,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "image_embeddings": image_embeddings,
            "raw_image_embeddings": raw_embeddings,
            "scene_logits": scene_logits if scene_logits is not None else torch.empty(0, device=images.device),
            "applied_scene_id": applied_scene_id,
        }


def build_vlm_guided_self_prompting_sam_v2(**kwargs) -> VLMGuidedSelfPromptingSAMV2:
    return VLMGuidedSelfPromptingSAMV2(**kwargs)

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .efficient_sam_hq import build_efficient_sam_hq
from .self_prompting_head import SelfPromptingHead
from .text_conditioner import BiFusionAdapterLite, TextSparsePromptProjector


class SceneAdapter(nn.Module):
    """Lightweight scene-conditioned FiLM adapter for SAM image embeddings."""

    def __init__(self, num_scenes: int = 7, channels: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_scenes, hidden_dim)
        self.to_gamma = nn.Linear(hidden_dim, channels)
        self.to_beta = nn.Linear(hidden_dim, channels)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.zeros_(self.embedding.weight.data[0])
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, image_embeddings: torch.Tensor, scene_id: Optional[torch.Tensor]) -> torch.Tensor:
        if scene_id is None:
            return image_embeddings
        scene_id = scene_id.to(device=image_embeddings.device, dtype=torch.long).clamp(min=0)
        emb = self.embedding(scene_id)
        gamma = self.to_gamma(emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.to_beta(emb).unsqueeze(-1).unsqueeze(-1)
        return image_embeddings * (1.0 + gamma.to(image_embeddings.dtype)) + beta.to(image_embeddings.dtype)


class MaskQualityHead(nn.Module):
    """Predict per-prompt quality from pooled image features and prompt coordinates."""

    def __init__(self, channels: int = 256, hidden_dim: int = 128) -> None:
        super().__init__()
        self.prompt_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.image_proj = nn.Linear(channels, hidden_dim)
        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, _, num_points, _ = point_coords.shape
        pooled = image_embeddings.mean(dim=(2, 3))
        img_context = self.image_proj(pooled).unsqueeze(1).expand(-1, num_points, -1)
        h, w = image_size
        denom = point_coords.new_tensor([max(w - 1, 1), max(h - 1, 1)])
        coords = point_coords[:, 0] / denom.view(1, 1, 2)
        labels = point_labels[:, 0].float().unsqueeze(-1)
        prompt_in = torch.cat([coords.clamp(0.0, 1.0), labels], dim=-1)
        prompt_context = self.prompt_mlp(prompt_in.to(self.prompt_mlp[0].weight.dtype))
        quality = self.out((img_context + prompt_context).to(self.out[1].weight.dtype))
        return quality.view(bsz, 1, num_points)


class VLMReasonedSelfPromptingSAM(nn.Module):
    """Scheme-1 wrapper: VLM-conditioned EfficientSAM-HQ with learned self prompts."""

    def __init__(
        self,
        model_type: str = "vitt",
        init_from_baseline: Optional[str] = None,
        text_dim: int = 512,
        text_sparse_num_tokens: int = 1,
        self_prompt_top_k_pos: int = 3,
        self_prompt_top_k_neg: int = 2,
        self_prompt_hidden_channels: int = 64,
        self_prompt_min_dist: int = 8,
        self_prompt_peak_thr: float = 0.1,
        self_prompt_low_response_thr: float = 0.3,
        bifusion_hidden_dim: int = 128,
        bifusion_num_heads: int = 4,
        use_bifusion: bool = True,
        use_text_sparse_prompt: bool = True,
        use_scene_adapter: bool = True,
        hq_token_only: bool = False,
    ) -> None:
        super().__init__()
        model_type = str(model_type).lower()
        if model_type not in {"vitt", "vits"}:
            raise ValueError("model_type must be 'vitt' or 'vits'.")
        self.model_type = model_type
        patch_dim = 192 if model_type == "vitt" else 384
        num_heads = 3 if model_type == "vitt" else 6
        self.sam = build_efficient_sam_hq(
            encoder_patch_embed_dim=patch_dim,
            encoder_num_heads=num_heads,
            init_from_baseline=init_from_baseline,
            use_adapter=False,
        )
        self.hq_token_only = bool(hq_token_only)
        self.use_bifusion = bool(use_bifusion)
        self.use_text_sparse_prompt = bool(use_text_sparse_prompt)
        self.use_scene_adapter = bool(use_scene_adapter)
        self.scene_adapter = SceneAdapter(num_scenes=7, channels=256) if self.use_scene_adapter else None
        self.bifusion = (
            BiFusionAdapterLite(
                img_dim=256,
                interms_dim=patch_dim,
                text_dim=text_dim,
                hidden_dim=bifusion_hidden_dim,
                num_heads=bifusion_num_heads,
                use_interms_level=True,
                img_res_scale=1.0,
                interms_res_scale=1.0,
                text_res_scale=1.0,
            )
            if self.use_bifusion
            else None
        )
        self.text_sparse_projector = (
            TextSparsePromptProjector(
                text_dim=text_dim,
                embed_dim=256,
                num_tokens=text_sparse_num_tokens,
                init_scale=0.02,
                use_raw_global_gate=True,
                raw_global_gate_init_bias=-2.0,
            )
            if self.use_text_sparse_prompt
            else None
        )
        self.self_prompt_head = SelfPromptingHead(
            in_channels=256,
            hidden_channels=self_prompt_hidden_channels,
            top_k_pos=self_prompt_top_k_pos,
            top_k_neg=self_prompt_top_k_neg,
            min_dist=self_prompt_min_dist,
            peak_thr=self_prompt_peak_thr,
            low_response_thr=self_prompt_low_response_thr,
        )
        self.mask_quality_head = MaskQualityHead(channels=256, hidden_dim=128)

    def _get_embeddings(self, images: torch.Tensor):
        out = self.sam.get_image_embeddings(images)
        if not isinstance(out, (tuple, list)) or len(out) < 2:
            raise RuntimeError("EfficientSamHQ.get_image_embeddings must return at least image and intermediate embeddings.")
        return out[0], out[1]

    def _fuse_vlm(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        clip_token_feat: Optional[torch.Tensor],
        clip_text_feat: Optional[torch.Tensor],
        clip_text_attn_mask: Optional[torch.Tensor],
        scene_id: Optional[torch.Tensor],
        use_vlm: bool,
    ):
        text_sparse = None
        text_global = clip_text_feat
        text_tokens = clip_token_feat
        text_mask = clip_text_attn_mask
        if not use_vlm:
            return image_embeddings, interm_embeddings, text_sparse, text_global, text_tokens, text_mask
        if self.scene_adapter is not None:
            image_embeddings = self.scene_adapter(image_embeddings, scene_id)
        text_input = clip_token_feat if clip_token_feat is not None else clip_text_feat
        if text_input is not None and self.bifusion is not None:
            image_embeddings, interm_embeddings, text_tokens, text_mask, text_global = self.bifusion(
                image_embeddings,
                interm_embeddings,
                text_input,
                attention_mask=clip_text_attn_mask if text_input.dim() == 3 else None,
            )
        if text_input is not None and self.text_sparse_projector is not None:
            sparse_source = text_tokens if text_tokens is not None else text_global
            if sparse_source is None:
                sparse_source = text_input
            sparse_mask = text_mask
            if sparse_mask is None and sparse_source is clip_token_feat:
                sparse_mask = clip_text_attn_mask
            text_sparse = self.text_sparse_projector(
                sparse_source,
                attention_mask=sparse_mask if sparse_source is not None and sparse_source.dim() == 3 else None,
                use_global_prompt_enhance=sparse_source is not None and sparse_source.dim() == 2,
            )
        return image_embeddings, interm_embeddings, text_sparse, text_global, text_tokens, text_mask

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        clip_text_feat: Optional[torch.Tensor] = None,
        clip_text_token_feat: Optional[torch.Tensor] = None,
        clip_text_attn_mask: Optional[torch.Tensor] = None,
        scene_id: Optional[torch.Tensor] = None,
        use_vlm: bool = True,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        bsz, _, image_h, image_w = images.shape
        if output_size is None:
            output_size = (image_h, image_w)

        image_embeddings, interm_embeddings = self._get_embeddings(images)
        image_embeddings, interm_embeddings, text_sparse, text_global, text_tokens, text_mask = self._fuse_vlm(
            image_embeddings=image_embeddings,
            interm_embeddings=interm_embeddings,
            clip_token_feat=clip_text_token_feat,
            clip_text_feat=clip_text_feat,
            clip_text_attn_mask=clip_text_attn_mask,
            scene_id=scene_id,
            use_vlm=bool(use_vlm),
        )

        heatmap, point_coords, point_labels, heatmap_logits = self.self_prompt_head(
            image_embeddings,
            output_size=output_size,
            gt_mask=masks,
        )
        quality = self.mask_quality_head(
            image_embeddings,
            point_coords,
            point_labels,
            image_size=output_size,
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
            text_sparse_embeddings=text_sparse,
        )
        mask_logits = pred_masks[:, 0, 0].unsqueeze(1)
        return {
            "mask_logits": mask_logits,
            "decoder_iou": decoder_iou,
            "heatmap": heatmap,
            "heatmap_logits": heatmap_logits,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "quality": quality,
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "text_global": text_global if text_global is not None else torch.empty(0, device=images.device),
            "text_tokens": text_tokens if text_tokens is not None else torch.empty(0, device=images.device),
            "text_mask": text_mask if text_mask is not None else torch.empty(0, device=images.device),
        }


def build_vlm_reasoned_self_prompting_sam(**kwargs) -> VLMReasonedSelfPromptingSAM:
    return VLMReasonedSelfPromptingSAM(**kwargs)

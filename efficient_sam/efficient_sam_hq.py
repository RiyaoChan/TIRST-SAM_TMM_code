import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .efficient_sam_prompt_encoder_hq import PromptEncoderHQ
from .two_way_transformer import TwoWayTransformer
from .efficient_sam_encoder_hq import ImageEncoderViTHQ
from .efficient_sam_decoder_hq import MaskDecoderHQ


class SobelDetailEnhancer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(dim, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(dim, 1, 1, 1))
        self.dim = int(dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.sobel_x, padding=1, groups=self.dim)
        grad_y = F.conv2d(x, self.sobel_y, padding=1, groups=self.dim)
        magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        out = torch.cat([x, magnitude], dim=1)
        return self.fusion(out)


class MultiScaleAggregator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_levels: int = 4):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ) for _ in range(num_levels)
        ])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_dim * num_levels, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, feats):
        if not feats:
            return None
        use_levels = min(len(feats), len(self.convs))
        processed = [self.convs[i](feats[i]) for i in range(use_levels)]
        if len(processed) < len(self.convs):
            # Pad missing levels with zeros to keep fusion shape stable.
            b, c, h, w = processed[0].shape
            for _ in range(len(self.convs) - len(processed)):
                processed.append(torch.zeros((b, c, h, w), device=processed[0].device, dtype=processed[0].dtype))
        out = torch.cat(processed, dim=1)
        return self.fusion_conv(out)


class EfficientSamHQ(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViTHQ,
        prompt_encoder: PromptEncoderHQ,
        mask_decoder: MaskDecoderHQ,
        pixel_mean: List[float] = [0.485, 0.456, 0.406],
        pixel_std: List[float] = [0.229, 0.224, 0.225],
        use_ms_fusion: bool = False,
        use_detail_enhancer: bool = False,
        use_hldf: bool = False,
        use_detail_branch_amgd: bool = False,
        return_encoder_multi_scale: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(1, 3, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(1, 3, 1, 1), False)
        try:
            neck_dim = int(self.image_encoder.neck[0].out_channels)
        except Exception:
            neck_dim = int(getattr(self.image_encoder, "transformer_output_dim", 256))
        mid_dim = max(1, neck_dim // 4)
        self.saliency_adapter = nn.Sequential(
            nn.Conv2d(1, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, neck_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.use_ms_fusion = bool(use_ms_fusion)
        self.use_detail_enhancer = bool(use_detail_enhancer)
        self.use_hldf = bool(use_hldf)
        self.use_detail_branch_amgd = bool(use_detail_branch_amgd)
        self.return_encoder_multi_scale = bool(return_encoder_multi_scale)
        try:
            embed_dim = int(self.image_encoder.patch_embed.proj.out_channels)
        except Exception:
            embed_dim = int(getattr(self.image_encoder, "transformer_output_dim", neck_dim))
        if self.use_detail_enhancer:
            self.detail_enhancer = SobelDetailEnhancer(embed_dim)
        if self.use_ms_fusion:
            self.ms_aggregator = MultiScaleAggregator(in_dim=embed_dim, out_dim=neck_dim)

    def _expects_detail_multi_scale(self) -> bool:
        return bool(self.use_hldf or self.use_detail_branch_amgd)

    def _split_image_encoder_output(self, out):
        if not isinstance(out, (tuple, list)):
            raise ValueError("image_encoder must return a tuple for HQ model.")
        neck_out, interm = out[:2]
        ms_feats = []
        detail_ms_feats = []
        expect_detail = self._expects_detail_multi_scale()
        expect_ms = bool(self.use_ms_fusion or self.return_encoder_multi_scale)
        if expect_ms and expect_detail and len(out) >= 4:
            ms_feats = out[2]
            detail_ms_feats = out[3]
        elif expect_ms and len(out) >= 3:
            ms_feats = out[2]
        elif expect_detail and len(out) >= 3:
            detail_ms_feats = out[2]
        return neck_out, interm, ms_feats, detail_ms_feats

    @torch.jit.export
    def get_image_embeddings(self, batched_images) -> Tuple[torch.Tensor, ...]:
        batched_images = self.preprocess(batched_images)
        out = self.image_encoder(batched_images)
        neck_out, interm, ms_feats, detail_ms_feats = self._split_image_encoder_output(out)
        if self.use_ms_fusion and ms_feats:
            fused = self.ms_aggregator(ms_feats)
            if fused is not None:
                neck_out = neck_out + fused
        if self.use_detail_enhancer and interm is not None:
            interm_c = interm.permute(0, 3, 1, 2)
            enhanced = self.detail_enhancer(interm_c)
            interm = enhanced.permute(0, 2, 3, 1)
        if self._expects_detail_multi_scale() and self.return_encoder_multi_scale:
            return neck_out, interm, detail_ms_feats, ms_feats
        if self._expects_detail_multi_scale():
            return neck_out, interm, detail_ms_feats
        if self.return_encoder_multi_scale:
            return neck_out, interm, ms_feats
        return neck_out, interm

    def get_image_embeddings_with_text(
        self,
        batched_images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        batched_images = self.preprocess(batched_images)
        if not hasattr(self.image_encoder, "forward_with_text"):
            out = self.image_encoder(batched_images)
            neck_out, interm, ms_feats, detail_ms_feats = self._split_image_encoder_output(out)
            if self.use_ms_fusion and ms_feats:
                fused = self.ms_aggregator(ms_feats)
                if fused is not None:
                    neck_out = neck_out + fused
            if self.use_detail_enhancer and interm is not None:
                interm_c = interm.permute(0, 3, 1, 2)
                enhanced = self.detail_enhancer(interm_c)
                interm = enhanced.permute(0, 2, 3, 1)
            if self._expects_detail_multi_scale() and self.return_encoder_multi_scale:
                return neck_out, interm, detail_ms_feats, ms_feats, text_tokens, text_attention_mask
            if self._expects_detail_multi_scale():
                return neck_out, interm, detail_ms_feats, text_tokens, text_attention_mask
            if self.return_encoder_multi_scale:
                return neck_out, interm, ms_feats, text_tokens, text_attention_mask
            return neck_out, interm, text_tokens, text_attention_mask
        out = self.image_encoder.forward_with_text(
            batched_images,
            text_tokens,
            text_attention_mask=text_attention_mask,
        )
        if not isinstance(out, (tuple, list)):
            raise ValueError("image_encoder.forward_with_text must return a tuple for HQ model.")
        if self.use_ms_fusion and self._expects_detail_multi_scale() and len(out) >= 6:
            neck_out, interm, ms_feats, detail_ms_feats, text_tokens_out, text_mask_out = out[:6]
        elif self.use_ms_fusion and len(out) >= 5:
            neck_out, interm, ms_feats, text_tokens_out, text_mask_out = out[:5]
            detail_ms_feats = []
        elif self._expects_detail_multi_scale() and len(out) >= 5:
            neck_out, interm, detail_ms_feats, text_tokens_out, text_mask_out = out[:5]
            ms_feats = []
        else:
            neck_out, interm, text_tokens_out, text_mask_out = out[:4]
            ms_feats = []
            detail_ms_feats = []
        if self.use_ms_fusion and ms_feats:
            fused = self.ms_aggregator(ms_feats)
            if fused is not None:
                neck_out = neck_out + fused
        if self.use_detail_enhancer and interm is not None:
            interm_c = interm.permute(0, 3, 1, 2)
            enhanced = self.detail_enhancer(interm_c)
            interm = enhanced.permute(0, 2, 3, 1)
        if self._expects_detail_multi_scale() and self.return_encoder_multi_scale:
            return neck_out, interm, detail_ms_feats, ms_feats, text_tokens_out, text_mask_out
        if self._expects_detail_multi_scale():
            return neck_out, interm, detail_ms_feats, text_tokens_out, text_mask_out
        if self.return_encoder_multi_scale:
            return neck_out, interm, ms_feats, text_tokens_out, text_mask_out
        return neck_out, interm, text_tokens_out, text_mask_out

    def apply_saliency_modulation(self, image_embeddings: torch.Tensor, saliency_map: torch.Tensor) -> torch.Tensor:
        if saliency_map is None:
            return image_embeddings
        target_h, target_w = image_embeddings.shape[-2:]
        saliency_small = F.interpolate(
            saliency_map, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        modulation_weight = self.saliency_adapter(saliency_small)
        return image_embeddings * (1.0 + modulation_weight)

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        interm_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
        hq_token_only: bool = False,
        batched_masks: Optional[torch.Tensor] = None,
        text_sparse_embeddings: Optional[torch.Tensor] = None,
        multi_scale_embeddings: Optional[List[torch.Tensor]] = None,
        detail_branch_embeddings: Optional[List[torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        rescaled_batched_points = self.get_rescaled_pts(batched_points, input_h, input_w)
        if batched_masks is not None:
            if batched_masks.dim() == 3:
                batched_masks = batched_masks.unsqueeze(1)
            if batched_masks.shape[0] == batch_size:
                batched_masks = batched_masks.repeat_interleave(max_num_queries, dim=0)
            elif batched_masks.shape[0] != batch_size * max_num_queries:
                raise ValueError("batched_masks must have batch size B or B*Q to match prompts.")
        if text_sparse_embeddings is not None:
            if text_sparse_embeddings.dim() != 3:
                raise ValueError("text_sparse_embeddings must have shape [B, T, C] or [B*Q, T, C].")
            if text_sparse_embeddings.shape[0] == batch_size:
                text_sparse_embeddings = text_sparse_embeddings.repeat_interleave(max_num_queries, dim=0)
            elif text_sparse_embeddings.shape[0] != batch_size * max_num_queries:
                raise ValueError("text_sparse_embeddings must have batch size B or B*Q to match prompts.")

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(
                rescaled_batched_points.reshape(
                    batch_size * max_num_queries, num_pts, 2
                ),
                batched_point_labels.reshape(
                    batch_size * max_num_queries, num_pts
                ),
            ),
            boxes=None,
            masks=batched_masks,
            text_embeds=text_sparse_embeddings,
        )  # sparse: [B*Q, N, C], dense: [B*Q, C, H, W]
        # Repeat along queries
        image_embeddings = image_embeddings.repeat_interleave(max_num_queries, dim=0)
        interm_embeddings = interm_embeddings.repeat_interleave(max_num_queries, dim=0)
        repeated_multi_scale = None
        if multi_scale_embeddings is not None:
            repeated_multi_scale = []
            for feat in multi_scale_embeddings:
                if feat.shape[0] != batch_size:
                    raise ValueError("Each multi_scale_embeddings tensor must have batch size B to match prompts.")
                repeated_multi_scale.append(feat.repeat_interleave(max_num_queries, dim=0))
        repeated_detail_branches = None
        if detail_branch_embeddings is not None:
            repeated_detail_branches = []
            for feat in detail_branch_embeddings:
                if feat.shape[0] != batch_size:
                    raise ValueError("Each detail_branch_embeddings tensor must have batch size B to match prompts.")
                repeated_detail_branches.append(feat.repeat_interleave(max_num_queries, dim=0))

        decoder_out = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=hq_token_only,
            interm_embeddings=interm_embeddings,
            multi_scale_embeddings=repeated_multi_scale,
            detail_branch_embeddings=repeated_detail_branches,
            return_aux=return_aux,
        )
        aux_dict: Dict[str, torch.Tensor] = {}
        if return_aux:
            low_res_masks, iou_predictions, aux_dict = decoder_out
        else:
            low_res_masks, iou_predictions = decoder_out

        if output_h > 0 and output_w > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, 1, output_h, output_w),
            )
        else:
            low_res_size = low_res_masks.shape[-1]
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    1,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, 1)
        )
        if not return_aux:
            return output_masks, iou_predictions

        reshaped_aux: Dict[str, torch.Tensor] = {}
        for key, value in aux_dict.items():
            if value is None:
                continue
            if output_h > 0 and output_w > 0 and value.shape[-2:] != (output_h, output_w):
                value = F.interpolate(value, (output_h, output_w), mode="bilinear", align_corners=False)
            elif output_h <= 0 and output_w <= 0:
                low_res_size = low_res_masks.shape[-1]
                if value.shape[-2:] != (low_res_size, low_res_size):
                    value = F.interpolate(value, (low_res_size, low_res_size), mode="bilinear", align_corners=False)
            value = torch.reshape(
                value,
                (batch_size, max_num_queries, 1, value.shape[-2], value.shape[-1]),
            )
            if key == "center_logits":
                reshaped_aux[key] = value
            elif key == "center_prob":
                continue
            else:
                reshaped_aux[key] = value
        if "center_logits" in reshaped_aux:
            reshaped_aux["center_prob"] = torch.sigmoid(reshaped_aux["center_logits"])
        elif "center_prob" in aux_dict:
            value = aux_dict["center_prob"]
            if output_h > 0 and output_w > 0 and value.shape[-2:] != (output_h, output_w):
                value = F.interpolate(value, (output_h, output_w), mode="bilinear", align_corners=False)
            value = torch.reshape(
                value,
                (batch_size, max_num_queries, 1, value.shape[-2], value.shape[-1]),
            )
            reshaped_aux["center_prob"] = value
        return output_masks, iou_predictions, reshaped_aux

    def get_rescaled_pts(self, batched_points: torch.Tensor, input_h: int, input_w: int):
        return torch.stack(
            [
                torch.where(
                    batched_points[..., 0] >= 0,
                    batched_points[..., 0] * self.image_encoder.img_size / input_w,
                    -1.0,
                ),
                torch.where(
                    batched_points[..., 1] >= 0,
                    batched_points[..., 1] * self.image_encoder.img_size / input_h,
                    -1.0,
                ),
            ],
            dim=-1,
        )

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.shape[2] != self.image_encoder.img_size
            or x.shape[3] != self.image_encoder.img_size
        ):
            x = F.interpolate(
                x,
                (self.image_encoder.img_size, self.image_encoder.img_size),
                mode="bilinear",
            )
        return (x - self.pixel_mean) / self.pixel_std


def build_efficient_sam_hq(
    encoder_patch_embed_dim,
    encoder_num_heads,
    init_from_baseline: Optional[str] = None,
    use_adapter: bool = True,
    use_ms_fusion: bool = False,
    use_detail_enhancer: bool = False,
    early_exit_layer: Optional[int] = None,
    use_amgd: bool = False,
    use_dog_amgd: bool = False,
    amgd_routing: str = "prompt",
    amgd_interm_layer: int = 0,
    amgd_branch_design: str = "legacy",
    amgd_detail_layer: Optional[int] = None,
    amgd_structure_layer: Optional[int] = None,
    amgd_background_layer: Optional[int] = None,
    dog_amgd_mode: str = "legacy",
    dog_amgd_strength: float = 0.25,
    use_hldf: bool = False,
    hldf_layers: Optional[List[int]] = None,
    hldf_hidden_dim: int = 96,
    hldf_use_hq_router: bool = True,
    hldf_router_temp: float = 1.0,
    use_center_mask_decoder: bool = False,
    center_gate_alpha: float = 0.2,
    return_encoder_multi_scale: bool = False,
):
    img_size = 1024
    encoder_patch_size = 16
    encoder_depth = 12
    encoder_mlp_ratio = 4.0
    encoder_neck_dims = [256, 256]
    prompt_embed_dim = encoder_neck_dims[-1]
    amgd_branch_design = str(amgd_branch_design).lower()
    use_detail_branch_amgd = bool(use_amgd and amgd_branch_design == "dsb_v1")
    detail_layer = max(0, int(amgd_interm_layer if amgd_detail_layer is None else amgd_detail_layer))
    structure_layer = max(0, int(amgd_interm_layer if amgd_structure_layer is None else amgd_structure_layer))
    background_layer = max(0, int(amgd_interm_layer if amgd_background_layer is None else amgd_background_layer))
    detail_branch_indices = sorted({detail_layer, structure_layer, background_layer})

    if use_hldf and use_amgd:
        raise ValueError("use_hldf and use_amgd are mutually exclusive.")
    if use_hldf and use_ms_fusion:
        raise ValueError("use_hldf and use_ms_fusion cannot be enabled together in the current implementation.")

    image_encoder = ImageEncoderViTHQ(
        img_size=img_size,
        patch_size=encoder_patch_size,
        in_chans=3,
        patch_embed_dim=encoder_patch_embed_dim,
        normalization_type="layer_norm",
        depth=encoder_depth,
        num_heads=encoder_num_heads,
        mlp_ratio=encoder_mlp_ratio,
        neck_dims=encoder_neck_dims,
        act_layer=nn.GELU,
        use_adapter=use_adapter,
        return_multi_scale=bool(use_ms_fusion or return_encoder_multi_scale),
        return_detail_multi_scale=(use_hldf or use_detail_branch_amgd),
        early_exit_layer=early_exit_layer,
        amgd_interm_layer=amgd_interm_layer,
        detail_ms_indices=hldf_layers if use_hldf else detail_branch_indices,
    )

    image_embedding_size = image_encoder.image_embedding_size
    transformer_dim = prompt_embed_dim

    model = EfficientSamHQ(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoderHQ(
            embed_dim=transformer_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoderHQ(
            transformer_dim=transformer_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=transformer_dim,
                mlp_dim=2048,
                num_heads=8,
                activation=nn.GELU,
                normalize_before_activation=False,
            ),
            num_multimask_outputs=1,
            activation=nn.GELU,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            vit_dim=encoder_patch_embed_dim,
            use_amgd=use_amgd,
            use_dog_amgd=use_dog_amgd,
            amgd_routing=amgd_routing,
            amgd_branch_design=amgd_branch_design,
            amgd_detail_layer=detail_layer,
            amgd_structure_layer=structure_layer,
            amgd_background_layer=background_layer,
            amgd_branch_layers=detail_branch_indices,
            dog_amgd_mode=dog_amgd_mode,
            dog_amgd_strength=dog_amgd_strength,
            use_hldf=use_hldf,
            hldf_num_levels=len(hldf_layers) if hldf_layers is not None and len(hldf_layers) > 0 else 3,
            hldf_hidden_dim=hldf_hidden_dim,
            hldf_use_hq_router=hldf_use_hq_router,
            hldf_router_temp=hldf_router_temp,
            use_center_mask_decoder=use_center_mask_decoder,
            center_gate_alpha=center_gate_alpha,
        ),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        use_ms_fusion=use_ms_fusion,
        use_detail_enhancer=use_detail_enhancer,
        use_hldf=use_hldf,
        use_detail_branch_amgd=use_detail_branch_amgd,
        return_encoder_multi_scale=return_encoder_multi_scale,
    )
    # Optional: initialize from baseline EfficientSAM checkpoint (partial, shape-matched)
    if init_from_baseline is not None and os.path.isfile(init_from_baseline):
        try:
            ckpt = torch.load(init_from_baseline, map_location="cpu")
            if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
                src_sd = ckpt["model"]
            else:
                src_sd = ckpt
            dst_sd = model.state_dict()
            loaded, total = 0, 0
            for k, v in src_sd.items():
                if k in dst_sd and isinstance(v, torch.Tensor) and v.shape == dst_sd[k].shape:
                    dst_sd[k] = v
                    loaded += 1
                total += 1
            model.load_state_dict(dst_sd, strict=False)
            print(f"[build_efficient_sam_hq] Partially loaded {loaded} tensors from baseline ({total} scanned).")
        except Exception as e:
            print(f"[build_efficient_sam_hq] Failed to init from baseline: {e}")
    return model

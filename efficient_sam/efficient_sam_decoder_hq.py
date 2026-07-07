from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPBlock


class DepthwisePointwiseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.GroupNorm(1, in_channels),
            activation(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_channels),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SharedStemUpsample(nn.Module):
    def __init__(
        self,
        *,
        vit_dim: int,
        transformer_dim: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_dim = transformer_dim // 8
        self.block = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, out_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, out_dim),
            activation(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HierarchicalLayerWiseDetailFusion(nn.Module):
    def __init__(
        self,
        *,
        vit_dim: int,
        transformer_dim: int,
        num_levels: int,
        hidden_dim: int = 96,
        use_hq_router: bool = True,
        router_temp: float = 1.0,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.vit_dim = int(vit_dim)
        self.hidden_dim = max(8, int(hidden_dim))
        self.num_levels = max(2, int(num_levels))
        self.use_hq_router = bool(use_hq_router)
        self.router_temp = max(1e-3, float(router_temp))
        self.output_dim = transformer_dim // 8

        self.align_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.vit_dim, self.hidden_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(1, self.hidden_dim),
                    activation(),
                )
                for _ in range(self.num_levels)
            ]
        )
        if self.use_hq_router:
            self.router = MLPBlock(
                input_dim=transformer_dim,
                hidden_dim=max(16, transformer_dim // 2),
                output_dim=self.num_levels + (self.num_levels - 1),
                num_layers=2,
                act=activation,
            )
        else:
            self.router = None
        self.top_down_blocks = nn.ModuleList(
            [
                DepthwisePointwiseBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=3,
                    dilation=1,
                    activation=activation,
                )
                for _ in range(self.num_levels - 1)
            ]
        )
        concat_dim = self.hidden_dim * self.num_levels
        self.local_branch = DepthwisePointwiseBlock(
            concat_dim,
            self.hidden_dim,
            kernel_size=3,
            dilation=1,
            activation=activation,
        )
        self.context_branch = DepthwisePointwiseBlock(
            concat_dim,
            self.hidden_dim,
            kernel_size=3,
            dilation=2,
            activation=activation,
        )
        self.detail_reconstruct = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, transformer_dim // 2, kernel_size=1, bias=False),
            nn.GroupNorm(1, transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, self.output_dim, kernel_size=2, stride=2),
            nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1),
        )

    def _to_nchw(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError("HLDF expects 4D feature maps per layer.")
        if feat.shape[1] == self.vit_dim:
            return feat
        if feat.shape[-1] == self.vit_dim:
            return feat.permute(0, 3, 1, 2)
        raise ValueError(
            f"Unsupported HLDF feature shape {tuple(feat.shape)} for vit_dim={self.vit_dim}."
        )

    def forward(
        self,
        multi_scale_embeddings: List[torch.Tensor],
        hq_token_context: torch.Tensor,
    ) -> torch.Tensor:
        if multi_scale_embeddings is None or len(multi_scale_embeddings) == 0:
            raise ValueError("HLDF requires non-empty multi_scale_embeddings.")
        if len(multi_scale_embeddings) < self.num_levels:
            raise ValueError(
                f"HLDF expects at least {self.num_levels} layer features, got {len(multi_scale_embeddings)}."
            )
        aligned: List[torch.Tensor] = []
        for idx in range(self.num_levels):
            feat_nchw = self._to_nchw(multi_scale_embeddings[idx])
            aligned.append(self.align_layers[idx](feat_nchw))

        bsz = aligned[0].shape[0]
        dtype = aligned[0].dtype
        device = aligned[0].device
        if self.router is not None:
            router_out = self.router(hq_token_context)
            alpha_logits = router_out[:, : self.num_levels] / self.router_temp
            alphas = F.softmax(alpha_logits, dim=-1)
            gates = torch.sigmoid(router_out[:, self.num_levels :])
        else:
            alphas = torch.full(
                (bsz, self.num_levels),
                1.0 / float(self.num_levels),
                device=device,
                dtype=dtype,
            )
            gates = torch.ones((bsz, self.num_levels - 1), device=device, dtype=dtype)

        hier_feats: List[Optional[torch.Tensor]] = [None] * self.num_levels
        current = aligned[-1] * alphas[:, -1].view(bsz, 1, 1, 1)
        hier_feats[-1] = current
        for depth_idx, layer_idx in enumerate(range(self.num_levels - 2, -1, -1)):
            propagated = self.top_down_blocks[depth_idx](current)
            gate = gates[:, depth_idx].view(bsz, 1, 1, 1)
            current = aligned[layer_idx] * alphas[:, layer_idx].view(bsz, 1, 1, 1) + gate * propagated
            hier_feats[layer_idx] = current

        fused = torch.cat([feat for feat in hier_feats if feat is not None], dim=1)
        local_detail = self.local_branch(fused)
        context_detail = self.context_branch(fused)
        detail = torch.cat([local_detail, context_detail], dim=1)
        return self.detail_reconstruct(detail)


class MaskDecoderHQ(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
        use_amgd: bool = False,
        use_dog_amgd: bool = False,
        amgd_routing: str = "prompt",
        amgd_branch_design: str = "legacy",
        amgd_detail_layer: int = 0,
        amgd_structure_layer: int = 0,
        amgd_background_layer: int = 0,
        amgd_branch_layers: Optional[List[int]] = None,
        dog_amgd_mode: str = "legacy",
        dog_amgd_strength: float = 0.25,
        use_hldf: bool = False,
        hldf_num_levels: int = 3,
        hldf_hidden_dim: int = 96,
        hldf_use_hq_router: bool = True,
        hldf_router_temp: float = 1.0,
        use_center_mask_decoder: bool = False,
        center_gate_alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.vit_dim = int(vit_dim)
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Upscaling path for SAM branch
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(transformer_dim, transformer_dim, transformer_dim // 8, 3, act=activation)
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth, act=activation
        )

        # HQ branch: token + MLP + feature fusion
        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim // 8,
            num_layers=3,
            act=activation,
        )
        # increase token count to include HQ token in the transformer outputs
        self.num_mask_tokens = self.num_mask_tokens + 1
        
        self.use_amgd = use_amgd
        self.use_dog_amgd = use_dog_amgd
        self.amgd_routing = str(amgd_routing).lower()
        if self.amgd_routing not in {"prompt", "uniform"}:
            self.amgd_routing = "prompt"
        self.use_hldf = bool(use_hldf)
        self.amgd_branch_design = str(amgd_branch_design).lower()
        if self.amgd_branch_design not in {"legacy", "dsb_v1"}:
            self.amgd_branch_design = "legacy"
        self.use_dsb_amgd = bool(self.use_amgd and self.amgd_branch_design == "dsb_v1")
        self.amgd_detail_layer = int(amgd_detail_layer)
        self.amgd_structure_layer = int(amgd_structure_layer)
        self.amgd_background_layer = int(amgd_background_layer)
        branch_layers = amgd_branch_layers or [
            self.amgd_detail_layer,
            self.amgd_structure_layer,
            self.amgd_background_layer,
        ]
        self.amgd_branch_layers = sorted({max(0, int(idx)) for idx in branch_layers})
        self.dog_amgd_mode = str(dog_amgd_mode).lower()
        if self.dog_amgd_mode not in {"legacy", "residual"}:
            self.dog_amgd_mode = "legacy"
        self.dog_amgd_strength = float(dog_amgd_strength)
        
        # Original single-scale path for when use_amgd is False
        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.amgd_fine = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            # Second upsampling
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=1)
        )

        self.amgd_mid = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            # Second upsampling 
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=3, padding=1)
        )

        self.amgd_coarse = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim),
            activation(),
            # Second upsampling
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
            nn.Conv2d(transformer_dim // 8, transformer_dim // 8, kernel_size=5, padding=2)
        )

        self.amgd_router = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim // 2,
            output_dim=3,
            num_layers=2,
            act=activation,
        )
        if self.use_dsb_amgd:
            branch_dim = transformer_dim // 8
            self.amgd_shared_stem = SharedStemUpsample(
                vit_dim=vit_dim,
                transformer_dim=transformer_dim,
                activation=activation,
            )
            self.amgd_detail_head = nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=1, bias=False),
            )
            self.amgd_structure_head = nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, groups=branch_dim, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=2, dilation=2, groups=branch_dim, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=1, bias=False),
            )
            self.amgd_background_head = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=5, padding=2, groups=branch_dim, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, branch_dim, kernel_size=1, bias=False),
            )
        else:
            self.amgd_shared_stem = None
            self.amgd_detail_head = None
            self.amgd_structure_head = None
            self.amgd_background_head = None
        if self.use_hldf:
            self.hldf = HierarchicalLayerWiseDetailFusion(
                vit_dim=vit_dim,
                transformer_dim=transformer_dim,
                num_levels=max(2, int(hldf_num_levels)),
                hidden_dim=hldf_hidden_dim,
                use_hq_router=hldf_use_hq_router,
                router_temp=hldf_router_temp,
                activation=activation,
            )
        else:
            self.hldf = None
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            nn.GroupNorm(1, transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )
        self.use_center_mask_decoder = bool(use_center_mask_decoder)
        self.center_gate_alpha = float(center_gate_alpha)
        if self.use_center_mask_decoder:
            branch_dim = transformer_dim // 8
            self.center_head = nn.Sequential(
                nn.Conv2d(branch_dim, branch_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(1, branch_dim),
                activation(),
                nn.Conv2d(branch_dim, 1, kernel_size=1),
            )
            self._init_center_head()
        else:
            self.center_head = None

    def _init_center_head(self) -> None:
        if self.center_head is None:
            return
        for mod in self.center_head.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.GroupNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)
        last_conv = self.center_head[-1]
        if isinstance(last_conv, nn.Conv2d) and last_conv.bias is not None:
            nn.init.constant_(last_conv.bias, -2.0)

    def _to_nchw(self, feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.dim() != 4:
            raise ValueError("Expected 4D feature maps for decoder-side detail branches.")
        if feat.shape[1] == expected_channels:
            return feat
        if feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        raise ValueError(
            f"Unsupported feature shape {tuple(feat.shape)} for expected_channels={expected_channels}."
        )

    def _build_dsb_features(
        self,
        detail_branch_embeddings: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if detail_branch_embeddings is None:
            raise ValueError("detail_branch_embeddings must be provided when amgd_branch_design=dsb_v1.")
        if len(detail_branch_embeddings) != len(self.amgd_branch_layers):
            raise ValueError(
                "detail_branch_embeddings length must match amgd_branch_layers in dsb_v1 mode."
            )
        feat_map = {
            int(layer_idx): self._to_nchw(feat, self.vit_dim)
            for layer_idx, feat in zip(self.amgd_branch_layers, detail_branch_embeddings)
        }
        missing = [
            idx
            for idx in [self.amgd_detail_layer, self.amgd_structure_layer, self.amgd_background_layer]
            if idx not in feat_map
        ]
        if missing:
            raise ValueError(f"Missing DSB-AMGD layer features for indices {missing}.")

        detail_stem = self.amgd_shared_stem(feat_map[self.amgd_detail_layer])
        structure_stem = self.amgd_shared_stem(feat_map[self.amgd_structure_layer])
        background_stem = self.amgd_shared_stem(feat_map[self.amgd_background_layer])
        detail = detail_stem + self.amgd_detail_head(detail_stem)
        structure = structure_stem + self.amgd_structure_head(structure_stem)
        background = background_stem + self.amgd_background_head(background_stem)
        return detail, structure, background

    def forward(
        self,
        image_embeddings: torch.Tensor,           # [B,C,H,W]
        image_pe: torch.Tensor,                   # [1,C,H,W]
        sparse_prompt_embeddings: torch.Tensor,   # [B, N, C]
        dense_prompt_embeddings: torch.Tensor,    # [B, C, H, W]
        multimask_output: bool,
        hq_token_only: bool,
        interm_embeddings: torch.Tensor,          # [B, H', W', C] (early ViT grid)
        multi_scale_embeddings: Optional[List[torch.Tensor]] = None,
        detail_branch_embeddings: Optional[List[torch.Tensor]] = None,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Compose image stream with dense prompt (mask) guidance
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)

        # Upscaling features
        upscaled_embedding_sam = self.output_upscaling(src)

        # HQ features fusion
        vit_features = interm_embeddings.permute(0, 3, 1, 2)  # [B, C', H', W']

        if self.use_hldf:
            vit_features_fused = self.hldf(
                multi_scale_embeddings=multi_scale_embeddings,
                hq_token_context=mask_tokens_out[:, -1, :],
            )
        elif self.use_amgd:
            # AMGD Adaptive Multi-Grained Feature Fusion
            hq_token_context = mask_tokens_out[:, -1, :]  # [B, C] extract HQ token
            if self.amgd_routing == "uniform":
                routing_weights = torch.full(
                    (b, 3),
                    1.0 / 3.0,
                    device=hq_token_context.device,
                    dtype=hq_token_context.dtype,
                )
            else:
                routing_weights = F.softmax(self.amgd_router(hq_token_context), dim=-1)  # [B, 3]

            if self.use_dsb_amgd:
                feat_detail, feat_structure, feat_background = self._build_dsb_features(
                    detail_branch_embeddings=detail_branch_embeddings
                )
                feat_fine, feat_mid, feat_coarse = feat_detail, feat_structure, feat_background
            else:
                feat_fine = self.amgd_fine(vit_features)      # [B, C/8, H, W]
                feat_mid = self.amgd_mid(vit_features)        # [B, C/8, H, W]
                feat_coarse = self.amgd_coarse(vit_features)  # [B, C/8, H, W]

            w_base = routing_weights[:, 0].view(b, 1, 1, 1)
            w_opt2 = routing_weights[:, 1].view(b, 1, 1, 1)
            w_opt3 = routing_weights[:, 2].view(b, 1, 1, 1)
            amgd_base = (feat_fine * w_base) + (feat_mid * w_opt2) + (feat_coarse * w_opt3)
            if self.use_dog_amgd:
                dog_fine = feat_fine - feat_mid
                dog_mid = feat_mid - feat_coarse
                if self.dog_amgd_mode == "residual":
                    # Residual mode preserves AMGD base and adds a scaled DoG correction.
                    dog_delta = (dog_fine * w_opt2) + (dog_mid * w_opt3)
                    vit_features_fused = amgd_base + self.dog_amgd_strength * dog_delta
                else:
                    # Legacy mode reproduces the original DoG-AMGD semantics.
                    vit_features_fused = (feat_fine * w_base) + (dog_fine * w_opt2) + (dog_mid * w_opt3)
            else:
                vit_features_fused = amgd_base
        else:
            # Original fallback 
            vit_features_fused = self.compress_vit_feat(vit_features)

        hq_features = self.embedding_encoder(image_embeddings) + vit_features_fused
        # Optional radial frequency gate on HQ features
        if getattr(self, "radial_gate", None) is not None:
            try:
                strength = float(getattr(self, "rgate_strength_dec", 0.5))
                hq_features = hq_features + strength * self.radial_gate(hq_features)
            except Exception:
                pass
        # Optional AFD gate on HQ features
        if getattr(self, "afd_gate", None) is not None:
            try:
                afd_strength = float(getattr(self, "afd_strength_dec", 0.5))
                afd_delta = self.afd_gate(hq_features) - hq_features
                hq_features = hq_features + afd_strength * afd_delta
            except Exception:
                pass
        # Optional MSFE gate on HQ features
        if getattr(self, "msfe_gate", None) is not None:
            try:
                msfe_strength = float(getattr(self, "msfe_strength_dec", 0.5))
                msfe_delta = self.msfe_gate(hq_features) - hq_features
                hq_features = hq_features + msfe_strength * msfe_delta
            except Exception:
                pass
        aux_dict: Dict[str, torch.Tensor] = {}
        if self.center_head is not None:
            center_logits = self.center_head(hq_features)
            center_prob = torch.sigmoid(center_logits)
            center_gate = 1.0 + self.center_gate_alpha * center_prob
            aux_dict["center_logits"] = center_logits
            aux_dict["center_prob"] = center_prob
        else:
            center_gate = None
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features
        if center_gate is not None:
            upscaled_embedding_hq = upscaled_embedding_hq * center_gate

        # Hypernets
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                # last token corresponds to HQ token
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, num_mask_tokens, C']

        b, c2, hh, ww = upscaled_embedding_sam.shape
        # first N-1 are SAM masks, last one is HQ mask
        masks_sam = (hyper_in[:, : self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c2, hh * ww)).view(b, -1, hh, ww)
        masks_hq = (hyper_in[:, self.num_mask_tokens - 1 : self.num_mask_tokens] @ upscaled_embedding_hq.view(b, c2, hh * ww)).view(b, -1, hh, ww)
        masks = torch.cat([masks_sam, masks_hq], dim=1)

        # IoU head
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select outputs
        if multimask_output:
            # choose among multi-mask outputs (exclude first default)
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_sel = iou_pred[:, mask_slice]
            iou_max, max_idx = torch.max(iou_sel, dim=1)
            iou_pred = iou_max.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam_best = masks_multi[torch.arange(masks_multi.size(0)), max_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks_sam_best = masks[:, mask_slice]

        masks_hq_only = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]
        if hq_token_only:
            final_masks = masks_hq_only
        else:
            final_masks = masks_sam_best + masks_hq_only

        if return_aux:
            return final_masks, iou_pred, aux_dict
        return final_masks, iou_pred

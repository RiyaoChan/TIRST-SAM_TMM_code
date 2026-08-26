import torch
from torch import nn

from efficient_sam.microquery import decode_prompt_queries


class _PromptEncoder(nn.Module):
    input_image_size = (32, 32)
    mask_input_size = (8, 8)

    def get_dense_pe(self) -> torch.Tensor:
        return torch.zeros(1, 4, 2, 2)

    def forward(self, *, points, boxes, masks, text_embeds):
        source = points[0] if points is not None else boxes if boxes is not None else masks
        count = int(source.shape[0])
        return torch.zeros(count, 1, 4), torch.zeros(count, 4, 2, 2)


class _MaskDecoder(nn.Module):
    def forward(
        self,
        image_embeddings,
        image_pe,
        *,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
        hq_token_only,
        interm_embeddings,
    ):
        count = int(image_embeddings.shape[0])
        return torch.zeros(count, 1, 4, 4), torch.full((count, 1), 0.5)


class _Model:
    def __init__(self) -> None:
        self.prompt_encoder = _PromptEncoder()
        self.mask_decoder = _MaskDecoder()

    @staticmethod
    def get_rescaled_pts(points: torch.Tensor, input_h: int, input_w: int) -> torch.Tensor:
        return points


def test_oracle_point_box_and_micro_mask_shapes() -> None:
    model = _Model()
    image = torch.zeros(1, 4, 2, 2)
    interm = torch.zeros(1, 2, 2, 4)
    common = dict(
        model=model,
        image_embeddings=image,
        interm_embeddings=interm,
        input_h=32,
        input_w=32,
        output_h=16,
        output_w=16,
        chunk_size=1,
    )
    point = decode_prompt_queries(
        **common,
        points=torch.tensor([[[[4.0, 5.0]], [[8.0, 9.0]]]]),
        point_labels=torch.ones(1, 2, 1, dtype=torch.int64),
    )
    box = decode_prompt_queries(
        **common, boxes=torch.tensor([[[1.0, 2.0, 6.0, 7.0], [3.0, 4.0, 8.0, 9.0]]])
    )
    micro_mask = decode_prompt_queries(**common, masks=torch.zeros(1, 2, 1, 8, 8))
    assert point.mask_logits.shape == box.mask_logits.shape == micro_mask.mask_logits.shape
    assert point.mask_logits.shape == (1, 2, 16, 16)
    assert point.quality.shape == (1, 2)


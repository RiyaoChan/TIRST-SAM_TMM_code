#!/usr/bin/env python
"""Create cached CLIP feature variants for TMM text-sensitivity evaluation.

The input file should be the token-level feature cache produced by
generate_mllm_prompts.py:
  {stem: {token_ids, attention_mask, token_features, global_feat}}

This script never calls the MLLM. It only rewrites cached CLIP features.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

try:
    import clip as clip_module
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False


GENERIC_TEXT = "a small infrared target in a cluttered background"
CONTRADICTORY_TEXT = "there is no target in this infrared image"
RANDOM_TEXTS = [
    "a colorful indoor room with furniture",
    "a close-up photo of a handwritten note",
    "a group of people standing near a road",
    "a bright daytime city street with buildings",
    "a natural landscape with trees and water",
]


def load_feature_cache(path: Path) -> Dict[str, dict]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"Expected a non-empty feature dict: {path}")
    first = next(iter(obj.values()))
    if not isinstance(first, dict) or "token_features" not in first or "global_feat" not in first:
        raise ValueError("Expected token-level feature cache with token_features and global_feat")
    return obj


def zero_like_item(item: dict) -> dict:
    return {
        "token_ids": torch.zeros_like(item["token_ids"]).long(),
        "attention_mask": torch.zeros_like(item["attention_mask"]).long(),
        "token_features": torch.zeros_like(item["token_features"]).to(item["token_features"].dtype),
        "global_feat": torch.zeros_like(item["global_feat"]).to(item["global_feat"].dtype),
    }


def random_like_item(item: dict, generator: torch.Generator) -> dict:
    token_features = torch.randn(
        item["token_features"].shape,
        generator=generator,
        dtype=torch.float32,
    ).to(item["token_features"].dtype)
    global_feat = torch.randn(
        item["global_feat"].shape,
        generator=generator,
        dtype=torch.float32,
    )
    global_feat = global_feat / global_feat.norm().clamp(min=1e-6)
    return {
        "token_ids": item["token_ids"].clone().long(),
        "attention_mask": item["attention_mask"].clone().long(),
        "token_features": token_features,
        "global_feat": global_feat.to(item["global_feat"].dtype),
    }


def encode_texts_with_clip(stem_to_text: Dict[str, str], clip_model: str, device: str, dtype: torch.dtype) -> Dict[str, dict]:
    if not HAS_CLIP:
        raise ImportError(
            "The OpenAI CLIP package is required for generic/blank/contradictory text variants. "
            "Install it or run only no_text/mismatched/random_feature variants."
        )

    model, _ = clip_module.load(clip_model, device=device)
    model.eval()
    stems = list(stem_to_text.keys())
    texts = [stem_to_text[s] for s in stems]
    out: Dict[str, dict] = {}

    for i in range(0, len(stems), 64):
        batch_stems = stems[i:i + 64]
        batch_texts = texts[i:i + 64]
        tokens = clip_module.tokenize(batch_texts, truncate=True).to(device)
        with torch.inference_mode():
            x = model.token_embedding(tokens).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)
            x = model.transformer(x)
            x = x.permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)
            eot_idx = tokens.argmax(dim=-1)
            global_feat = x[torch.arange(x.shape[0], device=tokens.device), eot_idx]
            if getattr(model, "text_projection", None) is not None:
                global_feat = global_feat @ model.text_projection
            global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            attention_mask = (tokens != 0).long()

        for j, stem in enumerate(batch_stems):
            out[stem] = {
                "token_ids": tokens[j].detach().cpu().long(),
                "attention_mask": attention_mask[j].detach().cpu().long(),
                "token_features": x[j].detach().cpu().to(dtype),
                "global_feat": global_feat[j].detach().cpu().to(dtype),
            }

    return out


def rotate_mismatched(features: Dict[str, dict], seed: int) -> Dict[str, dict]:
    stems = list(features.keys())
    shuffled = stems[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) > 1 and all(a == b for a, b in zip(stems, shuffled)):
        shuffled = shuffled[1:] + shuffled[:1]
    return {
        stem: {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in features[src].items()
        }
        for stem, src in zip(stems, shuffled)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Original token-level feature .pt")
    parser.add_argument("--out_dir", required=True, help="Output directory for variant .pt files")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = load_feature_cache(input_path)
    stems = list(features.keys())
    first = features[stems[0]]
    store_dtype = first["token_features"].dtype
    generator = torch.Generator().manual_seed(args.seed)

    variants: Dict[str, Dict[str, dict]] = {
        "no_text": {stem: zero_like_item(features[stem]) for stem in stems},
        "mismatched_caption": rotate_mismatched(features, args.seed),
        "random_caption": {stem: random_like_item(features[stem], generator) for stem in stems},
    }

    constant_texts = {
        "generic_text": GENERIC_TEXT,
        "contradictory_caption": CONTRADICTORY_TEXT,
        "blank_caption": "",
    }
    for name, text in constant_texts.items():
        variants[name] = encode_texts_with_clip(
            {stem: text for stem in stems},
            clip_model=args.clip_model,
            device=args.device,
            dtype=store_dtype,
        )

    manifest = {"original": str(input_path), "variants": {}}
    for name, data in variants.items():
        out_path = out_dir / f"{input_path.stem}__{name}.pt"
        torch.save(data, out_path)
        manifest["variants"][name] = str(out_path)
        print(f"saved {name}: {out_path}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()

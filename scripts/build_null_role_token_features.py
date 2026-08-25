#!/usr/bin/env python
"""Create a same-shape null/no-text role-token cache for matched training controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def null_cache_item(item: Mapping[str, object]) -> dict:
    tokens = item.get("token_features")
    global_feature = item.get("global_feat")
    attention_mask = item.get("attention_mask")
    if not torch.is_tensor(tokens) or tokens.ndim != 2:
        raise ValueError("Structured cache item requires token_features=[L,D].")
    if not torch.is_tensor(global_feature) or global_feature.ndim != 1:
        raise ValueError("Structured cache item requires global_feat=[D].")
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 1:
        raise ValueError("Structured cache item requires attention_mask=[L].")
    if int(tokens.shape[0]) != int(attention_mask.shape[0]):
        raise ValueError("Token length and attention-mask length must match.")

    output = dict(item)
    output["token_features"] = torch.zeros_like(tokens)
    output["global_feat"] = torch.zeros_like(global_feature)
    output["attention_mask"] = torch.zeros_like(attention_mask, dtype=torch.long)
    field_mask = item.get("field_mask")
    output["field_mask"] = (
        torch.zeros_like(field_mask, dtype=torch.long)
        if torch.is_tensor(field_mask)
        else torch.zeros_like(attention_mask, dtype=torch.long)
    )
    output["mask_policy"] = "matched_null_no_text_control"
    output["source_schema_version"] = item.get("schema_version")
    output["schema_version"] = "gpt-structured-role-tokens-null-control-v1"
    return output


def build_null_cache(source: Mapping[str, object]) -> dict:
    output = {}
    for name, item in source.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"Cache entry {name!r} is not structured.")
        output[str(name)] = null_cache_item(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else output_path.with_suffix(".manifest.json")
    if input_path == output_path:
        raise ValueError("Input and output cache paths must differ.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    source = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(source, Mapping) or not source:
        raise ValueError("Input cache must be a non-empty mapping.")
    output = build_null_cache(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    manifest = {
        "schema_version": "tirst-sam-matched-null-role-cache-v1",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "samples": len(output),
        "contract": {
            "token_features": "all zeros; original shape and dtype preserved",
            "global_feat": "all zeros; original shape and dtype preserved",
            "attention_mask": "all zeros",
            "role_metadata": "preserved for audit only; never read by SIRSTDataset model inputs",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

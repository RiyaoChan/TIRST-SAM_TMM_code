#!/usr/bin/env python
"""Build field-isolated CLIP role tokens from GPT structured attributes.

Unlike a conventional CLIP cache that embeds one complete caption, this cache
stores one independently encoded CLIP EOT/global vector per semantic role.  A
field-level attention mask can therefore suppress an unreliable role without
changing the remaining roles.

The intended leakage-free policy is:

* audited training records use GT-derived *verification masks* only;
* records absent from the audit (normally validation/test) use raw GPT fields;
* no GT value is ever substituted for a GPT value.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import clip as clip_module
except ImportError:  # pragma: no cover - exercised by the CLI error path
    clip_module = None


SCHEMA_VERSION = "gpt-structured-role-tokens-v1"
SUPPORTED_ROLES = ("presence", "count", "location", "size")
NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def read_json_object(path: Path) -> Dict[str, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return obj


def parse_roles(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        roles = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    else:
        roles = tuple(str(part).strip().lower() for part in value if str(part).strip())
    if not roles:
        raise ValueError("At least one role is required")
    if len(set(roles)) != len(roles):
        raise ValueError(f"Duplicate roles are not allowed: {roles}")
    unsupported = [role for role in roles if role not in SUPPORTED_ROLES]
    if unsupported:
        raise ValueError(f"Unsupported roles {unsupported}; supported={SUPPORTED_ROLES}")
    return roles


def normalize_attributes(raw: Mapping[str, object]) -> Dict[str, object]:
    present = bool(raw.get("target_present", False))
    try:
        count = max(0, int(raw.get("count", 0)))
    except (TypeError, ValueError):
        count = 0
    if not present:
        count = 0
    elif count == 0:
        count = 1
    return {
        "target_present": present,
        "count": count,
        "position": str(raw.get("position", "unknown") or "unknown").strip().lower(),
        "size": str(raw.get("size", "unknown") or "unknown").strip().lower(),
    }


def raw_field_mask(attributes: Mapping[str, object]) -> Dict[str, int]:
    present = bool(attributes["target_present"])
    position = str(attributes.get("position", "unknown"))
    size = str(attributes.get("size", "unknown"))
    return {
        "presence": 1,
        "count": 1,
        "location": int(present and position not in {"", "none", "unknown", "n/a"}),
        "size": int(present and size not in {"", "none", "unknown", "n/a"}),
    }


def audited_field_mask(
    attributes: Mapping[str, object],
    audit_record: Optional[Mapping[str, object]],
) -> Tuple[Dict[str, int], str]:
    """Return role mask without ever replacing GPT values by GT values."""

    if audit_record is None:
        return raw_field_mask(attributes), "raw_gpt_inference"

    core = audit_record.get("core_condition")
    if isinstance(core, Mapping) and isinstance(core.get("field_mask"), Mapping):
        core_mask = core["field_mask"]
        mask = {role: int(float(core_mask.get(role, 0)) > 0) for role in SUPPORTED_ROLES}
        return mask, str(core.get("status", "audited_core"))

    status = audit_record.get("field_status", {})
    if not isinstance(status, Mapping):
        status = {}
    presence_ok = status.get("presence") == "pass"
    negative_verified = presence_ok and not bool(attributes["target_present"])
    count_ok = negative_verified or status.get("count") == "pass"
    mask = {
        "presence": int(presence_ok),
        "count": int(presence_ok and count_ok),
        "location": int(presence_ok and bool(attributes["target_present"]) and status.get("location") == "pass"),
        "size": int(presence_ok and bool(attributes["target_present"]) and status.get("size") == "pass"),
    }
    if not presence_ok:
        policy = "reject_auto"
    elif count_ok:
        policy = "presence_count_verified_auto"
    else:
        policy = "presence_only_auto"
    return mask, policy


def _natural_value(value: object) -> str:
    return str(value).strip().lower().replace("-", " ").replace("_", " ")


def role_text(role: str, attributes: Mapping[str, object]) -> str:
    if role == "presence":
        value = "present" if bool(attributes["target_present"]) else "absent"
        return f"Infrared small target presence: {value}."
    if role == "count":
        count = int(attributes["count"])
        value = NUMBER_WORDS.get(count, str(count))
        return f"Infrared small target count: {value}."
    if role == "location":
        return f"Infrared small target location: {_natural_value(attributes['position'])}."
    if role == "size":
        return f"Infrared small target size: {_natural_value(attributes['size'])}."
    raise ValueError(f"Unsupported role: {role}")


def encode_unique_texts(
    texts: Iterable[str],
    clip_model: str,
    device: str,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    if clip_module is None:
        raise ImportError("OpenAI CLIP is required. Install the local project CLIP dependency first.")
    unique = sorted(set(texts))
    model, _ = clip_module.load(clip_model, device=device)
    model.eval()
    output: Dict[str, torch.Tensor] = {}
    for start in range(0, len(unique), batch_size):
        batch_texts = unique[start : start + batch_size]
        tokens = clip_module.tokenize(batch_texts, truncate=True).to(device)
        with torch.inference_mode():
            features = model.encode_text(tokens).float()
            features = F.normalize(features, dim=-1)
        for text, feature in zip(batch_texts, features):
            output[text] = feature.detach().cpu()
    return output


def assemble_cache_item(
    roles: Sequence[str],
    attributes: Mapping[str, object],
    field_mask: Mapping[str, int],
    text_embeddings: Mapping[str, torch.Tensor],
    store_dtype: torch.dtype,
    policy: str,
) -> dict:
    texts = [role_text(role, attributes) for role in roles]
    mask = torch.tensor([int(field_mask.get(role, 0) > 0) for role in roles], dtype=torch.long)
    token_features = torch.stack([text_embeddings[text].float() for text in texts], dim=0)
    token_features = F.normalize(token_features, dim=-1)
    token_features = token_features * mask.unsqueeze(-1).to(token_features.dtype)
    if int(mask.sum()) > 0:
        global_feat = token_features.sum(dim=0) / mask.sum().to(token_features.dtype)
        global_feat = F.normalize(global_feat.unsqueeze(0), dim=-1).squeeze(0)
    else:
        global_feat = torch.zeros(token_features.shape[-1], dtype=token_features.dtype)
    role_values = {
        "presence": bool(attributes["target_present"]),
        "count": int(attributes["count"]),
        "location": str(attributes["position"]),
        "size": str(attributes["size"]),
    }
    return {
        "token_ids": torch.arange(1, len(roles) + 1, dtype=torch.long),
        "attention_mask": mask,
        "token_features": token_features.to(store_dtype),
        "global_feat": global_feat.to(store_dtype),
        "role_names": list(roles),
        "role_texts": texts,
        "role_values": role_values,
        "field_mask": mask.clone(),
        "mask_policy": policy,
        "schema_version": SCHEMA_VERSION,
    }


def build_cache(
    attributes: Mapping[str, Mapping[str, object]],
    audits: Mapping[str, Mapping[str, object]],
    roles: Sequence[str],
    text_embeddings: Mapping[str, torch.Tensor],
    store_dtype: torch.dtype,
) -> Tuple[Dict[str, dict], Counter]:
    cache: Dict[str, dict] = {}
    stats: Counter = Counter()
    for stem in sorted(attributes):
        normalized = normalize_attributes(attributes[stem])
        mask, policy = audited_field_mask(normalized, audits.get(stem))
        cache[stem] = assemble_cache_item(
            roles=roles,
            attributes=normalized,
            field_mask=mask,
            text_embeddings=text_embeddings,
            store_dtype=store_dtype,
            policy=policy,
        )
        stats[f"policy::{policy}"] += 1
        stats[f"active_roles::{int(cache[stem]['attention_mask'].sum())}"] += 1
        for role, active in zip(roles, cache[stem]["attention_mask"].tolist()):
            stats[f"role::{role}::active"] += int(active)
    return cache, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attributes", required=True, help="GPT structured attributes JSON")
    parser.add_argument("--audit_json", default=None, help="Training-only attribute audit JSON")
    parser.add_argument("--output", required=True, help="Output role-token .pt cache")
    parser.add_argument("--roles", default="presence,count", help="Comma-separated ordered roles")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    attributes_path = Path(args.attributes)
    audit_path = Path(args.audit_json) if args.audit_json else None
    output_path = Path(args.output)
    roles = parse_roles(args.roles)
    attributes = read_json_object(attributes_path)
    audits = read_json_object(audit_path) if audit_path else {}

    normalized = {stem: normalize_attributes(item) for stem, item in attributes.items()}
    all_texts = [role_text(role, item) for item in normalized.values() for role in roles]
    text_embeddings = encode_unique_texts(
        all_texts,
        clip_model=args.clip_model,
        device=args.device,
        batch_size=max(1, args.batch_size),
    )
    store_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    cache, stats = build_cache(
        attributes=attributes,
        audits=audits,
        roles=roles,
        text_embeddings=text_embeddings,
        store_dtype=store_dtype,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "attributes": str(attributes_path.resolve()),
        "audit_json": str(audit_path.resolve()) if audit_path else None,
        "audit_scope": "training records only; records absent from audit use raw GPT fields",
        "roles": list(roles),
        "clip_model": args.clip_model,
        "dtype": args.dtype,
        "items": len(cache),
        "unique_role_texts": len(text_embeddings),
        "stats": dict(sorted(stats.items())),
        "output": str(output_path.resolve()),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

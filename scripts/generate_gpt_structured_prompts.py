#!/usr/bin/env python
"""Generate reproducible GPT-5.6 semantic teachers for IRSTD images.

Pipeline:
  1. Infrared image -> GPT-5.6 Sol -> strict structured attributes.
  2. Structured attributes -> deterministic fixed-template caption.
  3. Fixed caption -> existing CLIP text encoder -> token/global features.

The OpenAI API key is read from an environment variable and is never accepted
as a command-line argument. The script can be inspected and dry-run before a
key is configured.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from PIL import Image


SCHEMA_VERSION = "irstd-semantic-v1"
TEMPLATE_VERSION = "irstd-fixed-caption-v1"
DEFAULT_MODEL = "gpt-5.6-sol"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
API_NATIVE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

POSITIONS = (
    "none",
    "upper-left",
    "upper-center",
    "upper-right",
    "center-left",
    "center",
    "center-right",
    "lower-left",
    "lower-center",
    "lower-right",
    "multiple-regions",
    "uncertain",
)
SIZES = ("none", "tiny", "small", "mixed", "uncertain")
SHAPES = (
    "none",
    "point-like",
    "round",
    "elongated",
    "irregular",
    "aircraft-like",
    "ship-like",
    "vehicle-like",
    "mixed",
    "uncertain",
)
BACKGROUNDS = (
    "uniform",
    "cloud-cluttered",
    "sea-sky",
    "ground-cluttered",
    "urban",
    "mountainous",
    "complex-textured",
    "low-contrast",
    "unknown",
)
CONTRASTS = ("very-low", "low", "moderate", "high", "very-high", "uncertain")


ATTRIBUTE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_present": {"type": "boolean"},
        "count": {"type": "integer"},
        "position": {"type": "string", "enum": list(POSITIONS)},
        "size": {"type": "string", "enum": list(SIZES)},
        "shape": {"type": "string", "enum": list(SHAPES)},
        "background": {"type": "string", "enum": list(BACKGROUNDS)},
        "contrast": {"type": "string", "enum": list(CONTRASTS)},
        "confidence": {"type": "number"},
    },
    "required": [
        "target_present",
        "count",
        "position",
        "size",
        "shape",
        "background",
        "contrast",
        "confidence",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are an expert analyst for infrared small-target detection (IRSTD).
Inspect only the image pixels. Do not infer anything from the filename, dataset name, or
unseen metadata. A valid target may occupy only a few pixels, and bright background clutter
must not automatically be labeled as a target. The image may contain no defensible target.

Return the requested structured attributes only. Use target_present=false when no target can
be supported by visible evidence. When target_present=false, set count=0 and set position,
size, and shape to 'none'. Use 'uncertain' instead of inventing a precise attribute. The
confidence field is a self-assessed value from 0 to 1 and is not a calibrated probability."""

USER_PROMPT = """Analyze this infrared image for small targets. Classify the dominant target
attributes and background using the required schema. If multiple targets occupy different
areas, use position='multiple-regions' and describe their dominant size and shape."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_image_names(data_root: Path, split_txts: Iterable[str]) -> List[str]:
    names = set()
    for split_txt in split_txts:
        split_path = Path(split_txt)
        if not split_path.is_absolute():
            split_path = data_root / split_path
        if not split_path.is_file():
            print(f"[warn] Split file not found: {split_path}", file=sys.stderr)
            continue
        with split_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                name = line.strip()
                if name and not name.startswith("#"):
                    names.add(name)
    return sorted(names)


def collect_all_image_names(image_dir: Path) -> List[str]:
    """Collect every supported image in a directory, guarding against stem collisions."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    names_by_stem: Dict[str, str] = {}
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        previous = names_by_stem.get(path.stem)
        if previous is not None and previous != path.name:
            raise RuntimeError(
                f"Multiple image files share stem {path.stem!r}: {previous!r}, {path.name!r}"
            )
        names_by_stem[path.stem] = path.name
    return sorted(names_by_stem.values())


def find_image(image_dir: Path, name: str) -> Optional[Path]:
    original = image_dir / name
    if original.is_file():
        return original
    stem = Path(name).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def image_to_data_url(image_path: Path) -> str:
    extension = image_path.suffix.lower()
    mime_type = API_NATIVE_MIME_TYPES.get(extension)
    if mime_type is not None:
        payload = image_path.read_bytes()
    else:
        # Convert BMP/TIFF and other Pillow-readable inputs to API-safe PNG in memory.
        with Image.open(image_path) as image:
            converted = image.convert("RGB")
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG", compress_level=1)
            payload = buffer.getvalue()
        mime_type = "image/png"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _enum_value(value: Any, allowed: Iterable[str], fallback: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in set(allowed) else fallback


def normalize_attributes(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"Structured attributes must be an object, got {type(raw).__name__}")

    present = _as_bool(raw.get("target_present", False))
    try:
        count = max(0, int(raw.get("count", 0)))
    except (TypeError, ValueError):
        count = 0
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    background = _enum_value(raw.get("background", "unknown"), BACKGROUNDS, "unknown")
    contrast = _enum_value(raw.get("contrast", "uncertain"), CONTRASTS, "uncertain")

    if not present:
        return {
            "target_present": False,
            "count": 0,
            "position": "none",
            "size": "none",
            "shape": "none",
            "background": background,
            "contrast": contrast,
            "confidence": confidence,
        }

    position = _enum_value(raw.get("position", "uncertain"), POSITIONS, "uncertain")
    size = _enum_value(raw.get("size", "uncertain"), SIZES, "uncertain")
    shape = _enum_value(raw.get("shape", "uncertain"), SHAPES, "uncertain")

    return {
        "target_present": True,
        "count": max(1, count),
        "position": "uncertain" if position == "none" else position,
        "size": "uncertain" if size == "none" else size,
        "shape": "uncertain" if shape == "none" else shape,
        "background": background,
        "contrast": contrast,
        "confidence": confidence,
    }


COUNT_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
POSITION_PHRASES = {
    "upper-left": "in the upper-left region",
    "upper-center": "in the upper-center region",
    "upper-right": "in the upper-right region",
    "center-left": "in the center-left region",
    "center": "in the central region",
    "center-right": "in the center-right region",
    "lower-left": "in the lower-left region",
    "lower-center": "in the lower-center region",
    "lower-right": "in the lower-right region",
    "multiple-regions": "across multiple image regions",
    "uncertain": "at an uncertain image location",
}
SIZE_WORDS = {"tiny": "tiny", "small": "small", "mixed": "mixed-size", "uncertain": "small"}
SHAPE_WORDS = {
    "point-like": "point-like",
    "round": "round",
    "elongated": "elongated",
    "irregular": "irregular",
    "aircraft-like": "aircraft-like",
    "ship-like": "ship-like",
    "vehicle-like": "vehicle-like",
    "mixed": "mixed-shape",
    "uncertain": "indistinct",
}
BACKGROUND_PHRASES = {
    "uniform": "a uniform background",
    "cloud-cluttered": "a cloud-cluttered background",
    "sea-sky": "a sea-sky background",
    "ground-cluttered": "a ground-cluttered background",
    "urban": "an urban background",
    "mountainous": "a mountainous background",
    "complex-textured": "a complex-textured background",
    "low-contrast": "a low-contrast background",
    "unknown": "an unclassified infrared background",
}


def render_fixed_caption(attributes: Mapping[str, Any]) -> str:
    item = normalize_attributes(attributes)
    background = BACKGROUND_PHRASES[item["background"]]
    contrast = item["contrast"]

    if not item["target_present"]:
        suffix = "" if contrast == "uncertain" else f" with {contrast} scene contrast"
        return f"No infrared small target is visible against {background}{suffix}."

    count = int(item["count"])
    count_text = COUNT_WORDS.get(count, str(count))
    noun = "target" if count == 1 else "targets"
    verb = "is" if count == 1 else "are"
    size = SIZE_WORDS[item["size"]]
    shape = SHAPE_WORDS[item["shape"]]
    position = POSITION_PHRASES[item["position"]]
    contrast_clause = (
        "" if contrast == "uncertain" else f" with {contrast} target-to-background contrast"
    )
    return (
        f"{count_text} {size} {shape} infrared {noun} {verb} visible {position} "
        f"against {background}{contrast_clause}."
    )


def read_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")


def response_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result: Dict[str, int] = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = int(value)
    return result


def resolve_api_base_url(base_url: Optional[str], base_url_env: str) -> Optional[str]:
    value = (base_url or os.environ.get(base_url_env, "")).strip()
    if not value:
        return None
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid API base URL: {value!r}")
    path = parts.path.rstrip("/")
    if not path:
        path = "/v1"
    normalized = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
    return normalized.rstrip("/") + "/"


def build_openai_client(
    api_key_env: str,
    timeout_seconds: float,
    base_url: Optional[str] = None,
    base_url_env: str = "OPENAI_BASE_URL",
):
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set. Configure it locally; "
            "do not pass API keys on the command line or commit them to the repository."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise ImportError("The OpenAI Python SDK is required. Install requirements.txt first.") from error
    client_options: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    resolved_base_url = resolve_api_base_url(base_url, base_url_env)
    if resolved_base_url is not None:
        client_options["base_url"] = resolved_base_url
    return OpenAI(**client_options)


def request_attributes(client: Any, image_path: Path, args: argparse.Namespace):
    request: Dict[str, Any] = {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": USER_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
                        "detail": args.image_detail,
                    },
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "irstd_semantic_attributes",
                "schema": ATTRIBUTE_SCHEMA,
                "strict": True,
            }
        },
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "store": False,
    }
    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", "") or ""
    if not output_text.strip():
        status = getattr(response, "status", "unknown")
        raise RuntimeError(f"GPT response contained no structured text (status={status})")
    parsed = json.loads(output_text)
    return normalize_attributes(parsed), response


def is_non_retryable_api_error(error: Exception) -> bool:
    """Identify authentication/billing failures that backoff cannot resolve."""
    status_code = getattr(error, "status_code", None)
    error_code = str(getattr(error, "code", "") or "").lower()
    message = str(error).lower()
    permanent_markers = (
        "credit_balance_exhausted",
        "insufficient_quota",
        "no credits remaining",
        "invalid_api_key",
        "incorrect api key",
    )
    return status_code in {401, 403} or any(
        marker in error_code or marker in message for marker in permanent_markers
    )


def request_with_retries(client: Any, image_path: Path, args: argparse.Namespace):
    last_error: Optional[Exception] = None
    for attempt in range(1, args.max_retries + 2):
        try:
            return request_attributes(client, image_path, args)
        except Exception as error:  # SDK and schema errors share the same bounded retry policy.
            last_error = error
            if is_non_retryable_api_error(error):
                print(
                    f"[error] {image_path.name}: non-retryable API authentication/billing failure: "
                    f"{error}",
                    file=sys.stderr,
                )
                break
            if attempt > args.max_retries:
                break
            delay = min(args.max_retry_delay, args.retry_base_seconds * (2 ** (attempt - 1)))
            print(
                f"[warn] {image_path.name}: attempt {attempt} failed: {error}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def atomic_torch_save(torch_module: Any, value: Any, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch_module.save(value, temporary)
    os.replace(temporary, path)


def run_clip_stage(
    descriptions: Dict[str, str],
    token_feature_path: Path,
    global_feature_path: Path,
    args: argparse.Namespace,
) -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import torch
    from generate_mllm_prompts import (
        encode_texts_with_clip,
        encode_texts_with_clip_token_features,
    )

    print(f"Encoding {len(descriptions)} fixed captions with CLIP {args.clip_model} ...")
    token_features = encode_texts_with_clip_token_features(
        descriptions,
        clip_model_name=args.clip_model,
        device=args.clip_device,
        batch_size=args.clip_batch_size,
        store_dtype=args.clip_token_store_dtype,
    )
    atomic_torch_save(torch, token_features, token_feature_path)
    print(f"Saved token/global CLIP teacher features: {token_feature_path}")

    if args.save_global_clip:
        global_features = encode_texts_with_clip(
            descriptions,
            clip_model_name=args.clip_model,
            device=args.clip_device,
            batch_size=args.clip_batch_size,
        )
        atomic_torch_save(torch, global_features, global_feature_path)
        print(f"Saved standalone global CLIP teacher features: {global_feature_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate GPT-5.6 structured IRSTD attributes, fixed captions, and CLIP teachers."
    )
    parser.add_argument("--data_root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--split_txt",
        nargs="+",
        default=["50_50/train.txt", "50_50/test.txt"],
        help="Split files, relative to data_root unless absolute",
    )
    parser.add_argument("--img_dir", default="images", help="Image directory relative to data_root")
    parser.add_argument(
        "--all_images",
        action="store_true",
        help="Process every supported file in img_dir instead of collecting names from split files",
    )
    parser.add_argument("--output_dir", default=None, help="Output directory (default: data_root)")
    parser.add_argument("--prefix", default="gpt5p6", help="Prefix for all generated artifacts")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI vision model ID")
    parser.add_argument(
        "--image_detail", default="original", choices=["low", "high", "original", "auto"]
    )
    parser.add_argument(
        "--reasoning_effort",
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--max_output_tokens", type=int, default=300)
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--base_url",
        default=None,
        help="OpenAI-compatible API root; a bare host is normalized to /v1",
    )
    parser.add_argument("--base_url_env", default="OPENAI_BASE_URL")
    parser.add_argument("--timeout_seconds", type=float, default=180.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--retry_base_seconds", type=float, default=2.0)
    parser.add_argument("--max_retry_delay", type=float, default=60.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent GPT requests (default: 1; use a small value allowed by the gateway)",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print one successful/resumed progress line every N completed records",
    )
    parser.add_argument("--max_images", type=int, default=0, help="Limit images for a smoke test (0=all)")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing structured outputs")
    parser.add_argument(
        "--skip_gpt",
        action="store_true",
        help="Reuse an existing structured-attributes JSON and only render/encode it",
    )
    parser.add_argument("--skip_clip", action="store_true", help="Stop after writing fixed captions")
    parser.add_argument("--save_global_clip", action="store_true")
    parser.add_argument("--clip_model", default="ViT-B/32")
    parser.add_argument("--clip_device", default="cpu")
    parser.add_argument("--clip_batch_size", type=int, default=64)
    parser.add_argument(
        "--clip_token_store_dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Write captions/CLIP features for successful records even if some images fail",
    )
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths and print the plan without requiring an API key or writing outputs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_retries < 0:
        raise ValueError("--max_retries must be >= 0")
    if args.save_every <= 0:
        raise ValueError("--save_every must be > 0")
    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if args.print_every <= 0:
        raise ValueError("--print_every must be > 0")
    if args.fail_fast and args.workers != 1:
        raise ValueError("--fail_fast requires --workers 1")

    data_root = Path(args.data_root).expanduser().resolve()
    image_dir = (data_root / args.img_dir).resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_root
    names = (
        collect_all_image_names(image_dir)
        if args.all_images
        else collect_image_names(data_root, args.split_txt)
    )
    if args.max_images > 0:
        names = names[: args.max_images]
    if not names:
        raise RuntimeError("No image names were collected from the requested split files")

    attributes_path = output_dir / f"{args.prefix}_structured_attributes.json"
    descriptions_path = output_dir / f"{args.prefix}_fixed_descriptions.json"
    log_path = output_dir / f"{args.prefix}_generation_log.jsonl"
    manifest_path = output_dir / f"{args.prefix}_generation_manifest.json"
    token_feature_path = output_dir / f"{args.prefix}_clip_token_features.pt"
    global_feature_path = output_dir / f"{args.prefix}_clip_features.pt"

    print(f"Dataset: {data_root}")
    source_description = "the image directory" if args.all_images else f"{len(args.split_txt)} split file(s)"
    print(f"Images: {len(names)} from {source_description}")
    print(f"Model: {args.model}; detail={args.image_detail}; reasoning={args.reasoning_effort}")
    resolved_base_url = resolve_api_base_url(args.base_url, args.base_url_env)
    print(f"API base URL: {resolved_base_url or 'OpenAI official default'}")
    print(f"Attributes: {attributes_path}")
    print(f"Fixed captions: {descriptions_path}")
    print(f"CLIP token/global features: {token_feature_path}")
    if args.max_images > 0 and args.prefix == "gpt5p6":
        print("[warn] A smoke-test limit is active; consider --prefix gpt5p6_smoke")
    if args.dry_run:
        print("Dry run complete; no API request was made and no output was written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    attributes: Dict[str, Any] = {} if args.overwrite else read_json_object(attributes_path)
    failures: List[str] = []
    completed_since_save = 0
    client = (
        None
        if args.skip_gpt
        else build_openai_client(
            args.api_key_env,
            args.timeout_seconds,
            base_url=args.base_url,
            base_url_env=args.base_url_env,
        )
    )

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "template_version": TEMPLATE_VERSION,
        "updated_at": utc_now(),
        "model": args.model,
        "image_detail": args.image_detail,
        "reasoning_effort": args.reasoning_effort,
        "workers": args.workers,
        "store": False,
        "api_base_url": resolved_base_url or "OpenAI official default",
        "dataset": data_root.name,
        "image_dir": args.img_dir,
        "all_images": args.all_images,
        "split_txt": list(args.split_txt),
        "selected_images": len(names),
        "attribute_schema": ATTRIBUTE_SCHEMA,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
    }

    pending_requests = []
    for index, name in enumerate(names, start=1):
        stem = Path(name).stem
        if stem in attributes and not args.overwrite:
            try:
                attributes[stem] = normalize_attributes(attributes[stem])
                if index % args.print_every == 0 or index == len(names):
                    print(f"[{index}/{len(names)}] resume {stem}")
                continue
            except Exception:
                print(f"[warn] Existing record is invalid and will be regenerated: {stem}")

        if args.skip_gpt:
            failures.append(stem)
            append_jsonl(
                log_path,
                {"timestamp": utc_now(), "stem": stem, "status": "missing_existing_attribute"},
            )
            if args.fail_fast:
                break
            continue

        image_path = find_image(image_dir, name)
        if image_path is None:
            failures.append(stem)
            append_jsonl(
                log_path,
                {"timestamp": utc_now(), "stem": stem, "status": "image_not_found", "name": name},
            )
            if args.fail_fast:
                break
            continue

        pending_requests.append((index, stem, image_path))

    def generate_one(request_info):
        index, stem, image_path = request_info
        try:
            item, response = request_with_retries(client, image_path, args)
            return {
                "status": "ok",
                "index": index,
                "stem": stem,
                "image_path": image_path,
                "item": item,
                "response": response,
            }
        except Exception as error:
            return {
                "status": "error",
                "index": index,
                "stem": stem,
                "image_path": image_path,
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
                "non_retryable": is_non_retryable_api_error(error),
            }

    def iter_results():
        if args.workers == 1:
            for request_info in pending_requests:
                yield generate_one(request_info)
            return
        print(
            f"Submitting {len(pending_requests)} pending image(s) with {args.workers} workers ..."
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            request_iterator = iter(pending_requests)
            futures = set()
            for _ in range(min(args.workers, len(pending_requests))):
                futures.add(executor.submit(generate_one, next(request_iterator)))

            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                batch_results = [future.result() for future in done]
                stop_scheduling = any(
                    result["status"] == "error" and result.get("non_retryable", False)
                    for result in batch_results
                )
                for result in batch_results:
                    yield result

                if stop_scheduling:
                    for future in futures:
                        future.cancel()
                    print(
                        "[abort] A non-retryable authentication/billing error stopped new API "
                        "requests. Completed records remain checkpointed.",
                        file=sys.stderr,
                    )
                    return

                for _ in range(len(done)):
                    try:
                        request_info = next(request_iterator)
                    except StopIteration:
                        break
                    futures.add(executor.submit(generate_one, request_info))

    for result in iter_results():
        index = result["index"]
        stem = result["stem"]
        image_path = result["image_path"]
        if result["status"] == "ok":
            item = result["item"]
            response = result["response"]
            attributes[stem] = item
            caption = render_fixed_caption(item)
            completed_since_save += 1
            append_jsonl(
                log_path,
                {
                    "timestamp": utc_now(),
                    "stem": stem,
                    "image": image_path.name,
                    "status": "ok",
                    "model": args.model,
                    "response_id": getattr(response, "id", None),
                    "usage": response_usage(response),
                    "attributes": item,
                    "fixed_caption": caption,
                },
            )
            completed_total = len(attributes)
            if completed_total % args.print_every == 0 or completed_total == len(names):
                print(f"[{completed_total}/{len(names)}] latest={stem}: {caption}")
            if completed_since_save >= args.save_every:
                atomic_write_json(attributes_path, attributes)
                completed_since_save = 0
        else:
            failures.append(stem)
            append_jsonl(
                log_path,
                {
                    "timestamp": utc_now(),
                    "stem": stem,
                    "image": image_path.name,
                    "status": "error",
                    "error_type": result["error_type"],
                    "error": result["error"],
                },
            )
            print(f"[{index}/{len(names)}] ERROR {stem}: {result['error']}", file=sys.stderr)
            if args.fail_fast:
                break

    atomic_write_json(attributes_path, attributes)

    selected_descriptions: Dict[str, str] = {}
    for name in names:
        stem = Path(name).stem
        if stem in attributes:
            selected_descriptions[stem] = render_fixed_caption(attributes[stem])
    atomic_write_json(descriptions_path, selected_descriptions)

    manifest.update(
        {
            "updated_at": utc_now(),
            "completed_images": len(selected_descriptions),
            "failed_images": failures,
            "attributes_path": attributes_path.name,
            "descriptions_path": descriptions_path.name,
            "token_feature_path": token_feature_path.name,
        }
    )
    atomic_write_json(manifest_path, manifest)

    if failures and not args.allow_partial:
        raise RuntimeError(
            f"{len(failures)} image(s) are incomplete. Outputs were checkpointed; rerun to resume, "
            "or use --allow_partial only for an intentional partial artifact."
        )
    if not selected_descriptions:
        raise RuntimeError("No fixed captions are available for CLIP encoding")

    if args.skip_clip:
        print("Skipping CLIP stage by request.")
    else:
        run_clip_stage(
            selected_descriptions,
            token_feature_path=token_feature_path,
            global_feature_path=global_feature_path,
            args=args,
        )

    print(
        f"Completed {len(selected_descriptions)}/{len(names)} selected images; "
        f"failures={len(failures)}."
    )


if __name__ == "__main__":
    main()

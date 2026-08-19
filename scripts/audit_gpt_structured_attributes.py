#!/usr/bin/env python3
"""Audit GPT IRSTD attributes against training masks and build a review queue.

The script never mutates the source GPT/CLIP artifact.  It supports two source
modes:

1. A structured-attributes JSON produced by ``generate_gpt_structured_prompts``.
2. A cached CLIP token-feature ``.pt`` file.  In this fallback mode, the exact
   fixed caption seen by CLIP is recovered from ``token_ids`` with the public
   OpenAI CLIP BPE vocabulary, then parsed back into *effective* attributes.

Only the requested split is audited.  Use a training split when masks are used
for filtering; do not use test masks to create deployable text conditions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


SCRIPT_VERSION = "gpt-attribute-audit-v2-core"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

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

POSITION_PHRASES = {
    "in the upper-left region": "upper-left",
    "in the upper-center region": "upper-center",
    "in the upper-right region": "upper-right",
    "in the center-left region": "center-left",
    "in the central region": "center",
    "in the center-right region": "center-right",
    "in the lower-left region": "lower-left",
    "in the lower-center region": "lower-center",
    "in the lower-right region": "lower-right",
    "across multiple image regions": "multiple-regions",
    "at an uncertain image location": "uncertain",
}
BACKGROUND_PHRASES = {
    "a uniform background": "uniform",
    "a cloud-cluttered background": "cloud-cluttered",
    "a sea-sky background": "sea-sky",
    "a ground-cluttered background": "ground-cluttered",
    "an urban background": "urban",
    "a mountainous background": "mountainous",
    "a complex-textured background": "complex-textured",
    "a low-contrast background": "low-contrast",
    "an unclassified infrared background": "unknown",
}
COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
CONTRAST_INDEX = {name: index for index, name in enumerate(CONTRASTS[:-1])}

FIELD_NAMES = ("presence", "count", "location", "size", "shape", "background", "contrast")
CRITICAL_FIELDS = ("presence", "count", "location", "size")
MANUAL_CHOICES = ("", "correct", "wrong", "uncertain", "annotation_issue")
FINAL_CHOICES = ("", "accept", "partial", "reject", "annotation_issue")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    temporary.replace(path)


def read_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def bytes_to_unicode() -> Dict[int, str]:
    """Byte-to-Unicode map used by the public OpenAI CLIP tokenizer."""
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(values, [chr(value) for value in chars]))


class ClipTokenDecoder:
    def __init__(self, bpe_vocab: Path) -> None:
        byte_encoder = bytes_to_unicode()
        self.byte_decoder = {value: key for key, value in byte_encoder.items()}
        with gzip.open(bpe_vocab, "rt", encoding="utf-8") as stream:
            lines = stream.read().split("\n")
        # This is the same merge slice used by OpenAI CLIP SimpleTokenizer.
        merge_lines = lines[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(line.split()) for line in merge_lines if line.strip()]
        vocab = list(byte_encoder.values())
        vocab += [value + "</w>" for value in byte_encoder.values()]
        vocab += ["".join(merge) for merge in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.decoder = {index: value for index, value in enumerate(vocab)}

    def decode(self, token_ids: Sequence[int], attention_mask: Optional[Sequence[int]]) -> str:
        if attention_mask is None:
            valid_ids = list(token_ids)
        else:
            valid_ids = [token for token, keep in zip(token_ids, attention_mask) if int(keep) != 0]
        pieces = []
        for token in valid_ids:
            piece = self.decoder.get(int(token))
            if piece is None:
                raise KeyError(f"Token ID {token} is absent from the CLIP BPE vocabulary")
            if piece not in ("<|startoftext|>", "<|endoftext|>"):
                pieces.append(piece)
        encoded = "".join(pieces)
        raw = bytearray(self.byte_decoder[char] for char in encoded)
        text = raw.decode("utf-8", errors="replace").replace("</w>", " ")
        return normalize_decoded_caption(text)


def normalize_decoded_caption(text: str) -> str:
    text = re.sub(r"\s*-\s*", "-", text.strip().lower())
    # CLIP's byte-pair decoder can emit multi-digit numbers as separate digit
    # tokens (for example, ``14`` becomes ``1 4``).  Rejoin adjacent digits
    # before parsing the deterministic count prefix.
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_attribute_mapping(raw: Mapping[str, Any]) -> Dict[str, Any]:
    present = bool(raw.get("target_present", False))
    try:
        count = max(0, int(raw.get("count", 0)))
    except (TypeError, ValueError):
        count = 0
    try:
        confidence = float(raw.get("confidence", float("nan")))
    except (TypeError, ValueError):
        confidence = float("nan")

    if not present:
        return {
            "target_present": False,
            "count": 0,
            "position": "none",
            "size": "none",
            "shape": "none",
            "background": str(raw.get("background", "unknown")),
            "contrast": str(raw.get("contrast", "uncertain")),
            "confidence": confidence,
        }
    return {
        "target_present": True,
        "count": max(1, count),
        "position": str(raw.get("position", "uncertain")),
        "size": str(raw.get("size", "uncertain")),
        "shape": str(raw.get("shape", "uncertain")),
        "background": str(raw.get("background", "unknown")),
        "contrast": str(raw.get("contrast", "uncertain")),
        "confidence": confidence,
    }


def parse_fixed_caption(caption: str) -> Dict[str, Any]:
    """Parse the deterministic caption emitted by the GPT generation script."""
    text = normalize_decoded_caption(caption)
    background = next(
        (value for phrase, value in BACKGROUND_PHRASES.items() if phrase in text),
        "unknown",
    )
    contrast = "uncertain"
    for value in CONTRASTS[:-1]:
        if f"with {value} target-to-background contrast" in text or f"with {value} scene contrast" in text:
            contrast = value
            break

    if text.startswith("no infrared small target is visible"):
        return {
            "target_present": False,
            "count": 0,
            "position": "none",
            "size": "none",
            "shape": "none",
            "background": background,
            "contrast": contrast,
            "confidence": float("nan"),
        }

    first = text.split(" ", 1)[0]
    if first in COUNT_WORDS:
        count = COUNT_WORDS[first]
    else:
        try:
            count = max(1, int(first))
        except ValueError as error:
            raise ValueError(f"Cannot parse target count from fixed caption: {caption!r}") from error

    match = re.match(
        r"^(?:one|two|three|four|five|\d+) "
        r"(?P<size>tiny|small|mixed-size) "
        r"(?P<shape>point-like|round|elongated|irregular|aircraft-like|ship-like|vehicle-like|mixed-shape|indistinct) "
        r"infrared targets? (?:is|are) visible ",
        text,
    )
    if match is None:
        raise ValueError(f"Cannot parse effective size/shape from fixed caption: {caption!r}")
    size_text = match.group("size")
    size = "mixed" if size_text == "mixed-size" else size_text
    shape_text = match.group("shape")
    shape = {"mixed-shape": "mixed", "indistinct": "uncertain"}.get(shape_text, shape_text)
    position = next((value for phrase, value in POSITION_PHRASES.items() if phrase in text), "uncertain")
    return {
        "target_present": True,
        "count": count,
        "position": position,
        "size": size,
        "shape": shape,
        "background": background,
        "contrast": contrast,
        "confidence": float("nan"),
    }


def load_effective_attributes(
    attributes_json: Optional[Path],
    features_path: Optional[Path],
    bpe_vocab: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], str]:
    if attributes_json is not None:
        raw = read_json_object(attributes_json)
        attributes = {
            str(stem): _normalize_attribute_mapping(item)
            for stem, item in raw.items()
            if isinstance(item, Mapping)
        }
        return attributes, {}, "structured_json"

    if features_path is None or bpe_vocab is None:
        raise ValueError("Provide --attributes_json, or both --features_path and --bpe_vocab")
    try:
        import torch
    except ImportError as error:
        raise ImportError("PyTorch is required to recover captions from cached CLIP features") from error
    try:
        features = torch.load(features_path, map_location="cpu", weights_only=False)
    except TypeError:
        features = torch.load(features_path, map_location="cpu")
    if not isinstance(features, dict):
        raise TypeError(f"Expected a dict in cached CLIP features: {features_path}")

    decoder = ClipTokenDecoder(bpe_vocab)
    captions: Dict[str, str] = {}
    attributes: Dict[str, Dict[str, Any]] = {}
    for stem, item in features.items():
        if not isinstance(item, Mapping) or "token_ids" not in item:
            continue
        token_ids = item["token_ids"]
        attention_mask = item.get("attention_mask")
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if attention_mask is not None and hasattr(attention_mask, "tolist"):
            attention_mask = attention_mask.tolist()
        caption = decoder.decode(token_ids, attention_mask)
        stem_text = str(stem)
        captions[stem_text] = caption
        attributes[stem_text] = parse_fixed_caption(caption)
    return attributes, captions, "recovered_effective_caption"


def collect_split_names(data_root: Path, split_txt: str) -> List[str]:
    split_path = Path(split_txt)
    if not split_path.is_absolute():
        split_path = data_root / split_path
    with split_path.open("r", encoding="utf-8") as stream:
        names = [line.strip() for line in stream if line.strip() and not line.startswith("#")]
    return names


def find_by_stem(directory: Path, name: str, mask: bool = False) -> Optional[Path]:
    name_path = Path(name)
    direct = directory / name_path.name
    if direct.is_file():
        return direct
    stem = name_path.stem
    candidate_stems = [stem]
    if mask:
        candidate_stems += [f"{stem}_pixels0", f"{stem}_mask", f"{stem}_pixels"]
    for candidate_stem in candidate_stems:
        for extension in IMAGE_EXTENSIONS:
            candidate = directory / f"{candidate_stem}{extension}"
            if candidate.is_file():
                return candidate
    matches = sorted(
        path for path in directory.glob(f"{stem}*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return matches[0] if matches else None


@dataclass
class Component:
    area: int
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    centroid_x: float
    centroid_y: float

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1


def connected_components(mask: np.ndarray) -> List[Component]:
    foreground = mask.astype(bool)
    height, width = foreground.shape
    visited = np.zeros_like(foreground, dtype=bool)
    components: List[Component] = []
    for start_y, start_x in zip(*np.nonzero(foreground)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        xs: List[int] = []
        ys: List[int] = []
        while stack:
            y, x = stack.pop()
            xs.append(x)
            ys.append(y)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and foreground[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        components.append(
            Component(
                area=len(xs),
                min_x=min(xs),
                min_y=min(ys),
                max_x=max(xs),
                max_y=max(ys),
                centroid_x=float(sum(xs)) / len(xs),
                centroid_y=float(sum(ys)) / len(ys),
            )
        )
    return components


def position_cell(x: float, y: float, width: int, height: int) -> str:
    col = min(2, max(0, int(3.0 * x / max(1, width))))
    row = min(2, max(0, int(3.0 * y / max(1, height))))
    rows = ("upper", "center", "lower")
    cols = ("left", "center", "right")
    if row == 1 and col == 1:
        return "center"
    return f"{rows[row]}-{cols[col]}"


def is_near_grid_boundary(component: Component, width: int, height: int, margin_fraction: float) -> bool:
    x = component.centroid_x / max(1.0, float(width))
    y = component.centroid_y / max(1.0, float(height))
    return any(abs(value - boundary) <= margin_fraction for value in (x, y) for boundary in (1.0 / 3.0, 2.0 / 3.0))


def derive_shape(component: Component) -> str:
    if component.area <= 9:
        return "point-like"
    aspect = max(component.width, component.height) / max(1.0, min(component.width, component.height))
    if aspect >= 2.0:
        return "elongated"
    fill = component.area / max(1.0, float(component.width * component.height))
    return "round" if fill >= 0.6 else "irregular"


def contrast_from_image(image: Image.Image, mask: np.ndarray) -> Tuple[float, str]:
    if not mask.any():
        return float("nan"), "uncertain"
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if gray.shape != mask.shape:
        resized_mask = Image.fromarray((mask.astype(np.uint8) * 255)).resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(resized_mask) > 0
    mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    dilated = np.asarray(mask_image.filter(ImageFilter.MaxFilter(11))) > 0
    ring = dilated & ~mask
    if not ring.any():
        return float("nan"), "uncertain"
    target_mean = float(gray[mask].mean())
    background_mean = float(gray[ring].mean())
    background_std = float(gray[ring].std())
    scr = abs(target_mean - background_mean) / max(background_std, 1.0)
    if scr < 1.0:
        category = "very-low"
    elif scr < 2.0:
        category = "low"
    elif scr < 4.0:
        category = "moderate"
    elif scr < 8.0:
        category = "high"
    else:
        category = "very-high"
    return scr, category


def derive_gt_attributes(
    image: Image.Image,
    mask_image: Image.Image,
    tiny_max_area: int,
    boundary_margin: float,
) -> Tuple[Dict[str, Any], List[Component], np.ndarray]:
    mask = np.asarray(mask_image.convert("L")) > 0
    components = connected_components(mask)
    height, width = mask.shape
    if not components:
        return (
            {
                "target_present": False,
                "count": 0,
                "position": "none",
                "position_boundary_ambiguous": False,
                "size": "none",
                "shape": "none",
                "contrast": "uncertain",
                "local_scr": float("nan"),
                "component_areas": [],
            },
            components,
            mask,
        )

    positions = {position_cell(component.centroid_x, component.centroid_y, width, height) for component in components}
    position = next(iter(positions)) if len(positions) == 1 else "multiple-regions"
    boundary_ambiguous = any(
        is_near_grid_boundary(component, width, height, boundary_margin) for component in components
    )
    size_values = {"tiny" if component.area <= tiny_max_area else "small" for component in components}
    size = next(iter(size_values)) if len(size_values) == 1 else "mixed"
    shape_values = {derive_shape(component) for component in components}
    shape = next(iter(shape_values)) if len(shape_values) == 1 else "mixed"
    local_scr, contrast = contrast_from_image(image, mask)
    return (
        {
            "target_present": True,
            "count": len(components),
            "position": position,
            "position_boundary_ambiguous": boundary_ambiguous,
            "size": size,
            "shape": shape,
            "contrast": contrast,
            "local_scr": local_scr,
            "component_areas": [component.area for component in components],
        },
        components,
        mask,
    )


def compare_attributes(gpt: Mapping[str, Any], gt: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, float]]:
    status: Dict[str, str] = {}
    weights: Dict[str, float] = {}

    status["presence"] = "pass" if bool(gpt.get("target_present")) == bool(gt.get("target_present")) else "conflict"
    weights["presence"] = 1.0 if status["presence"] == "pass" else 0.0

    if status["presence"] == "conflict":
        for field in ("count", "location", "size", "shape", "contrast"):
            status[field] = "blocked_by_presence"
            weights[field] = 0.0
    elif not bool(gt.get("target_present")):
        for field in ("count", "location", "size", "shape"):
            status[field] = "not_applicable"
            weights[field] = 0.0
        status["contrast"] = "unverified"
        weights["contrast"] = 0.0
    else:
        status["count"] = "pass" if int(gpt.get("count", -1)) == int(gt.get("count", -2)) else "conflict"
        weights["count"] = 1.0 if status["count"] == "pass" else 0.0

        if gpt.get("position") == "uncertain" or bool(gt.get("position_boundary_ambiguous")):
            status["location"] = "uncertain"
        else:
            status["location"] = "pass" if gpt.get("position") == gt.get("position") else "conflict"
        weights["location"] = 1.0 if status["location"] == "pass" else 0.0

        if gpt.get("size") == "uncertain":
            status["size"] = "uncertain"
        else:
            status["size"] = "pass" if gpt.get("size") == gt.get("size") else "conflict"
        weights["size"] = 0.8 if status["size"] == "pass" else 0.0

        gpt_shape = str(gpt.get("shape", "uncertain"))
        if gpt_shape in ("uncertain", "aircraft-like", "ship-like", "vehicle-like"):
            status["shape"] = "unverified"
        else:
            status["shape"] = "heuristic_pass" if gpt_shape == gt.get("shape") else "heuristic_conflict"
        weights["shape"] = 0.3 if status["shape"] == "heuristic_pass" else 0.0

        gpt_contrast = str(gpt.get("contrast", "uncertain"))
        gt_contrast = str(gt.get("contrast", "uncertain"))
        if gpt_contrast == "uncertain" or gt_contrast == "uncertain":
            status["contrast"] = "unverified"
        else:
            distance = abs(CONTRAST_INDEX[gpt_contrast] - CONTRAST_INDEX[gt_contrast])
            status["contrast"] = "heuristic_pass" if distance <= 1 else "heuristic_conflict"
        weights["contrast"] = 0.3 if status["contrast"] == "heuristic_pass" else 0.0

    status["background"] = "unverified"
    weights["background"] = 0.0
    return status, weights


def classify_sample(status: Mapping[str, str]) -> Tuple[str, int, List[str]]:
    if status.get("presence") == "conflict":
        return "reject_auto", 1, ["presence_conflict"]
    if status.get("count") == "conflict":
        return "presence_only_auto", 2, ["count_masked"]
    # Sample-level acceptance is deliberately based on the minimum sufficient
    # objective condition: target presence and count.  Location/size are kept
    # only when they independently pass; subjective fields never trigger human
    # review in the first core-text experiment.
    return "presence_count_verified_auto", 5, []


def build_core_condition(
    gpt: Mapping[str, Any],
    gt: Mapping[str, Any],
    status: Mapping[str, str],
) -> Dict[str, Any]:
    presence_ok = status.get("presence") == "pass"
    target_present = bool(gpt.get("target_present")) if presence_ok else None
    negative_verified = presence_ok and not bool(gt.get("target_present"))
    count_ok = negative_verified or status.get("count") == "pass"
    location_ok = (
        presence_ok
        and bool(gt.get("target_present"))
        and status.get("location") == "pass"
        and not bool(gt.get("position_boundary_ambiguous"))
    )
    size_ok = (
        presence_ok
        and bool(gt.get("target_present"))
        and status.get("size") == "pass"
    )
    field_mask = {
        "presence": 1.0 if presence_ok else 0.0,
        "count": 1.0 if count_ok else 0.0,
        "location": 1.0 if location_ok else 0.0,
        "size": 1.0 if size_ok else 0.0,
        "shape": 0.0,
        "background": 0.0,
        "contrast": 0.0,
    }
    if not presence_ok:
        core_status = "reject_auto"
        usable = False
    elif count_ok:
        core_status = "presence_count_verified_auto"
        usable = True
    else:
        core_status = "presence_only_auto"
        usable = True

    attributes = {
        "target_present": target_present,
        "count": int(gpt.get("count", 0)) if count_ok else None,
        "location": str(gpt.get("position")) if location_ok else None,
        "size": str(gpt.get("size")) if size_ok else None,
        "shape": None,
        "background": None,
        "contrast": None,
    }
    if not usable:
        caption = ""
    elif target_present is False:
        caption = "no infrared small target is visible"
    else:
        tokens = ["infrared small target is visible"]
        if count_ok:
            tokens.append(f"count={attributes['count']}")
        if location_ok:
            tokens.append(f"location={attributes['location']}")
        if size_ok:
            tokens.append(f"size={attributes['size']}")
        caption = "; ".join(tokens)
    return {
        "policy": "presence_count_core_v1",
        "status": core_status,
        "usable": usable,
        "attributes": attributes,
        "field_mask": field_mask,
        "caption": caption,
        "excluded_fields": ["shape", "background", "contrast"],
        "multi_target": bool(gt.get("target_present")) and int(gt.get("count", 0)) > 1,
    }


def safe_float(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def json_safe_mapping(value: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            result[key] = json_safe_mapping(item)
        elif isinstance(item, list):
            result[key] = [safe_float(element) for element in item]
        else:
            result[key] = safe_float(item)
    return result


def fit_image(
    image: Image.Image,
    width: int,
    height: int,
    allow_upscale: bool = False,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    image = image.convert("RGB")
    if allow_upscale:
        scale = min(width / max(1, image.width), height / max(1, image.height))
        resized = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image = image.resize(resized, resample)
    else:
        image.thumbnail((width, height), resample)
    canvas = Image.new("RGB", (width, height), (12, 16, 24))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def mask_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = image.convert("RGB")
    if rgb.size != (mask.shape[1], mask.shape[0]):
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).resize(rgb.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image) > 0
    array = np.asarray(rgb).copy()
    red = np.zeros_like(array)
    red[..., 0] = 255
    array[mask] = (0.35 * array[mask] + 0.65 * red[mask]).astype(np.uint8)
    return Image.fromarray(array)


def target_crop(image: Image.Image, components: Sequence[Component], margin: int = 16) -> Image.Image:
    if not components:
        return ImageOps.autocontrast(image.convert("RGB"))
    min_x = max(0, min(component.min_x for component in components) - margin)
    min_y = max(0, min(component.min_y for component in components) - margin)
    max_x = min(image.width, max(component.max_x for component in components) + margin + 1)
    max_y = min(image.height, max(component.max_y for component in components) + margin + 1)
    crop = image.convert("RGB").crop((min_x, min_y, max_x, max_y))
    return ImageOps.autocontrast(crop)


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_panel(
    output_path: Path,
    dataset: str,
    stem: str,
    image: Image.Image,
    mask: np.ndarray,
    components: Sequence[Component],
    caption: str,
    gpt: Mapping[str, Any],
    gt: Mapping[str, Any],
    status: Mapping[str, str],
    queue_reasons: Sequence[str],
) -> None:
    width, height = 1600, 830
    canvas = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = find_font(26)
    body_font = find_font(20)
    small_font = find_font(17)
    draw.text((24, 18), f"{dataset} / {stem}", fill=(18, 32, 55), font=title_font)
    draw.text((24, 55), f"Queue: {'; '.join(queue_reasons) or 'pass sample'}", fill=(165, 45, 45), font=body_font)

    panel_width, panel_height = 370, 300
    x_positions = (20, 410, 800, 1190)
    panel_y = 120
    raw = image.convert("RGB")
    enhanced = ImageEnhance.Contrast(ImageOps.autocontrast(raw)).enhance(1.35)
    overlay = mask_overlay(raw, mask)
    crop = target_crop(raw, components)
    for x, label, item in zip(
        x_positions,
        ("Raw IR", "Contrast enhanced", "GT mask overlay", "Target crop"),
        (raw, enhanced, overlay, crop),
    ):
        draw.text((x, panel_y - 28), label, fill=(32, 48, 70), font=small_font)
        is_crop = label == "Target crop"
        fitted = fit_image(
            item,
            panel_width,
            panel_height,
            allow_upscale=is_crop,
            resample=Image.Resampling.NEAREST if is_crop else Image.Resampling.LANCZOS,
        )
        canvas.paste(fitted, (x, panel_y))

    text_y = 455
    caption_text = textwrap.fill(f"CLIP caption: {caption or '[not available]'}", width=145)
    draw.multiline_text((24, text_y), caption_text, fill=(28, 40, 58), font=small_font, spacing=4)
    text_y += 65
    gpt_text = (
        f"GPT effective: present={gpt.get('target_present')} count={gpt.get('count')} "
        f"position={gpt.get('position')} size={gpt.get('size')} shape={gpt.get('shape')} "
        f"background={gpt.get('background')} contrast={gpt.get('contrast')}"
    )
    gt_text = (
        f"GT derived: present={gt.get('target_present')} count={gt.get('count')} "
        f"position={gt.get('position')} size={gt.get('size')} shape={gt.get('shape')} "
        f"contrast={gt.get('contrast')} SCR={gt.get('local_scr')} areas={gt.get('component_areas')}"
    )
    status_text = "Auto status: " + ", ".join(f"{field}={status.get(field)}" for field in FIELD_NAMES)
    draw.multiline_text((24, text_y), textwrap.fill(gpt_text, width=145), fill=(28, 40, 58), font=small_font, spacing=4)
    draw.multiline_text((24, text_y + 55), textwrap.fill(gt_text, width=145), fill=(28, 40, 58), font=small_font, spacing=4)
    draw.multiline_text((24, text_y + 110), textwrap.fill(status_text, width=145), fill=(28, 40, 58), font=small_font, spacing=4)
    draw.text(
        (24, 785),
        "Manual rule: mark uncertain when pixels do not support a definite judgment; do not infer from GPT wording.",
        fill=(80, 90, 105),
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def flatten_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    gpt = record["gpt_effective"]
    gt = record["gt_derived"]
    status = record["field_status"]
    weights = record["field_weight"]
    core = record.get("core_condition", {})
    core_attributes = core.get("attributes", {})
    core_mask = core.get("field_mask", {})
    flat: Dict[str, Any] = {
        "dataset": record["dataset"],
        "stem": record["stem"],
        "split": record["split"],
        "sample_status": record["sample_status"],
        "queue_priority": record.get("queue_priority", ""),
        "queue_reasons": ";".join(record.get("queue_reasons", [])),
        "panel_path": record.get("panel_path", ""),
        "caption": record.get("caption", ""),
        "image_path": record.get("image_path", ""),
        "mask_path": record.get("mask_path", ""),
    }
    flat.update(
        {
            "gpt_presence": gpt.get("target_present"),
            "gpt_count": gpt.get("count"),
            "gpt_location": gpt.get("position"),
            "gpt_size": gpt.get("size"),
            "gpt_shape": gpt.get("shape"),
            "gpt_background": gpt.get("background"),
            "gpt_contrast": gpt.get("contrast"),
            "gpt_confidence": safe_float(gpt.get("confidence")),
            "gt_presence": gt.get("target_present"),
            "gt_count": gt.get("count"),
            "gt_location": gt.get("position"),
            "gt_location_boundary_ambiguous": gt.get("position_boundary_ambiguous"),
            "gt_size": gt.get("size"),
            "gt_shape_heuristic": gt.get("shape"),
            "gt_contrast_heuristic": gt.get("contrast"),
            "gt_local_scr": safe_float(gt.get("local_scr")),
            "gt_component_areas": json.dumps(gt.get("component_areas", []), ensure_ascii=False),
            "core_status": core.get("status", ""),
            "core_usable": core.get("usable", False),
            "core_caption": core.get("caption", ""),
            "core_multi_target": core.get("multi_target", False),
        }
    )
    for field in FIELD_NAMES:
        flat[f"status_{field}"] = status.get(field, "")
        flat[f"weight_{field}"] = weights.get(field, "")
        flat[f"core_{field}"] = core_attributes.get(
            "target_present" if field == "presence" else field,
            "",
        )
        flat[f"core_mask_{field}"] = core_mask.get(field, 0.0)
    return flat


def build_review_queue(records: List[Dict[str, Any]], fraction: float, seed: int) -> List[Dict[str, Any]]:
    # The core policy is fully determined by training GT and field masks.
    # Human review is optional data-quality research, not a prerequisite for C.
    return []


def write_review_readme(path: Path, dataset: str, source_mode: str) -> None:
    text = f"""# GPT 属性人工审核说明

数据集：`{dataset}`  
审计版本：`{SCRIPT_VERSION}`  
文本来源模式：`{source_mode}`

## 重要边界

- 本目录只审计训练划分，不使用测试 GT 生成可部署文本条件。
- 原始 GPT/CLIP 文件未被修改；`gpt_source_manifest.json` 保存其 SHA-256。
- 当来源模式为 `recovered_effective_caption` 时，审核对象是从 CLIP token IDs 无损恢复的固定文本，即实际进入 CLIP 的 teacher 文本。原始 GPT 自报 confidence 没有保存在 PT 中，因此不可恢复并留空。
- `presence_conflict` 已由训练 GT 确定为 `reject_auto`，写入 `automatic_rejects.csv`，不进入人工队列，也不使用 GT 把它改写成 GPT 条件 C。
- 首轮 `C_core` 只要求 presence/count；location/size 仅在自动一致时保留，否则字段屏蔽。
- `shape/background/contrast` 全部不进入首轮 `C_core`，多目标样本也不要求逐目标人工描述。

## 填写方式

核心条件由训练 GT 自动筛选，mandatory 人工审核数为 0。若后续进行可选数据质量抽查，可结合 `manual_review_panels` 查看图像，但不得把 GT 修正伪装成 GPT 条件 C。

- `correct`：GPT 字段与图像/GT 一致；
- `wrong`：可以明确判断字段错误；
- `uncertain`：像素证据不足，禁止猜测；
- `annotation_issue`：怀疑 GT mask 漏标或错标；
- 最终结论使用 `accept / partial / reject / annotation_issue`。

只评价语义，不评价英文流畅性。位置接近 3×3 分区边界、目标形状不足以辨认时应选择 `uncertain`。
"""
    path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GPT IR small-target attributes against training GT")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split_txt", default="50_50/train.txt")
    parser.add_argument("--img_dir", default="images")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--attributes_json", default=None)
    parser.add_argument("--features_path", default=None)
    parser.add_argument("--bpe_vocab", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tiny_max_area", type=int, default=25)
    parser.add_argument("--location_boundary_margin", type=float, default=0.04)
    parser.add_argument("--review_pass_fraction", type=float, default=0.10)
    parser.add_argument("--review_seed", type=int, default=20260819)
    parser.add_argument("--skip_panels", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = data_root / args.img_dir
    mask_dir = data_root / args.mask_dir
    dataset = args.dataset_name or data_root.name
    attributes_json = Path(args.attributes_json).expanduser().resolve() if args.attributes_json else None
    features_path = Path(args.features_path).expanduser().resolve() if args.features_path else None
    bpe_vocab = Path(args.bpe_vocab).expanduser().resolve() if args.bpe_vocab else None
    source_path = attributes_json or features_path
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(f"GPT source artifact not found: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    attributes, captions, source_mode = load_effective_attributes(attributes_json, features_path, bpe_vocab)
    names = collect_split_names(data_root, args.split_txt)
    records: List[Dict[str, Any]] = []
    missing: List[Dict[str, str]] = []

    for index, name in enumerate(names, start=1):
        stem = Path(name).stem
        image_path = find_by_stem(image_dir, name, mask=False)
        mask_path = find_by_stem(mask_dir, name, mask=True)
        gpt = attributes.get(stem)
        if image_path is None or mask_path is None or gpt is None:
            missing.append(
                {
                    "stem": stem,
                    "image": str(image_path or ""),
                    "mask": str(mask_path or ""),
                    "gpt": "present" if gpt is not None else "missing",
                }
            )
            continue
        with Image.open(image_path) as image_stream, Image.open(mask_path) as mask_stream:
            image = image_stream.convert("RGB")
            mask_image = mask_stream.convert("L")
        gt, components, mask = derive_gt_attributes(
            image,
            mask_image,
            tiny_max_area=max(1, int(args.tiny_max_area)),
            boundary_margin=max(0.0, float(args.location_boundary_margin)),
        )
        status, weights = compare_attributes(gpt, gt)
        sample_status, priority, reasons = classify_sample(status)
        core_condition = build_core_condition(gpt, gt, status)
        records.append(
            {
                "dataset": dataset,
                "stem": stem,
                "split": args.split_txt,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "caption": captions.get(stem, ""),
                "gpt_effective": dict(gpt),
                "gt_derived": gt,
                "field_status": status,
                "field_weight": weights,
                "core_condition": core_condition,
                "sample_status": sample_status,
                "queue_priority": priority,
                "queue_reasons": reasons,
                "_image": image,
                "_mask": mask,
                "_components": components,
            }
        )
        if index % 100 == 0 or index == len(names):
            print(f"[{dataset}] audited {index}/{len(names)}")

    auto_rejects = [record for record in records if record["sample_status"] == "reject_auto"]
    queue = build_review_queue(records, args.review_pass_fraction, args.review_seed)
    panel_dir = output_dir / "manual_review_panels"
    if not args.skip_panels:
        for index, record in enumerate(queue, start=1):
            panel_path = panel_dir / f"{dataset}__{record['stem']}.png"
            draw_panel(
                panel_path,
                dataset=dataset,
                stem=record["stem"],
                image=record["_image"],
                mask=record["_mask"],
                components=record["_components"],
                caption=record["caption"],
                gpt=record["gpt_effective"],
                gt=record["gt_derived"],
                status=record["field_status"],
                queue_reasons=record["queue_reasons"],
            )
            record["panel_path"] = str(panel_path.relative_to(output_dir))
            if index % 100 == 0 or index == len(queue):
                print(f"[{dataset}] panels {index}/{len(queue)}")

    public_records: Dict[str, Dict[str, Any]] = {}
    for record in records:
        public_records[record["stem"]] = json_safe_mapping(
            {key: value for key, value in record.items() if not key.startswith("_")}
        )
    atomic_write_json(output_dir / "gpt_attribute_audit.json", public_records)
    if captions:
        atomic_write_json(output_dir / "recovered_fixed_descriptions.json", captions)
    atomic_write_json(
        output_dir / "reconstructed_effective_attributes.json",
        {stem: json_safe_mapping(value) for stem, value in attributes.items()},
    )
    atomic_write_json(
        output_dir / "verified_core_attributes.json",
        {
            record["stem"]: json_safe_mapping(record["core_condition"])
            for record in records
        },
    )

    source_manifest = {
        "audit_version": SCRIPT_VERSION,
        "created_at": utc_now(),
        "dataset": dataset,
        "split": args.split_txt,
        "source_mode": source_mode,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_mtime_utc": datetime.fromtimestamp(source_path.stat().st_mtime, timezone.utc).isoformat(),
        "source_sha256": sha256_file(source_path),
        "selected_split_records": len(names),
        "audited_records": len(records),
        "missing_records": missing,
        "gpt_confidence_recoverable": source_mode == "structured_json",
        "tiny_max_area_original_pixels": int(args.tiny_max_area),
        "location_boundary_margin_fraction": float(args.location_boundary_margin),
        "review_pass_fraction": float(args.review_pass_fraction),
        "review_seed": int(args.review_seed),
        "core_condition_policy": "presence+count define sample usability; matching location/size retained; shape/background/contrast masked",
        "presence_conflict_policy": "reject_auto; excluded from C_core",
        "manual_review_policy": "no mandatory manual review; optional annotation QC only",
        "contrast_heuristic": "SCR=abs(target_mean-ring_mean)/max(ring_std,1); bins [1,2,4,8]",
        "shape_heuristic": "area<=9 point-like; aspect>=2 elongated; fill>=0.6 round; else irregular",
    }
    atomic_write_json(output_dir / "gpt_source_manifest.json", source_manifest)

    sample_counter = Counter(record["sample_status"] for record in records)
    summary = {
        "dataset": dataset,
        "source_mode": source_mode,
        "selected": len(names),
        "audited": len(records),
        "missing": len(missing),
        "automatic_rejects": len(auto_rejects),
        "review_queue": len(queue),
        "mandatory_manual_review": 0,
        **{f"sample_{key}": value for key, value in sorted(sample_counter.items())},
    }
    summary_rows = [summary]
    write_csv(output_dir / "gpt_attribute_audit_summary.csv", list(summary.keys()), summary_rows)

    field_rows: List[Dict[str, Any]] = []
    for field in FIELD_NAMES:
        counter = Counter(record["field_status"].get(field, "missing") for record in records)
        total = sum(counter.values())
        for status_name, count in sorted(counter.items()):
            field_rows.append(
                {
                    "dataset": dataset,
                    "field": field,
                    "status": status_name,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                }
            )
    write_csv(
        output_dir / "gpt_attribute_field_summary.csv",
        ("dataset", "field", "status", "count", "fraction"),
        field_rows,
    )

    audit_rows = [flatten_record(record) for record in records]
    audit_fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    if audit_rows:
        write_csv(output_dir / "gpt_attribute_audit_flat.csv", audit_fieldnames, audit_rows)
        write_csv(output_dir / "automatic_core_screening.csv", audit_fieldnames, audit_rows)

    automatic_reject_rows = [flatten_record(record) for record in auto_rejects]
    if automatic_reject_rows:
        write_csv(
            output_dir / "automatic_rejects.csv",
            list(automatic_reject_rows[0].keys()),
            automatic_reject_rows,
        )

    review_rows: List[Dict[str, Any]] = []
    for record in queue:
        flat = flatten_record(record)
        for field in FIELD_NAMES:
            flat[f"manual_{field}"] = ""
            flat[f"corrected_{field}"] = ""
        flat["manual_final_decision"] = ""
        flat["manual_notes"] = ""
        review_rows.append(flat)
    review_fieldnames = list(audit_fieldnames)
    for field in FIELD_NAMES:
        review_fieldnames.extend((f"manual_{field}", f"corrected_{field}"))
    review_fieldnames.extend(("manual_final_decision", "manual_notes"))
    # Always rewrite the file, even when empty, so rerunning into a previous
    # output directory cannot leave a stale mandatory-review queue behind.
    write_csv(output_dir / "manual_review_queue.csv", review_fieldnames, review_rows)

    write_review_readme(output_dir / "README_MANUAL_REVIEW_CN.md", dataset, source_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

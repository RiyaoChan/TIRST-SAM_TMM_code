#!/usr/bin/env python
"""Generate VLM meta-annotations from IRCoT outputs and GT masks.

The output is used by Scheme 1 v2 design: VLM knowledge enters training through
offline scene labels, difficulty scores, and hard negative points.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


COMMON_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SCENE_TO_ID = {
    "unknown": 0,
    "sky": 1,
    "sea": 2,
    "urban": 3,
    "forest": 4,
    "mixed": 5,
    "other": 6,
}


def read_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_json(path: str, data: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def stem_of(name: str) -> str:
    return Path(name).stem


def lookup_by_stem(data: Dict[str, Any], stem: str) -> Any:
    if stem in data:
        return data[stem]
    for key, value in data.items():
        if stem_of(str(key)) == stem:
            return value
    return None


def load_split(data_root: Path, split_txt: str) -> List[str]:
    split_path = Path(split_txt)
    if not split_path.is_absolute():
        split_path = data_root / split_txt
    with split_path.open("r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip() and not line.startswith("#")]


def apply_suffix(name: str, suffix: str) -> str:
    if not suffix:
        return name
    p = Path(name)
    if p.suffix:
        return str(p.with_name(p.stem + suffix + p.suffix))
    return name + suffix


def resolve_mask(data_root: Path, mask_dir: str, name: str, mask_suffix: str) -> Optional[Path]:
    root = data_root / mask_dir
    p = Path(name)
    candidates: List[Path] = []
    if p.suffix:
        candidates.append(root / apply_suffix(p.name, mask_suffix))
        candidates.append(root / p.name)
    else:
        if mask_suffix:
            for ext in COMMON_EXTS:
                candidates.append(root / f"{p.name}{mask_suffix}{ext}")
        for ext in COMMON_EXTS:
            candidates.append(root / f"{p.name}{ext}")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def mask_objects(mask_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    lab = label(mask, connectivity=2)
    objects: List[Dict[str, Any]] = []
    for prop in regionprops(lab):
        y1, x1, y2, x2 = prop.bbox
        cy, cx = prop.centroid
        objects.append(
            {
                "point": [float(cx), float(cy)],
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "area": int(prop.area),
            }
        )
    return mask, objects


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def candidate_point(candidate: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    point = candidate.get("point")
    if isinstance(point, Sequence) and len(point) >= 2:
        return safe_float(point[0], math.nan), safe_float(point[1], math.nan)
    return None


def candidate_hits(candidate: Dict[str, Any], gt_objects: Sequence[Dict[str, Any]], radius: float) -> bool:
    pt = candidate_point(candidate)
    if pt is None:
        return False
    x, y = pt
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    for obj in gt_objects:
        gx, gy = obj["point"]
        if math.hypot(x - float(gx), y - float(gy)) <= radius:
            return True
    return False


def count_hits(candidates: Sequence[Dict[str, Any]], gt_objects: Sequence[Dict[str, Any]], radius: float) -> int:
    hits = 0
    for obj in gt_objects:
        gx, gy = obj["point"]
        for cand in candidates:
            pt = candidate_point(cand)
            if pt is None:
                continue
            if math.hypot(pt[0] - gx, pt[1] - gy) <= radius:
                hits += 1
                break
    return hits


def extract_scene(prompt_item: Any, debug_item: Any) -> Dict[str, Any]:
    scene_info: Dict[str, Any] = {}
    if isinstance(debug_item, dict) and isinstance(debug_item.get("step1_scene"), dict):
        scene_info = dict(debug_item["step1_scene"])
    elif isinstance(prompt_item, dict):
        chain = prompt_item.get("ircot_chain")
        if isinstance(chain, dict) and isinstance(chain.get("step1_scene"), dict):
            scene_info = dict(chain["step1_scene"])
    scene = str(scene_info.get("scene_type", "unknown")).lower()
    if scene not in SCENE_TO_ID:
        scene = "other"
    scene_info["scene_type"] = scene
    scene_info["scene_id"] = SCENE_TO_ID[scene]
    return scene_info


def difficulty_level(score: float) -> str:
    if score < 0.3:
        return "easy"
    if score < 0.6:
        return "medium"
    return "hard"


def compute_difficulty(hit_count: int, gt_count: int, false_positive_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    hit_rate = max(0.0, min(1.0, float(hit_count) / float(gt_count)))
    clutter_score = min(float(false_positive_count) / 20.0, 1.0)
    return max(0.0, min(1.0, (1.0 - hit_rate) * 0.6 + clutter_score * 0.4))


def normalize_candidate(
    candidate: Dict[str, Any],
    source: str,
    gt_objects: Sequence[Dict[str, Any]],
    radius: float,
) -> Optional[Dict[str, Any]]:
    pt = candidate_point(candidate)
    if pt is None:
        return None
    x, y = pt
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    conf = safe_float(candidate.get("confidence", candidate.get("combined_score", candidate.get("saliency_score", 0.0))))
    return {
        "x": float(x),
        "y": float(y),
        "confidence": max(0.0, min(1.0, conf)),
        "source": source,
        "hit_gt": bool(candidate_hits(candidate, gt_objects, radius)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VLM meta annotations for Scheme 1 v2.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split_txt", required=True)
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--mask_suffix", default="")
    parser.add_argument("--ircot_json", required=True)
    parser.add_argument("--ircot_debug_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--match_radius", type=float, default=12.0)
    parser.add_argument("--max_hard_negatives", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    names = load_split(data_root, args.split_txt)
    prompt_data = read_json(args.ircot_json)
    debug_data = read_json(args.ircot_debug_json)

    meta: Dict[str, Any] = {}
    total_gt = 0
    total_hit = 0
    total_hard_neg = 0
    scene_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}

    for name in names:
        stem = stem_of(name)
        mask_path = resolve_mask(data_root, args.mask_dir, name, args.mask_suffix)
        if mask_path is None:
            continue
        _, gt_objects = mask_objects(mask_path)
        prompt_item = lookup_by_stem(prompt_data, stem)
        debug_item = lookup_by_stem(debug_data, stem)
        final_candidates = prompt_item.get("targets", []) if isinstance(prompt_item, dict) else []
        if not isinstance(final_candidates, list):
            final_candidates = []
        step2_candidates = debug_item.get("step2_candidates", []) if isinstance(debug_item, dict) else []
        if not isinstance(step2_candidates, list):
            step2_candidates = []

        scene_info = extract_scene(prompt_item, debug_item)
        scene = scene_info["scene_type"]
        scene_counts[scene] = scene_counts.get(scene, 0) + 1

        final_hit = count_hits(final_candidates, gt_objects, float(args.match_radius))
        gt_count = len(gt_objects)
        total_gt += gt_count
        total_hit += final_hit

        all_candidates: List[Dict[str, Any]] = []
        for cand in step2_candidates:
            if isinstance(cand, dict):
                norm = normalize_candidate(cand, "step2", gt_objects, float(args.match_radius))
                if norm is not None:
                    all_candidates.append(norm)
        for cand in final_candidates:
            if isinstance(cand, dict):
                norm = normalize_candidate(cand, "final", gt_objects, float(args.match_radius))
                if norm is not None:
                    all_candidates.append(norm)

        hard_negs = [c for c in all_candidates if not c["hit_gt"]]
        hard_negs.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
        hard_negs = hard_negs[: max(0, int(args.max_hard_negatives))]
        true_pos = [c for c in all_candidates if c["hit_gt"]]
        true_pos.sort(key=lambda c: float(c.get("confidence", 0.0)), reverse=True)

        diff = compute_difficulty(final_hit, gt_count, len(hard_negs))
        level = difficulty_level(diff)
        difficulty_counts[level] = difficulty_counts.get(level, 0) + 1
        total_hard_neg += len(hard_negs)

        meta[stem] = {
            "scene_type": scene,
            "scene_id": scene_info["scene_id"],
            "scene_description": scene_info.get("description", ""),
            "background_complexity": scene_info.get("background_complexity", "unknown"),
            "noise_level": scene_info.get("noise_level", "unknown"),
            "difficulty": level,
            "difficulty_score": diff,
            "gt_target_count": gt_count,
            "ircot_hit_count": int(final_hit),
            "ircot_miss_count": int(max(0, gt_count - final_hit)),
            "num_step2_candidates_saved": len(step2_candidates),
            "num_final_candidates": len(final_candidates),
            "hard_negative_points": hard_negs,
            "ircot_true_positive_candidates": true_pos[: max(0, int(args.max_hard_negatives))],
            "mask": str(mask_path),
        }

    out = {
        "meta": meta,
        "summary": {
            "num_images": len(meta),
            "split_txt": args.split_txt,
            "match_radius": args.match_radius,
            "total_gt": total_gt,
            "total_ircot_hits": total_hit,
            "final_recall": total_hit / total_gt if total_gt else 0.0,
            "total_hard_negative_points": total_hard_neg,
            "scene_counts": scene_counts,
            "difficulty_counts": difficulty_counts,
        },
    }
    save_json(args.output, out)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()

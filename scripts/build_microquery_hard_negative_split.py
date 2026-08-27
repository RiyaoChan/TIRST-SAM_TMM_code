#!/usr/bin/env python3
"""Build a train-derived, GT-excluding hard-negative crop manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, laplace


def dilate_gt(mask: np.ndarray, pixels: int = 12) -> np.ndarray:
    """Dilate a binary GT mask with a square radius, for crop exclusion only."""

    mask = np.asarray(mask, dtype=bool)
    width = 2 * int(pixels) + 1
    return binary_dilation(mask, structure=np.ones((width, width), dtype=bool))


def crop_is_background(
    dilated_mask: np.ndarray, left: int, top: int, crop_size: int = 256
) -> bool:
    region = dilated_mask[top : top + crop_size, left : left + crop_size]
    return region.shape == (crop_size, crop_size) and not bool(region.any())


def enumerate_background_crops(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int = 256,
    dilation_pixels: int = 12,
    stride: int = 32,
) -> list[dict]:
    """Enumerate eligible crops and rank textured/bright hard backgrounds first."""

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if image.ndim == 3:
        image = image.mean(axis=2)
    height, width = image.shape
    if height < crop_size or width < crop_size:
        return []
    dilated = dilate_gt(mask, dilation_pixels)
    x_values = list(range(0, width - crop_size + 1, max(1, int(stride))))
    y_values = list(range(0, height - crop_size + 1, max(1, int(stride))))
    if x_values[-1] != width - crop_size:
        x_values.append(width - crop_size)
    if y_values[-1] != height - crop_size:
        y_values.append(height - crop_size)
    rows = []
    for top in y_values:
        for left in x_values:
            if not crop_is_background(dilated, left, top, crop_size):
                continue
            crop = image[top : top + crop_size, left : left + crop_size]
            rows.append(
                {
                    "left": int(left),
                    "top": int(top),
                    "crop_size": int(crop_size),
                    "hardness": float(
                        np.std(crop)
                        + np.percentile(crop, 99.5) - np.mean(crop)
                        + np.mean(np.abs(laplace(crop)))
                    ),
                }
            )
    return sorted(rows, key=lambda row: (-row["hardness"], row["top"], row["left"]))


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--image_dir", default="images")
    parser.add_argument("--mask_dir", default="masks")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--dilation_pixels", type=int, default=12)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max_per_image", type=int, default=2)
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def find_file(directory: Path, name: str) -> Path:
    stem = Path(name).stem
    candidates = [directory / name]
    candidates.extend(directory / f"{stem}{suffix}" for suffix in (".png", ".bmp", ".jpg", ".tif"))
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for {name} under {directory}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    split_path = Path(args.train_split).resolve()
    output_dir = Path(args.output_dir).resolve()
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    names = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pending = []
    for name in names:
        image_path = find_file(data_root / args.image_dir, name)
        mask_path = find_file(data_root / args.mask_dir, name)
        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path)) > 0
        candidates = enumerate_background_crops(
            image,
            mask,
            args.crop_size,
            args.dilation_pixels,
            args.stride,
        )[: args.max_per_image]
        for local_index, candidate in enumerate(candidates):
            left, top = candidate["left"], candidate["top"]
            crop_name = f"{Path(name).stem}_{top:04d}_{left:04d}_{local_index}.png"
            pending.append(
                {
                    "crop": crop_name,
                    "source": name,
                    "_image_path": str(image_path),
                    **candidate,
                }
            )
    selected = sorted(
        pending,
        key=lambda row: (-row["hardness"], row["source"], row["top"], row["left"]),
    )[: args.limit]
    rows = []
    for candidate in selected:
        image = np.asarray(Image.open(candidate.pop("_image_path")))
        left, top = candidate["left"], candidate["top"]
        crop = image[top : top + args.crop_size, left : left + args.crop_size]
        Image.fromarray(crop).save(crop_dir / candidate["crop"])
        rows.append({**candidate, "crop_sha256": sha256_array(crop)})
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["crop", "source", "left", "top", "crop_size", "hardness", "crop_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_split": str(split_path),
        "source_scope": "train-only",
        "crop_count": len(rows),
        "crop_size": args.crop_size,
        "dilation_pixels": args.dilation_pixels,
        "status": "READY" if rows else "EMPTY_NO_ELIGIBLE_256_CROPS",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

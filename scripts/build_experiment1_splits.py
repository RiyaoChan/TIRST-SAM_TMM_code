#!/usr/bin/env python3
"""Build leakage-safe train/val/test splits for Experiment 1.

The test split is copied byte-for-byte at the sample-id level. Validation is
drawn only from the original training split with deterministic, component-aware
stratification. No existing split file is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from skimage.measure import label


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate sample ids in {path}")
    return names


def write_split(path: Path, names: Iterable[str]) -> None:
    values = list(names)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in values), encoding="utf-8")


def find_mask(mask_dir: Path, sample_id: str, mask_suffix: str = "") -> Path:
    normalized = sample_id.replace("\\", "/")
    if normalized.startswith("images/"):
        normalized = normalized[len("images/") :]
    rel = Path(normalized)
    stem = rel.stem if rel.suffix else rel.name
    parent = rel.parent if str(rel.parent) != "." else Path()
    candidates = []
    for suffix in (mask_suffix, ""):
        for extension in ((rel.suffix,) if rel.suffix else IMAGE_EXTENSIONS):
            candidates.append(mask_dir / parent / f"{stem}{suffix}{extension}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask found for sample {sample_id!r} under {mask_dir}")


def area_bucket(area: int) -> str:
    if area <= 0:
        return "0"
    if area <= 9:
        return "1-9"
    if area <= 16:
        return "10-16"
    if area <= 25:
        return "17-25"
    return ">25"


def count_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def mask_statistics(mask_path: Path) -> dict:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    components = label(mask.astype(np.uint8), connectivity=2)
    counts = np.bincount(components.reshape(-1))
    areas = counts[1:].astype(np.int64).tolist() if counts.size > 1 else []
    total_area = int(mask.sum())
    max_area = int(max(areas, default=0))
    component_count = int(len(areas))
    return {
        "has_target": bool(component_count > 0),
        "component_count": component_count,
        "max_area": max_area,
        "total_area": total_area,
        "component_count_bucket": count_bucket(component_count),
        "max_area_bucket": area_bucket(max_area),
        "total_area_bucket": area_bucket(total_area),
    }


def stratum_key(stats: dict) -> str:
    return "|".join(
        (
            "target" if stats["has_target"] else "empty",
            f"count={stats['component_count_bucket']}",
            f"max={stats['max_area_bucket']}",
            f"total={stats['total_area_bucket']}",
        )
    )


def stratified_split(
    names: list[str],
    stats_by_name: dict[str, dict],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str], dict]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie in (0, 1)")
    n_val = max(1, min(len(names) - 1, int(round(len(names) * val_fraction))))
    groups: dict[str, list[str]] = defaultdict(list)
    for name in names:
        groups[stratum_key(stats_by_name[name])].append(name)

    rng = random.Random(int(seed))
    for key in sorted(groups):
        rng.shuffle(groups[key])

    quotas = {}
    remainders = []
    capacity = 0
    for key, members in groups.items():
        raw = len(members) * float(val_fraction)
        cap = max(0, len(members) - 1)
        base = min(cap, int(math.floor(raw)))
        quotas[key] = base
        capacity += cap
        remainders.append((raw - math.floor(raw), len(members), key))

    method = "component_stratified_largest_remainder"
    if capacity < n_val:
        shuffled = list(names)
        rng.shuffle(shuffled)
        val_set = set(shuffled[:n_val])
        method = "fixed_seed_random_fallback"
    else:
        remaining = n_val - sum(quotas.values())
        for _, _, key in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2])):
            if remaining <= 0:
                break
            if quotas[key] < len(groups[key]) - 1:
                quotas[key] += 1
                remaining -= 1
        if remaining > 0:
            for key in sorted(groups, key=lambda value: (-len(groups[value]), value)):
                while remaining > 0 and quotas[key] < len(groups[key]) - 1:
                    quotas[key] += 1
                    remaining -= 1
        if remaining != 0:
            raise RuntimeError("Unable to allocate the requested validation size")
        val_set = {
            name
            for key, members in groups.items()
            for name in members[: quotas[key]]
        }

    train_names = [name for name in names if name not in val_set]
    val_names = [name for name in names if name in val_set]
    if len(val_names) != n_val or set(train_names) & set(val_names):
        raise RuntimeError("Invalid train/validation partition")

    audit = {
        "method": method,
        "requested_val_fraction": float(val_fraction),
        "target_val_count": n_val,
        "strata": {
            key: {
                "source": len(members),
                "train": sum(name in set(train_names) for name in members),
                "val": sum(name in val_set for name in members),
            }
            for key, members in sorted(groups.items())
        },
    }
    return train_names, val_names, audit


def aggregate_stats(names: Iterable[str], stats_by_name: dict[str, dict]) -> dict:
    selected = [stats_by_name[name] for name in names]
    component_hist = Counter(item["component_count_bucket"] for item in selected)
    max_area_hist = Counter(item["max_area_bucket"] for item in selected)
    total_area_hist = Counter(item["total_area_bucket"] for item in selected)
    return {
        "samples": len(selected),
        "target_present": sum(bool(item["has_target"]) for item in selected),
        "target_absent": sum(not bool(item["has_target"]) for item in selected),
        "components": sum(int(item["component_count"]) for item in selected),
        "component_count_buckets": dict(sorted(component_hist.items())),
        "max_area_buckets": dict(sorted(max_area_hist.items())),
        "total_area_buckets": dict(sorted(total_area_hist.items())),
    }


def build_splits(
    data_root: Path,
    source_train: str = "50_50/train.txt",
    source_test: str = "50_50/test.txt",
    output_dir: str = "splits/experiment1_seed20260825",
    seed: int = 20260825,
    val_fraction: float = 0.1,
    mask_suffix: str = "",
) -> dict:
    data_root = data_root.resolve()
    train_path = data_root / source_train
    test_path = data_root / source_test
    mask_dir = data_root / "masks"
    output_path = data_root / output_dir
    if output_path.resolve() in {train_path.parent.resolve(), test_path.parent.resolve()}:
        raise ValueError("Refusing to overwrite the source split directory")

    source_train_names = read_split(train_path)
    test_names = read_split(test_path)
    overlap = set(source_train_names) & set(test_names)
    if overlap:
        raise ValueError(f"Source train/test overlap detected: {sorted(overlap)[:5]}")

    stats_by_name = {
        name: mask_statistics(find_mask(mask_dir, name, mask_suffix=mask_suffix))
        for name in source_train_names
    }
    train_names, val_names, split_audit = stratified_split(
        source_train_names,
        stats_by_name,
        val_fraction=val_fraction,
        seed=seed,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    write_split(output_path / "train.txt", train_names)
    write_split(output_path / "val.txt", val_names)
    write_split(output_path / "test.txt", test_names)

    manifest = {
        "schema_version": 1,
        "experiment": "TIRST-SAM Experiment 1",
        "dataset": data_root.name,
        "seed": int(seed),
        "source": {
            "train_split": source_train.replace("\\", "/"),
            "test_split": source_test.replace("\\", "/"),
            "train_sha256": sha256_file(train_path),
            "test_sha256": sha256_file(test_path),
            "train_count": len(source_train_names),
            "test_count": len(test_names),
        },
        "output": {
            "directory": output_dir.replace("\\", "/"),
            "train_count": len(train_names),
            "val_count": len(val_names),
            "test_count": len(test_names),
            "test_ids_exact_copy": test_names == read_split(output_path / "test.txt"),
        },
        "split_audit": split_audit,
        "statistics": {
            "source_train": aggregate_stats(source_train_names, stats_by_name),
            "train": aggregate_stats(train_names, stats_by_name),
            "val": aggregate_stats(val_names, stats_by_name),
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--source_train", default="50_50/train.txt")
    parser.add_argument("--source_test", default="50_50/test.txt")
    parser.add_argument("--output_dir", default="splits/experiment1_seed20260825")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--mask_suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_splits(
        Path(args.data_root),
        source_train=args.source_train,
        source_test=args.source_test,
        output_dir=args.output_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        mask_suffix=args.mask_suffix,
    )
    print(json.dumps(manifest["output"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.build_experiment1_splits import build_splits, read_split


def _write_mask(path: Path, components: list[tuple[int, int, int]]) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    for y, x, size in components:
        mask[y : y + size, x : x + size] = 255
    Image.fromarray(mask).save(path)


def _make_dataset(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "50_50").mkdir(parents=True)
    train = []
    for index in range(20):
        name = f"train_{index:02d}"
        train.append(name)
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(root / "images" / f"{name}.png")
        components = [] if index % 7 == 0 else [(4, 4, 1 + index % 4)]
        if index % 5 == 0:
            components.append((20, 20, 2))
        _write_mask(root / "masks" / f"{name}.png", components)
    test = ["test_00", "test_01"]
    for name in test:
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(root / "images" / f"{name}.png")
        _write_mask(root / "masks" / f"{name}.png", [(10, 10, 2)])
    (root / "50_50" / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (root / "50_50" / "test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")


def test_build_splits_is_deterministic_and_keeps_test_untouched(tmp_path):
    _make_dataset(tmp_path)
    first = build_splits(tmp_path, seed=123, val_fraction=0.2, output_dir="splits/a")
    second = build_splits(tmp_path, seed=123, val_fraction=0.2, output_dir="splits/b")

    train_a = read_split(tmp_path / "splits/a/train.txt")
    val_a = read_split(tmp_path / "splits/a/val.txt")
    assert train_a == read_split(tmp_path / "splits/b/train.txt")
    assert val_a == read_split(tmp_path / "splits/b/val.txt")
    assert len(train_a) == 16
    assert len(val_a) == 4
    assert not (set(train_a) & set(val_a))
    assert read_split(tmp_path / "splits/a/test.txt") == ["test_00", "test_01"]
    assert first["output"]["test_ids_exact_copy"] is True
    assert first["source"]["train_sha256"] == second["source"]["train_sha256"]


def test_build_splits_never_places_source_test_in_train_or_val(tmp_path):
    _make_dataset(tmp_path)
    build_splits(tmp_path, seed=20260825, val_fraction=0.1)
    train = set(read_split(tmp_path / "splits/experiment1_seed20260825/train.txt"))
    val = set(read_split(tmp_path / "splits/experiment1_seed20260825/val.txt"))
    test = set(read_split(tmp_path / "splits/experiment1_seed20260825/test.txt"))
    assert not train & test
    assert not val & test

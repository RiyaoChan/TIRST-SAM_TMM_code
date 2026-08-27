import numpy as np
import pytest

from scripts.microquery_end2end_dataset import load_candidate_cache


def test_cache_requires_exact_schema_and_image_order(tmp_path):
    path = tmp_path / "candidates.npz"
    np.savez(
        path,
        image_names=np.asarray(["a", "b"]),
        candidate_xy=np.zeros((2, 10, 2), np.float32),
        candidate_scores=np.zeros((2, 10), np.float32),
        candidate_valid=np.ones((2, 10), bool),
    )
    loaded = load_candidate_cache(path, ["a", "b"], 10)
    assert loaded["candidate_xy"].shape == (2, 10, 2)
    with pytest.raises(RuntimeError):
        load_candidate_cache(path, ["b", "a"], 10)


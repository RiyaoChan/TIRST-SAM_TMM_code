import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_gpt_structured_attributes.py"
SPEC = importlib.util.spec_from_file_location("audit_gpt_structured_attributes", SCRIPT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def test_parse_positive_fixed_caption_with_split_digit_count():
    caption = (
        "1 4 tiny point - like infrared targets are visible across multiple image regions "
        "against an urban background with moderate target - to - background contrast ."
    )
    parsed = AUDIT.parse_fixed_caption(caption)
    assert parsed["target_present"] is True
    assert parsed["count"] == 14
    assert parsed["position"] == "multiple-regions"
    assert parsed["size"] == "tiny"
    assert parsed["shape"] == "point-like"
    assert parsed["background"] == "urban"
    assert parsed["contrast"] == "moderate"


def test_parse_negative_fixed_caption():
    parsed = AUDIT.parse_fixed_caption(
        "No infrared small target is visible against a cloud-cluttered background with low scene contrast."
    )
    assert parsed["target_present"] is False
    assert parsed["count"] == 0
    assert parsed["position"] == "none"
    assert parsed["background"] == "cloud-cluttered"
    assert parsed["contrast"] == "low"


def test_connected_components_use_eight_connectivity():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    mask[6, 6] = True
    components = AUDIT.connected_components(mask)
    assert sorted(component.area for component in components) == [1, 2]


def test_presence_conflict_blocks_target_fields():
    gpt = {
        "target_present": False,
        "count": 0,
        "position": "none",
        "size": "none",
        "shape": "none",
        "background": "urban",
        "contrast": "low",
    }
    gt = {
        "target_present": True,
        "count": 1,
        "position": "center",
        "position_boundary_ambiguous": False,
        "size": "tiny",
        "shape": "point-like",
        "contrast": "high",
    }
    status, weights = AUDIT.compare_attributes(gpt, gt)
    assert status["presence"] == "conflict"
    assert status["count"] == "blocked_by_presence"
    assert weights["presence"] == 0.0
    assert weights["location"] == 0.0


def test_presence_conflict_is_auto_rejected_and_excluded_from_human_queue():
    rejected = {
        "stem": "negative_caption",
        "sample_status": "reject_auto",
        "queue_priority": 1,
        "queue_reasons": ["presence_conflict"],
    }
    ambiguous = {
        "stem": "location_boundary",
        "sample_status": "partial_auto",
        "queue_priority": 3,
        "queue_reasons": ["location_uncertain"],
    }
    queue = AUDIT.build_review_queue([rejected, ambiguous], fraction=0.0, seed=20260819)
    assert queue == []


def test_matching_presence_and_count_define_verified_core_without_manual_review():
    status = {
        "presence": "pass",
        "count": "pass",
        "location": "conflict",
        "size": "conflict",
        "shape": "heuristic_conflict",
        "background": "unverified",
        "contrast": "heuristic_conflict",
    }
    sample_status, _, reasons = AUDIT.classify_sample(status)
    assert sample_status == "presence_count_verified_auto"
    assert reasons == []


def test_multi_target_core_masks_subjective_fields_and_keeps_matching_count():
    gpt = {
        "target_present": True,
        "count": 2,
        "position": "multiple-regions",
        "size": "mixed",
        "shape": "mixed",
        "background": "urban",
        "contrast": "high",
    }
    gt = {
        "target_present": True,
        "count": 2,
        "position": "multiple-regions",
        "position_boundary_ambiguous": False,
        "size": "mixed",
    }
    status = {
        "presence": "pass",
        "count": "pass",
        "location": "pass",
        "size": "pass",
        "shape": "heuristic_conflict",
        "background": "unverified",
        "contrast": "heuristic_conflict",
    }
    core = AUDIT.build_core_condition(gpt, gt, status)
    assert core["status"] == "presence_count_verified_auto"
    assert core["usable"] is True
    assert core["attributes"]["count"] == 2
    assert core["field_mask"]["count"] == 1.0
    assert core["field_mask"]["shape"] == 0.0
    assert core["field_mask"]["background"] == 0.0
    assert core["field_mask"]["contrast"] == 0.0

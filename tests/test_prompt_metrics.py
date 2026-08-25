import math

import torch

from efficient_sam.prompt_metrics import PromptMetricAccumulator, greedy_match_candidates, extract_components
from efficient_sam.prompt_proposal import PromptProposal


def _proposal(coords, scores, dense, valid=None):
    coords = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(0)
    if valid is None:
        valid = [True] * len(scores[0])
    return PromptProposal(
        dense_logits=None,
        dense_probs=dense,
        candidate_xy=coords,
        candidate_scores=scores,
        candidate_valid=torch.tensor(valid, dtype=torch.bool).unsqueeze(0),
    )


def test_greedy_matching_is_one_to_one_and_tracks_duplicates():
    mask = torch.zeros(16, 16)
    mask[3:5, 3:5] = 1
    mask[10:12, 11:13] = 1
    components = extract_components(mask.numpy())
    result = greedy_match_candidates(
        candidate_xy=torch.tensor([[3.5, 3.5], [4.0, 4.0], [11.5, 10.5]]).numpy(),
        components=components,
    )
    assert result["matched_components"] == {0, 1}
    assert result["matched_candidates"] == {0, 2}
    assert result["duplicate_candidates"] == {1}


def test_prompt_metrics_budget_curve_and_area_bins():
    mask = torch.zeros(1, 16, 16)
    mask[0, 2:4, 2:4] = 1
    mask[0, 10:13, 10:13] = 1
    dense = torch.zeros(1, 1, 16, 16)
    dense[0, 0, 2:4, 2:4] = 0.9
    dense[0, 0, 10:13, 10:13] = 0.8
    proposal = _proposal(
        coords=[[2.5, 2.5], [0.0, 15.0], [11.0, 11.0]],
        scores=[0.9, 0.85, 0.8],
        dense=dense,
    )
    metrics = PromptMetricAccumulator(budgets=(1, 3))
    metrics.update(proposal, mask, ["sample"])
    result = metrics.finalize()
    at_one, at_three = result["budget_rows"]
    assert math.isclose(at_one["component_recall"], 0.5)
    assert math.isclose(at_one["prompt_precision"], 1.0)
    assert math.isclose(at_three["component_recall"], 1.0)
    assert math.isclose(at_three["prompt_precision"], 2 / 3)
    assert result["summary"]["components"] == 2
    assert {row["area_bin"] for row in result["per_component_rows"]} == {"1-9"}
    assert result["summary"]["dense_prompt_auprc"] > 0.9


def test_zero_prompt_is_valid_and_counts_as_zero_coverage():
    mask = torch.zeros(1, 8, 8)
    mask[0, 4, 4] = 1
    proposal = _proposal(
        coords=[[0.0, 0.0], [0.0, 0.0]],
        scores=[0.0, 0.0],
        dense=torch.zeros(1, 1, 8, 8),
        valid=[False, False],
    )
    metrics = PromptMetricAccumulator(budgets=(1,))
    metrics.update(proposal, mask, ["zero"])
    result = metrics.finalize()
    row = result["budget_rows"][0]
    assert row["zero_prompt_fraction"] == 1.0
    assert row["component_recall"] == 0.0
    assert math.isnan(row["prompt_precision"])

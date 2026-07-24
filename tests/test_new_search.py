# ruff: noqa: E402

import pytest
from sympy import Symbol

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from theoretical.compositional_explanations.new_search.algorithm import (
    AND_NOT,
    OR,
    reconstruct_formula,
    render_formula,
    select_unique_topk,
)


def test_select_unique_topk_keeps_the_strongest_new_semantics():
    packed_features = torch.tensor(
        [[0b01], [0b10]],
        dtype=torch.int32,
    )
    parent_vectors = packed_features[None, :, :]
    retained_ids = torch.tensor([[0, 1]])
    candidate_scores = torch.full(
        (1, 2, 2, 3),
        -torch.inf,
    )
    candidate_scores[0, 0, 1, OR] = 0.8
    candidate_scores[0, 1, 0, OR] = 0.9

    scores, indices, valid = select_unique_topk(
        candidate_scores,
        parent_vectors,
        packed_features,
        retained_ids,
        beam_size=2,
    )

    torch.testing.assert_close(scores[0, 0], torch.tensor(0.9))
    assert indices[0, 0] == 7
    assert torch.equal(valid, torch.tensor([[True, False]]))
    assert torch.isneginf(scores[0, 1])


def test_reconstruct_formula_follows_composite_parent_ids():
    feature_formulas = [
        Symbol("A"),
        Symbol("B"),
        Symbol("C"),
    ]
    operations = torch.tensor([OR, AND_NOT])
    parent_ids = torch.tensor([0, 3])
    feature_ids = torch.tensor([1, 2])

    formula = reconstruct_formula(
        formula_id=4,
        feature_count=3,
        feature_formulas=feature_formulas,
        operations=operations,
        parent_ids=parent_ids,
        feature_ids=feature_ids,
    )

    assert render_formula(formula) == "((NOT C) AND (A OR B))"

import torch

from feature import beamsearch
from feature.formula import Leaf


def test_best_leaf_is_not_reexpanded_as_frontier(monkeypatch):
    feature_a = Leaf(label="tok", token_id=1, token="A")
    feature_b = Leaf(label="tok", token_id=2, token="B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, False])),
        (feature_b, torch.tensor([True, True, False])),
    ]
    activation_vectors = torch.tensor([[True], [False], [False]])

    expanded = []
    original_expand_candidate = beamsearch.expand_candidate

    def recording_expand_candidate(candidate, feature_formula, feature_mask):
        expanded.append(candidate.formula.flatten())
        return original_expand_candidate(candidate, feature_formula, feature_mask)

    monkeypatch.setattr(beamsearch, "expand_candidate", recording_expand_candidate)

    beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        beam_size=1,
        formula_length=3,
        complexity_penalty=0.1,
        device="cpu",
    )

    assert expanded.count("tok:A") == len(feature_vectors)


def test_best_seen_can_still_return_short_formula():
    feature_a = Leaf(label="tok", token_id=1, token="A")
    feature_b = Leaf(label="tok", token_id=2, token="B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, False])),
        (feature_b, torch.tensor([True, True, False])),
    ]
    activation_vectors = torch.tensor([[True], [False], [False]])

    result = beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        beam_size=1,
        formula_length=3,
        complexity_penalty=0.1,
        device="cpu",
    )[0]

    assert result.best == ("tok:A", 1.0)

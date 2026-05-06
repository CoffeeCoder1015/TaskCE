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
    assert not hasattr(result, "samples")


def test_parallel_workers_match_serial_results():
    feature_a = Leaf(label="tok", token_id=1, token="A")
    feature_b = Leaf(label="tok", token_id=2, token="B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, True, False])),
        (feature_b, torch.tensor([False, True, True, False])),
    ]
    activation_vectors = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
            [False, False, False],
        ]
    )

    serial_results = beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        beam_size=2,
        formula_length=3,
        device="cpu",
    )
    parallel_results = beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        beam_size=2,
        formula_length=3,
        num_workers=2,
        device="cpu",
    )

    assert parallel_results == serial_results


def test_frontier_uses_heapq(monkeypatch):
    feature_a = Leaf(label="tok", token_id=1, token="A")
    feature_b = Leaf(label="tok", token_id=2, token="B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, False])),
        (feature_b, torch.tensor([False, True, False])),
    ]
    activation_vectors = torch.tensor([[True], [False], [False]])

    heap_calls = {"push": 0, "pop": 0}
    original_heappush = beamsearch.heapq.heappush
    original_heappop = beamsearch.heapq.heappop

    def recording_heappush(heap, item):
        heap_calls["push"] += 1
        return original_heappush(heap, item)

    def recording_heappop(heap):
        heap_calls["pop"] += 1
        return original_heappop(heap)

    monkeypatch.setattr(beamsearch.heapq, "heappush", recording_heappush)
    monkeypatch.setattr(beamsearch.heapq, "heappop", recording_heappop)

    beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        beam_size=2,
        formula_length=2,
        device="cpu",
    )

    assert heap_calls["push"] > 0
    assert heap_calls["pop"] > 0


def test_parallel_workers_handle_empty_activation_set():
    feature = Leaf(label="tok", token_id=1, token="A")
    feature_vectors = [(feature, torch.tensor([True, False]))]
    activation_vectors = torch.empty((2, 0), dtype=torch.bool)

    assert beamsearch.beamsearch_all(
        feature_vectors,
        activation_vectors,
        num_workers=2,
        device="cpu",
    ) == []

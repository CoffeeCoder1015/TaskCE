from sympy import Symbol
import torch

from feature.search import LevelSearch, Search, SearchResult, searchConfig, search_all


def exact_match_inputs():
    feature_a = Symbol("tok:A")
    feature_b = Symbol("tok:B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, False])),
        (feature_b, torch.tensor([False, True, False])),
    ]
    neuron = torch.tensor([True, False, False])
    return neuron, feature_vectors


def test_search_returns_best_formula_and_score_for_exact_match():
    neuron, feature_vectors = exact_match_inputs()

    best_formula, best_score = Search(neuron, feature_vectors, config=searchConfig())

    assert best_formula == "tok:A"
    assert best_score == 1.0


def test_level_search_returns_best_formula_and_score_for_exact_match():
    neuron, feature_vectors = exact_match_inputs()

    best_formula, best_score = LevelSearch(neuron, feature_vectors, config=searchConfig())

    assert best_formula == "tok:A"
    assert best_score == 1.0


def test_search_all_serial_returns_indexed_results():
    feature_a = Symbol("tok:A")
    feature_b = Symbol("tok:B")
    feature_vectors = [
        (feature_a, torch.tensor([True, False, False])),
        (feature_b, torch.tensor([False, True, False])),
    ]
    activation_vectors = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
        ]
    )

    results = search_all(activation_vectors, feature_vectors, device="cpu")

    assert results == [
        SearchResult(activation_index=0, best_formula="tok:A", best_score=1.0),
        SearchResult(activation_index=1, best_formula="tok:B", best_score=1.0),
    ]


def test_search_all_parallel_matches_serial_results():
    feature_a = Symbol("tok:A")
    feature_b = Symbol("tok:B")
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
    config = searchConfig(pruned_queue_size=2, formula_length=3)

    serial_results = search_all(
        activation_vectors,
        feature_vectors,
        num_workers=1,
        device="cpu",
        config=config,
    )
    parallel_results = search_all(
        activation_vectors,
        feature_vectors,
        num_workers=2,
        device="cpu",
        config=config,
    )

    assert parallel_results == serial_results


def test_search_all_parallel_handles_empty_activation_set():
    feature = Symbol("tok:A")
    feature_vectors = [(feature, torch.tensor([True, False]))]
    activation_vectors = torch.empty((2, 0), dtype=torch.bool)

    assert search_all(
        activation_vectors,
        feature_vectors,
        num_workers=2,
        device="cpu",
    ) == []

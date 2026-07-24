# ruff: noqa: E402

from itertools import product

import pytest
from sympy import Symbol

torch = pytest.importorskip("torch")

from theoretical.compositional_explanations.semantic_dag_search.algorithm import (
    SearchConfig,
    SearchOutcome,
    search,
    search_batch,
)
from theoretical.compositional_explanations.semantic_dag_search.features import (
    prepare_feature_matrix,
    prepare_initial_features,
)
from theoretical.compositional_explanations.semantic_dag_search.parallel import (
    NeuronSearchResult,
    search_all,
)
from theoretical.compositional_explanations.semantic_dag_search.scoring import (
    pair_iou_scores,
)


def truth_table_features():
    rows = list(product((False, True), repeat=4))
    formulas = tuple(Symbol(name) for name in ("A", "B", "C", "D"))
    features = [
        (
            formula,
            torch.tensor([row[index] for row in rows]),
        )
        for index, formula in enumerate(formulas)
    ]
    target = torch.tensor(
        [
            (row[0] or row[1]) and (row[2] or row[3])
            for row in rows
        ]
    )
    return target, features


def test_search_returns_best_atomic_formula_for_exact_match():
    features = [
        (Symbol("tok:A"), torch.tensor([True, False, False])),
        (Symbol("tok:B"), torch.tensor([False, True, False])),
    ]

    outcome = search(
        torch.tensor([True, False, False]),
        features,
        SearchConfig(maximum_formula_length=1),
    )

    assert outcome == SearchOutcome(formula="tok:A", iou=1.0)


def test_search_composes_retained_subformulas_from_prior_levels():
    target, features = truth_table_features()

    three_term_outcome = search(
        target,
        features,
        SearchConfig(maximum_formula_length=3, beam_size=4),
    )
    four_term_outcome = search(
        target,
        features,
        SearchConfig(maximum_formula_length=4, beam_size=4),
    )

    assert three_term_outcome.iou < 1.0
    assert four_term_outcome.iou == 1.0
    assert " AND " in four_term_outcome.formula
    assert four_term_outcome.formula.count(" OR ") == 2
    assert all(
        atom in four_term_outcome.formula
        for atom in ("A", "B", "C", "D")
    )


def test_search_records_when_no_feature_overlaps_the_neuron():
    features = [
        (Symbol("tok:A"), torch.tensor([True, False, False])),
        (Symbol("tok:B"), torch.tensor([False, True, False])),
    ]

    outcome = search(
        torch.tensor([False, False, True]),
        features,
    )

    assert outcome == SearchOutcome(
        formula="LOW_ACTS_PRUNED",
        iou=0.0,
    )


def test_search_batch_matches_independent_searches():
    target, features = truth_table_features()
    second_target = features[0][1] | features[2][1]
    targets = torch.stack((target, second_target), dim=1)
    config = SearchConfig(maximum_formula_length=4, beam_size=4)
    prepared_features = prepare_feature_matrix(
        prepare_initial_features(features),
        torch.device("cpu"),
    )

    batched = search_batch(targets, prepared_features, config)
    independent = [
        search(targets[:, index], features, config)
        for index in range(targets.shape[1])
    ]

    assert batched == independent


def test_pair_count_scoring_matches_direct_boolean_operations():
    target = torch.tensor([True, True, False, False, True])
    left = torch.tensor([True, False, True, False, True])
    rights = torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, True, False, True],
        ]
    )
    pair_sizes = (rights & left).sum(dim=1).float()[None, :]
    pair_target_sizes = (
        rights & left & target
    ).sum(dim=1).float()[None, :]

    scores = pair_iou_scores(
        target_size=target.sum().float(),
        left_sizes=left.sum().float()[None],
        left_target_sizes=(left & target).sum().float()[None],
        right_sizes=rights.sum(dim=1).float(),
        right_target_sizes=(rights & target).sum(dim=1).float(),
        pair_sizes=pair_sizes,
        pair_target_sizes=pair_target_sizes,
    )
    direct_candidates = torch.stack(
        [
            left & rights[0],
            left | rights[0],
            left & ~rights[0],
            rights[0] & ~left,
            left & rights[1],
            left | rights[1],
            left & ~rights[1],
            rights[1] & ~left,
        ]
    )
    intersections = (direct_candidates & target).sum(dim=1).float()
    unions = (
        direct_candidates.sum(dim=1)
        + target.sum()
        - intersections
    ).float()
    direct_scores = intersections / unions

    assert scores.flatten().tolist() == pytest.approx(
        direct_scores.tolist()
    )


def test_search_rejects_one_atomic_formula_with_multiple_vectors():
    features = [
        (Symbol("tok:A"), torch.tensor([True, False])),
        (Symbol("tok:A"), torch.tensor([False, True])),
    ]

    with pytest.raises(ValueError, match="maps to multiple vectors"):
        search(torch.tensor([True, False]), features)


def test_feature_preparation_keeps_first_identical_trajectory(capsys):
    features = [
        (Symbol("tok:first"), torch.tensor([True, False, True])),
        (Symbol("tok:duplicate"), torch.tensor([True, False, True])),
        (Symbol("tok:other"), torch.tensor([False, True, False])),
    ]

    prepared = prepare_initial_features(features)

    assert [formula for formula, _vector in prepared] == [
        Symbol("tok:first"),
        Symbol("tok:other"),
    ]
    warning = capsys.readouterr().err
    assert (
        "identical initial feature trajectories: "
        "tok:first == tok:duplicate"
    ) in warning


def test_search_all_preserves_neuron_id_order():
    features = [
        (Symbol("tok:A"), torch.tensor([True, False, False])),
        (Symbol("tok:B"), torch.tensor([False, True, False])),
    ]
    activation_vectors = torch.tensor(
        [
            [True, False],
            [False, True],
            [False, False],
        ]
    )

    results = search_all(
        activation_vectors,
        [42, 11],
        features,
        device="cpu",
        config=SearchConfig(maximum_formula_length=1),
    )

    assert results == [
        NeuronSearchResult(neuron_id=42, formula="tok:A", iou=1.0),
        NeuronSearchResult(neuron_id=11, formula="tok:B", iou=1.0),
    ]

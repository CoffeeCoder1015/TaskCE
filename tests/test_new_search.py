# ruff: noqa: E402

from dataclasses import fields
from inspect import Parameter, signature

import pytest
from sympy import Symbol

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from theoretical.compositional_explanations.new_search.algorithm import (
    AND_NOT,
    OR,
    SearchConfig as NewSearchConfig,
    SearchResult as NewSearchResult,
    reconstruct_formula,
    render_formula,
    search_all as new_search_all,
    select_unique_topk,
)
from theoretical.compositional_explanations.search.algorithm import (
    SearchConfig as ReferenceSearchConfig,
    SearchResult as ReferenceSearchResult,
    search_all as reference_search_all,
)


def test_search_algorithms_expose_the_same_drop_in_interface():
    expected_parameters = (
        "activation_vectors",
        "feature_vectors",
        "num_workers",
        "device",
        "config",
    )
    reference_parameters = signature(reference_search_all).parameters
    new_parameters = signature(new_search_all).parameters

    assert tuple(reference_parameters) == expected_parameters
    assert tuple(new_parameters) == expected_parameters
    for parameter_name in expected_parameters:
        reference_parameter = reference_parameters[parameter_name]
        new_parameter = new_parameters[parameter_name]
        assert reference_parameter.kind == new_parameter.kind
        assert reference_parameter.default == new_parameter.default

    assert reference_parameters["activation_vectors"].default is Parameter.empty
    assert reference_parameters["feature_vectors"].default is Parameter.empty


def test_search_algorithms_share_configuration_and_result_names():
    reference_config = ReferenceSearchConfig()
    new_config = NewSearchConfig()

    assert reference_config.maximum_formula_length == (
        new_config.maximum_formula_length
    )
    assert reference_config.beam_size == new_config.beam_size
    assert [field.name for field in fields(ReferenceSearchResult)] == [
        field.name for field in fields(NewSearchResult)
    ]


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

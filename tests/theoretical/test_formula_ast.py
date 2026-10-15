import pytest
import z3

from theoretical.compositional_explanations.formula_ast.analysis import (
    formula_to_string,
    parse_formula,
)


premise_a = z3.Bool("premise:a")
premise_b = z3.Bool("premise:b")
hypothesis_a = z3.Bool("hypothesis:a")
walking = z3.Bool("hypothesis:walking")
humans = z3.Bool("hypothesis:humans")
hypothesis_people = z3.Bool("hypothesis:people")
premise_people = z3.Bool("premise:people")

FORMULAS = [
    pytest.param(
        "(premise:a AND hypothesis:a)",
        z3.And(premise_a, hypothesis_a),
        id="and-positive-atom",
    ),
    pytest.param(
        "(premise:a OR hypothesis:a)",
        z3.Or(premise_a, hypothesis_a),
        id="or-positive-atom",
    ),
    pytest.param(
        "(premise:a AND (NOT hypothesis:a))",
        z3.And(premise_a, z3.Not(hypothesis_a)),
        id="and-negated-atom",
    ),
    pytest.param(
        "(premise:a AND hypothesis:a AND premise:b)",
        z3.And(premise_a, hypothesis_a, premise_b),
        id="nary-and",
    ),
    pytest.param(
        "(premise:a OR hypothesis:a OR premise:b)",
        z3.Or(premise_a, hypothesis_a, premise_b),
        id="nary-or",
    ),
    pytest.param(
        "((premise:a OR hypothesis:a) AND (NOT premise:b))",
        z3.And(z3.Or(premise_a, hypothesis_a), z3.Not(premise_b)),
        id="mixed-nested",
    ),
    pytest.param(
        "(hypothesis:walking OR ((NOT premise:people) AND "
        "(hypothesis:humans OR hypothesis:people OR hypothesis:walking)))",
        z3.Or(walking, z3.And(z3.Not(premise_people), z3.Or(humans, hypothesis_people, walking))),
        id="repeated-name-across-branches",
    ),
]


def assert_equivalent(actual: z3.BoolRef, expected: z3.BoolRef) -> None:
    solver = z3.Solver()
    solver.add(z3.Xor(actual, expected))
    result = solver.check()
    if result == z3.sat:
        pytest.fail(f"formulas differ under {solver.model()}")
    if result == z3.unknown:
        pytest.fail(f"equivalence check returned unknown: {solver.reason_unknown()}")
    assert result == z3.unsat


@pytest.mark.parametrize(("text", "expected"), FORMULAS)
def test_parse_formula_matches_independent_expression(text, expected):
    assert_equivalent(parse_formula(text), expected)


@pytest.mark.parametrize(("text", "expected"), FORMULAS)
def test_serialized_expression_parses_equivalently(text, expected):
    rendered = formula_to_string(expected)
    assert_equivalent(parse_formula(rendered), expected)


@pytest.mark.parametrize(
    ("expected", "rendered"),
    [
        pytest.param(
            z3.And(premise_a, hypothesis_a),
            "(premise:a AND hypothesis:a)",
            id="and",
        ),
        pytest.param(
            z3.Or(premise_a, hypothesis_a, premise_b),
            "(premise:a OR hypothesis:a OR premise:b)",
            id="nary-or",
        ),
        pytest.param(
            z3.And(premise_a, z3.Not(hypothesis_a)),
            "(premise:a AND (NOT hypothesis:a))",
            id="and-not",
        ),
        pytest.param(
            z3.Or(walking, z3.And(z3.Not(premise_people), z3.Or(humans, hypothesis_people, walking))),
            "(hypothesis:walking OR ((NOT premise:people) AND "
            "(hypothesis:humans OR hypothesis:people OR hypothesis:walking)))",
            id="repeated-name-nested",
        ),
    ],
)
def test_formula_to_string_format(expected, rendered):
    assert formula_to_string(expected) == rendered

from feature.formula import And, Constant, Leaf, Not
from feature.search import (
    assert_explanation_formula,
    is_explanation_formula,
    length_penalty_factor,
    simplified_formula_key,
)


def test_length_penalty_factor_is_clamped_linear():
    leaf = Leaf(label="tok", token_id=1, token="A")
    length_two = And(left=leaf, right=leaf)

    assert length_penalty_factor(leaf, 0.01) == 1.0
    assert length_penalty_factor(length_two, 0.01) == 0.99
    assert length_penalty_factor(length_two, 2.0) == 0.0
    constant = And(left=leaf, right=Not(leaf)).simplify_tree()[0]
    assert length_penalty_factor(constant, 0.01) == 1.0


def test_simplified_formula_key_uses_simplified_tree_hash():
    leaf = Leaf(label="tok", token_id=1, token="A")

    assert simplified_formula_key(And(left=leaf, right=leaf)) == leaf
    assert hash(simplified_formula_key(And(left=leaf, right=leaf))) == hash(leaf)


def test_constants_are_not_top_level_explanations():
    leaf = Leaf(label="tok", token_id=1, token="A")

    assert is_explanation_formula(leaf)
    assert not is_explanation_formula(Constant(value=True))


def test_constant_top_level_explanations_raise_assertion():
    try:
        assert_explanation_formula(Constant(value=True))
    except AssertionError as error:
        assert "Top-level explanation simplified to constant TRUE" in str(error)
    else:
        raise AssertionError("Expected constant explanation assertion")

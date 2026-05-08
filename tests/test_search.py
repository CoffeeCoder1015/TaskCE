from feature.formula import And, Leaf
from feature.search import length_penalty_factor


def test_length_penalty_factor_is_clamped_linear():
    leaf = Leaf(label="tok", token_id=1, token="A")
    length_two = And(left=leaf, right=leaf)

    assert length_penalty_factor(leaf, 0.01) == 1.0
    assert length_penalty_factor(length_two, 0.01) == 0.99
    assert length_penalty_factor(length_two, 2.0) == 0.0

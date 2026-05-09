from feature.formula import And, Constant, Leaf, Not, Or, to_tree_structure
from sympy import Symbol
from sympy.logic.boolalg import And as SympyAnd


def test_to_tree_structure_restores_leaf_from_metadata():
    metadata = {"tok:A": ["tok", 1, "A"]}

    tree = to_tree_structure(Symbol("tok:A"), metadata)

    assert tree == Leaf(label="tok", token_id=1, token="A")


def test_simplify_tree_returns_custom_tree_and_metadata():
    leaf = Leaf(label="tok", token_id=1, token="A")

    tree, metadata = leaf.simplify_tree()

    assert tree == leaf
    assert metadata == {"tok:A": ["tok", 1, "A"]}


def test_simplify_tree_collapses_duplicate_and():
    leaf = Leaf(label="tok", token_id=1, token="A")

    tree, _ = And(left=leaf, right=leaf).simplify_tree()

    assert tree == leaf


def test_flatten_sympy_does_not_simplify_intermediate_expression():
    leaf = Leaf(label="tok", token_id=1, token="A")

    expression, metadata = And(left=leaf, right=leaf).flatten_sympy()

    assert expression.func is SympyAnd
    assert expression.args == (Symbol("tok:A"), Symbol("tok:A"))
    assert metadata == {"tok:A": ["tok", 1, "A"]}


def test_simplify_tree_converts_unary_and_binary_sympy_nodes():
    feature_a = Leaf(label="tok", token_id=1, token="A")
    feature_b = Leaf(label="tok", token_id=2, token="B")

    tree, _ = Or(left=Not(feature_a), right=feature_b).simplify_tree()

    assert tree.flatten() == "(tok:B OR (NOT tok:A))"


def test_simplify_tree_converts_boolean_constants():
    leaf = Leaf(label="tok", token_id=1, token="A")

    true_tree, _ = Or(left=leaf, right=Not(leaf)).simplify_tree()
    false_tree, _ = And(left=leaf, right=Not(leaf)).simplify_tree()

    assert true_tree == Constant(value=True)
    assert false_tree == Constant(value=False)

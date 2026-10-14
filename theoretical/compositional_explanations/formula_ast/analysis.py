"""Load compositional search CSVs for subsequent formula experiments."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
import z3


@dataclass(frozen=True)
class FormulaNode:
    """One written atom, Boolean constant, or logical operation."""

    kind: str
    value: str | bool | None = None
    children: tuple["FormulaNode", ...] = ()


def to_z3(node: FormulaNode) -> z3.BoolRef:
    """Translate a parsed formula using shared Boolean feature names."""
    if node.kind == "atom":
        return z3.Bool(node.value)
    if node.kind == "constant":
        return z3.BoolVal(node.value)
    if node.kind == "NOT":
        return z3.Not(to_z3(node.children[0]))
    if node.kind == "AND":
        return z3.And(*(to_z3(child) for child in node.children))
    if node.kind == "OR":
        return z3.Or(*(to_z3(child) for child in node.children))
    raise ValueError(f"Unknown formula node kind {node.kind!r}")


def parse_formula(text: str) -> FormulaNode:
    """Preserve the written tree described by search's expression grammar.

    expression := atom | True | False | '(' compound ')'
    compound   := NOT expression
                | expression AND expression (AND expression)*
                | expression OR expression (OR expression)*
    """
    tokens = deque(re.findall(r"\(|\)|[^\s()]+", text))

    def peek() -> str | None:
        return tokens[0] if tokens else None

    def take() -> str:
        if not tokens:
            raise ValueError("Unexpected end of formula")
        return tokens.popleft()

    def parse_expression() -> FormulaNode:
        token = take()
        if token != "(":
            if token in {"AND", "OR", "NOT", ")", "LOW_ACTS_PRUNED"}:
                raise ValueError(f"Expected an atom, got {token!r}")
            if token in {"True", "False"}:
                return FormulaNode("constant", token == "True")
            return FormulaNode("atom", token)

        node = parse_compound()
        if peek() != ")":
            raise ValueError("Expected closing parenthesis")
        take()
        return node

    def parse_compound() -> FormulaNode:
        if peek() == "NOT":
            take()
            return FormulaNode("NOT", children=(parse_expression(),))

        children = [parse_expression()]
        operator = peek()
        if operator not in {"AND", "OR"}:
            raise ValueError("Expected AND or OR")
        while peek() == operator:
            take()
            children.append(parse_expression())
        return FormulaNode(operator, children=tuple(children))

    node = parse_expression()
    if tokens:
        raise ValueError(f"Unexpected token {peek()!r}")
    return node


def load_compexp(path: str | Path) -> pd.DataFrame:
    """Retain search metadata and add an AST column for each usable formula.

    Empty formulas and LOW_ACTS_PRUNED have no AST. Row order and neuron IDs
    remain those of the input CSV; formulas are never simplified or reordered.
    """
    rows = pd.read_csv(path, dtype={"formula": str}, keep_default_na=False)
    missing = {"neuron", "formula", "iou"} - set(rows.columns)
    if missing:
        raise ValueError(f"Missing search CSV columns: {', '.join(sorted(missing))}")
    if "ast" in rows.columns:
        raise ValueError("Input CSV already contains an ast column")
    neurons = pd.to_numeric(rows["neuron"], errors="raise")
    if ((neurons < 0) | (neurons % 1 != 0)).any() or neurons.duplicated().any():
        raise ValueError("Neuron IDs must be unique nonnegative integers")
    rows["neuron"] = neurons.astype("int64")
    scores = pd.to_numeric(rows["iou"], errors="raise")
    if not scores.between(0, 1).all():
        raise ValueError("IoU scores must be finite and between 0 and 1")
    rows["iou"] = scores

    nodes = []
    for row in rows.itertuples(index=False):
        formula = row.formula.strip()
        if formula in {"", "LOW_ACTS_PRUNED"}:
            nodes.append(None)
            continue
        try:
            nodes.append(parse_formula(formula))
        except ValueError as error:
            raise ValueError(f"Neuron {row.neuron}: {error}") from error
    rows["ast"] = nodes
    return rows

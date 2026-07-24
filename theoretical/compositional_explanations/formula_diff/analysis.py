from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sympy import Symbol
from sympy.logic.boolalg import And as SymAnd
from sympy.logic.boolalg import Not as SymNot
from sympy.logic.boolalg import Or as SymOr

LOW_ACTS_PRUNED = "LOW_ACTS_PRUNED"
COMMUTATIVE_OPERATORS = {"AND", "OR"}
CLOSEST_BUCKETS_FILENAME = "closest_finetuned_formula_match_buckets.json"
HEATMAP_DPI = 300
DEFAULT_LAYER = "14th"
DEFAULT_TASKS = ("claim", "snli")


def write_formula_diff_generation(
    task: str,
    base_csv: Path,
    finetuned_csv: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_rows = read_formula_rows(base_csv)
    finetuned_rows = read_formula_rows(finetuned_csv)

    same_neuron_path = output_dir / f"{task}_same_neuron_structure_diff.csv"
    old_closest_path = output_dir / f"{task}_closest_finetuned_formula_matches.csv"
    closest_buckets_path = output_dir / CLOSEST_BUCKETS_FILENAME
    matrix_path = output_dir / f"{task}_formula_diff_score_matrix.csv"
    heatmap_path = output_dir / f"{task}_formula_diff_score_heatmap.png"
    old_closest_path.unlink(missing_ok=True)

    same_neuron_rows: list[dict[str, object]] = []
    score_matrix_rows: list[list[object]] = []
    heatmap_rows: list[list[int | None]] = []
    closest_bucket_rows: list[dict[str, object]] = []

    for row_index, base in enumerate(base_rows):
        score_matrix_row: list[object] = [base.neuron]
        heatmap_row: list[int | None] = []
        row_cells: list[FormulaDiffCell] = []
        lowest_score: int | None = None
        closest_cells: list[FormulaDiffCell] = []

        for finetuned in finetuned_rows:
            if base.formula_node is None or finetuned.formula_node is None:
                cell = FormulaDiffCell(
                    finetuned_neuron=finetuned.neuron,
                    finetuned_formula=finetuned.canonical_formula,
                    diff=None,
                    score=None,
                )
            else:
                diff_result = diff_formulas(base.formula_node, finetuned.formula_node)
                cell = FormulaDiffCell(
                    finetuned_neuron=finetuned.neuron,
                    finetuned_formula=finetuned.canonical_formula,
                    diff=diff_result.diff,
                    score=diff_result.score,
                )

            row_cells.append(cell)
            score_matrix_row.append(cell.score)
            heatmap_row.append(cell.score)

            if cell.score is not None:
                if lowest_score is None or cell.score < lowest_score:
                    lowest_score = cell.score
                    closest_cells = [cell]
                elif cell.score == lowest_score:
                    closest_cells.append(cell)

        same_neuron_rows.append(
            same_neuron_row_from_cell(task, base, row_cells[row_index])
        )
        score_matrix_rows.append(score_matrix_row)
        heatmap_rows.append(heatmap_row)
        closest_bucket_rows.append(
            closest_block_from_row(base, lowest_score, closest_cells)
        )

    save_formula_diff_outputs(
        task,
        same_neuron_path,
        closest_buckets_path,
        matrix_path,
        same_neuron_rows,
        closest_bucket_rows,
        score_matrix_rows,
        [row.neuron for row in finetuned_rows],
    )
    write_formula_diff_score_heatmap(
        heatmap_rows,
        heatmap_path,
        title=f"{task} formula diff score matrix",
    )


@dataclass(frozen=True)
class FormulaNode:
    kind: str | None = None
    value: str | None = None
    children: tuple["FormulaNode", ...] = ()
    expr: Any | None = None

    def __post_init__(self) -> None:
        expr = self.expr
        if self.kind == "pruned":
            object.__setattr__(self, "expr", None)
            object.__setattr__(self, "value", None)
            object.__setattr__(self, "children", ())
            return

        if expr is None:
            if self.kind == "atom":
                expr = Symbol(self.value or "")
            elif self.kind == "not":
                expr = SymNot(self.children[0].expr)
            elif self.kind == "AND":
                expr = SymAnd(*(child.expr for child in self.children))
            elif self.kind == "OR":
                expr = SymOr(*(child.expr for child in self.children))
            else:
                raise ValueError(f"unknown formula node kind {self.kind!r}")
            object.__setattr__(self, "expr", expr)

        if isinstance(expr, Symbol):
            object.__setattr__(self, "kind", "atom")
            object.__setattr__(self, "value", str(expr))
            object.__setattr__(self, "children", ())
            return

        if expr.func is SymNot:
            object.__setattr__(self, "kind", "not")
        elif expr.func is SymAnd:
            object.__setattr__(self, "kind", "AND")
        elif expr.func is SymOr:
            object.__setattr__(self, "kind", "OR")
        else:
            object.__setattr__(self, "kind", "atom")
            object.__setattr__(self, "value", str(expr))
            object.__setattr__(self, "children", ())
            return

        object.__setattr__(self, "value", None)
        object.__setattr__(
            self,
            "children",
            normalize_children(
                self.kind,
                tuple(FormulaNode(expr=child) for child in expr.args),
            ),
        )

    def canonical(self) -> str:
        if self.kind == "pruned":
            return LOW_ACTS_PRUNED
        if self.kind == "atom":
            return self.value or ""
        if self.kind == "not":
            return f"(NOT {self.children[0].canonical()})"

        separator = f" {self.kind} "
        return f"({separator.join(child.canonical() for child in self.children)})"

def normalize_children(
    kind: str | None,
    children: tuple[FormulaNode, ...],
) -> tuple[FormulaNode, ...]:
    if kind in COMMUTATIVE_OPERATORS:
        return tuple(sorted(children, key=structural_sort_key))
    return children


def structural_sort_key(
    node: FormulaNode,
) -> tuple[tuple[str, ...], tuple[int, int, int], str, str]:
    return (
        leaf_terms(node),
        subtree_stats(node),
        node_label(node),
        node.canonical(),
    )


def leaf_terms(node: FormulaNode) -> tuple[str, ...]:
    if node.kind == "atom":
        return (node.value or "",)
    terms: list[str] = []
    for child in node.children:
        terms.extend(leaf_terms(child))
    return tuple(sorted(terms))


def tree_size(node: FormulaNode) -> int:
    return 1 + sum(tree_size(child) for child in node.children)


def subtree_stats(node: FormulaNode) -> tuple[int, int, int]:
    if node.kind == "pruned":
        return (0, 0, 0)
    if node.kind == "atom":
        return (1, 1, 0)

    child_node_count = 0
    child_term_count = 0
    child_operator_count = 0
    for child in node.children:
        node_count, term_count, operator_count = subtree_stats(child)
        child_node_count += node_count
        child_term_count += term_count
        child_operator_count += operator_count
    return (
        1 + child_node_count,
        child_term_count,
        1 + child_operator_count,
    )


def node_label(node: FormulaNode) -> str:
    if node.kind == "atom":
        return f"term:{node.value or ''}"
    return node.kind or ""


def parse_formula(text: str) -> FormulaNode:
    if text == LOW_ACTS_PRUNED:
        return FormulaNode(kind="pruned")

    tokens = _tokenize_formula(text)
    index = 0

    def peek() -> str | None:
        if index >= len(tokens):
            return None
        return tokens[index]

    def consume(expected: str) -> None:
        nonlocal index
        actual = peek()
        if actual != expected:
            raise ValueError(f"expected {expected!r}, got {actual!r}")
        index += 1

    def parse_expr() -> Any:
        nonlocal index
        token = peek()
        if token is None:
            raise ValueError("unexpected end of formula")

        if token != "(":
            index += 1
            return Symbol(token)

        consume("(")
        if peek() == "NOT":
            consume("NOT")
            child = parse_expr()
            consume(")")
            return SymNot(child)

        first = parse_expr()
        operator = peek()
        if operator not in COMMUTATIVE_OPERATORS:
            raise ValueError(f"expected AND or OR, got {operator!r}")

        children = [first]
        while peek() in COMMUTATIVE_OPERATORS:
            next_operator = peek()
            if next_operator != operator:
                raise ValueError(
                    f"mixed operators in one parenthesized expression: "
                    f"{operator!r} then {next_operator!r}"
                )
            consume(operator)
            children.append(parse_expr())

        consume(")")
        if operator == "AND":
            return SymAnd(*children)
        return SymOr(*children)

    expr = parse_expr()
    if peek() is not None:
        raise ValueError(f"unexpected token {peek()!r}")
    return FormulaNode(expr=expr)


def _tokenize_formula(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "()":
            tokens.append(char)
            index += 1
            continue

        start = index
        while index < len(text) and not text[index].isspace() and text[index] not in "()":
            index += 1
        tokens.append(text[start:index])

    return tokens


# Per-cell formula comparison.


@dataclass(frozen=True)
class FormulaDiffResult:
    diff: dict[str, object]
    added_node_count: int = 0
    removed_node_count: int = 0
    penalty: int = 0

    @property
    def score(self) -> int:
        return self.added_node_count + self.removed_node_count + self.penalty


def diff_formulas(base: FormulaNode, finetuned: FormulaNode) -> FormulaDiffResult:
    if finetuned.canonical() == base.canonical():
        return FormulaDiffResult(
            diff={
                "kind": "shared",
                "formula": finetuned.canonical(),
            }
        )

    if node_label(finetuned) != node_label(base):
        return FormulaDiffResult(
            diff={
                "kind": "changed",
                "base": base.canonical(),
                "finetuned": finetuned.canonical(),
            },
            added_node_count=tree_size(finetuned),
            removed_node_count=tree_size(base),
        )

    if finetuned.kind in COMMUTATIVE_OPERATORS:
        child_results: list[FormulaDiffResult] = []
        used_base: set[int] = set()
        used_finetuned: set[int] = set()

        for finetuned_index, finetuned_child in enumerate(finetuned.children):
            match_index = next(
                (
                    index
                    for index, base_child in enumerate(base.children)
                    if index not in used_base
                    and base_child.canonical() == finetuned_child.canonical()
                ),
                None,
            )
            if match_index is None:
                continue

            used_base.add(match_index)
            used_finetuned.add(finetuned_index)
            child_results.append(
                diff_formulas(base.children[match_index], finetuned_child)
            )

        remaining_base = [
            child
            for index, child in enumerate(base.children)
            if index not in used_base
        ]
        remaining_finetuned = [
            child
            for index, child in enumerate(finetuned.children)
            if index not in used_finetuned
        ]

        pair_count = min(len(remaining_base), len(remaining_finetuned))
        for index in range(pair_count):
            child_results.append(
                diff_formulas(remaining_base[index], remaining_finetuned[index])
            )

        for base_child in remaining_base[pair_count:]:
            child_results.append(
                FormulaDiffResult(
                    diff={"kind": "base_only", "formula": base_child.canonical()},
                    removed_node_count=tree_size(base_child),
                )
            )
        for finetuned_child in remaining_finetuned[pair_count:]:
            child_results.append(
                FormulaDiffResult(
                    diff={
                        "kind": "finetuned_only",
                        "formula": finetuned_child.canonical(),
                    },
                    added_node_count=tree_size(finetuned_child),
                )
            )

        return FormulaDiffResult(
            diff={
                "kind": "operator",
                "operator": finetuned.kind,
                "children": [result.diff for result in child_results],
            },
            added_node_count=sum(result.added_node_count for result in child_results),
            removed_node_count=sum(
                result.removed_node_count for result in child_results
            ),
            penalty=sum(result.penalty for result in child_results),
        )

    if finetuned.kind == "not":
        child_result = diff_formulas(base.children[0], finetuned.children[0])
        return FormulaDiffResult(
            diff={
                "kind": "operator",
                "operator": "NOT",
                "children": [child_result.diff],
            },
            added_node_count=child_result.added_node_count,
            removed_node_count=child_result.removed_node_count,
            penalty=child_result.penalty,
        )

    return FormulaDiffResult(
        diff={
            "kind": "changed",
            "base": base.canonical(),
            "finetuned": finetuned.canonical(),
        },
        added_node_count=tree_size(finetuned),
        removed_node_count=tree_size(base),
    )


def formula_structure_diff(finetuned: FormulaNode, base: FormulaNode) -> dict[str, object]:
    return diff_formulas(base, finetuned).diff


# Matrix-level formula diff generation.

def formula_diff_score(diff_node: dict[str, object]) -> int:
    kind = diff_node["kind"]
    if kind == "shared":
        return 0
    if kind in {"base_only", "finetuned_only"}:
        return formula_text_size(str(diff_node["formula"]))
    if kind == "changed":
        return (
            formula_text_size(str(diff_node["base"]))
            + formula_text_size(str(diff_node["finetuned"]))
        )
    return sum(
        formula_diff_score(child)
        for child in diff_node.get("children", [])
    )


@lru_cache(maxsize=None)
def formula_text_size(text: str) -> int:
    return tree_size(parse_formula(text))


def formula_diff_text(diff_node: dict[str, object], depth: int = 0) -> str:
    indent = "  " * depth
    kind = diff_node["kind"]
    if kind == "operator":
        child_text = [
            formula_diff_text(child, depth + 1)
            for child in diff_node["children"]
        ]
        return "\n".join([f"{indent}{diff_node['operator']}"] + child_text)
    if kind == "shared":
        return f"{indent}= {diff_node['formula']}"
    if kind == "base_only":
        return f"{indent}- base {diff_node['formula']}"
    if kind == "finetuned_only":
        return f"{indent}+ fine-tuned {diff_node['formula']}"
    return "\n".join(
        [
            f"{indent}- base {diff_node['base']}",
            f"{indent}+ fine-tuned {diff_node['finetuned']}",
        ]
    )


@dataclass(frozen=True)
class FormulaCsvRow:
    neuron: int
    formula: str
    formula_node: FormulaNode | None
    canonical_formula: str


@dataclass(frozen=True)
class FormulaDiffCell:
    finetuned_neuron: int
    finetuned_formula: str
    diff: dict[str, object] | None
    score: int | None


def read_formula_rows(path: Path) -> list[FormulaCsvRow]:
    frame = pd.read_csv(path)
    rows: list[FormulaCsvRow] = []

    for row in frame.sort_values("neuron").itertuples(index=False):
        formula = getattr(row, "formula")
        formula_text = LOW_ACTS_PRUNED if pd.isna(formula) else str(formula)
        if formula_text == LOW_ACTS_PRUNED:
            rows.append(
                FormulaCsvRow(
                    neuron=int(getattr(row, "neuron")),
                    formula=formula_text,
                    formula_node=None,
                    canonical_formula=formula_text,
                )
            )
            continue

        formula_node = parse_formula(formula_text)
        rows.append(
            FormulaCsvRow(
                neuron=int(getattr(row, "neuron")),
                formula=formula_text,
                formula_node=formula_node,
                canonical_formula=formula_node.canonical(),
            )
        )

    return rows


def same_neuron_row_from_cell(
    task: str,
    base: FormulaCsvRow,
    cell: FormulaDiffCell,
) -> dict[str, object]:
    return {
        "task": task,
        "neuron": base.neuron,
        "diff_score": cell.score,
        "base_formula": base.canonical_formula,
        "finetuned_formula": cell.finetuned_formula,
        "formula_diff": json.dumps(cell.diff, ensure_ascii=False),
    }


def closest_block_from_row(
    base: FormulaCsvRow,
    lowest_score: int | None,
    closest_cells: list[FormulaDiffCell],
) -> dict[str, object]:
    return {
        "neuron_id": base.neuron,
        "base_formula": base.canonical_formula,
        "lowest_diff_score": lowest_score,
        "candidates": [
            {
                "neuron_id": cell.finetuned_neuron,
                "formula": cell.finetuned_formula,
                "diff": cell.diff,
            }
            for cell in closest_cells
        ],
    }


def save_formula_diff_outputs(
    task: str,
    same_neuron_path: Path,
    closest_buckets_path: Path,
    matrix_path: Path,
    same_neuron_rows: list[dict[str, object]],
    closest_bucket_rows: list[dict[str, object]],
    score_matrix_rows: list[list[object]],
    finetuned_neurons: list[int],
) -> None:
    pd.DataFrame(
        same_neuron_rows,
        columns=[
            "task",
            "neuron",
            "diff_score",
            "base_formula",
            "finetuned_formula",
            "formula_diff",
        ],
        dtype=object,
    ).to_csv(same_neuron_path, index=False)
    pd.DataFrame(
        score_matrix_rows,
        columns=["base_neuron"]
        + [f"finetuned_{neuron}" for neuron in finetuned_neurons],
        dtype=object,
    ).to_csv(matrix_path, index=False)

    if closest_buckets_path.exists():
        payload = json.loads(closest_buckets_path.read_text(encoding="utf-8"))
    else:
        payload = {}

    payload[task] = closest_bucket_rows
    closest_buckets_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_formula_diff_score_heatmap(
    score_rows: list[list[int | None]],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    row_count = max(len(score_rows), 1)
    column_count = max((len(row) for row in score_rows), default=1)
    pixels_per_cell = 10
    width_pixels = max(2400, column_count * pixels_per_cell)
    height_pixels = max(1800, row_count * pixels_per_cell)
    plt.figure(
        figsize=(width_pixels / HEATMAP_DPI, height_pixels / HEATMAP_DPI),
        dpi=HEATMAP_DPI,
    )
    if score_rows and score_rows[0]:
        numeric_rows = [
            [score if score is not None else np.nan for score in row]
            for row in score_rows
        ]
        image = plt.imshow(numeric_rows, aspect="auto", interpolation="nearest")
        plt.colorbar(image, label="formula diff score")
    else:
        plt.imshow([[0]], aspect="auto", interpolation="nearest")
    plt.title(title)
    plt.xlabel("fine-tuned neuron id")
    plt.ylabel("base neuron id")
    plt.tight_layout()
    plt.savefig(output_path, dpi=HEATMAP_DPI)
    plt.close()


def run_formula_diff_generation(layer: str, tasks: tuple[str, ...]) -> None:
    layer_dir = Path(layer)
    base_dir = layer_dir / "results_base"
    finetuned_dir = layer_dir / "results"
    output_dir = finetuned_dir / "formula_diff"

    for task in tasks:
        write_formula_diff_generation(
            task=task,
            base_csv=base_dir / f"{task}_beam_results.csv",
            finetuned_csv=finetuned_dir / f"{task}_beam_results.csv",
            output_dir=output_dir,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate formula diff CSVs and score heatmaps for one layer."
    )
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--task", action="append", dest="tasks")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tasks = tuple(args.tasks) if args.tasks else DEFAULT_TASKS
    run_formula_diff_generation(args.layer, tasks)
    print(f"Wrote formula diff outputs for {args.layer}: {', '.join(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

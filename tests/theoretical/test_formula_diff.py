import json
from pathlib import Path

import pandas as pd
from theoretical.compositional_explanations.formula_diff import (
    analysis as formula_diff,
)


def test_parse_formula_canonicalizes_commutative_terms():
    left = formula_diff.parse_formula("(tok:B OR tok:A OR (tok:C OR tok:A))")
    right = formula_diff.parse_formula("((tok:A OR tok:B) OR tok:C OR tok:A)")

    assert left.canonical() == right.canonical()


def test_formula_structure_diff_compares_same_neuron_not_nearest_formula():
    base = formula_diff.parse_formula("(tok:A OR tok:B)")
    finetuned = formula_diff.parse_formula("(tok:B OR tok:C)")

    text = formula_diff.formula_diff_text(
        formula_diff.formula_structure_diff(finetuned, base)
    )

    assert text == "\n".join(
        [
            "OR",
            "  = tok:B",
            "  - base tok:A",
            "  + fine-tuned tok:C",
        ]
    )


def test_formula_structure_diff_keeps_nested_shared_formula():
    base = formula_diff.parse_formula(
        "((NOT hypothesis:group) AND (NOT premise:person) "
        "AND (hypothesis:people OR hypothesis:person OR hypothesis:smiling))"
    )
    finetuned = formula_diff.parse_formula(
        "((NOT hypothesis:group) AND (NOT premise:person) "
        "AND (hypothesis:people OR hypothesis:person OR premise:shirt))"
    )

    text = formula_diff.formula_diff_text(
        formula_diff.formula_structure_diff(finetuned, base)
    )

    assert text == "\n".join(
        [
            "AND",
            "  = (NOT hypothesis:group)",
            "  = (NOT premise:person)",
            "  OR",
            "    = hypothesis:people",
            "    = hypothesis:person",
            "    - base hypothesis:smiling",
            "    + fine-tuned premise:shirt",
        ]
    )


def test_formula_diff_score_is_zero_for_identical_formula():
    base = formula_diff.parse_formula("(tok:A OR tok:B)")
    finetuned = formula_diff.parse_formula("(tok:B OR tok:A)")
    diff = formula_diff.formula_structure_diff(finetuned, base)

    assert formula_diff.formula_diff_score(diff) == 0


def test_write_formula_diff_generation_writes_new_per_task_contract(tmp_path: Path):
    base_csv = tmp_path / "base.csv"
    finetuned_csv = tmp_path / "finetuned.csv"
    output_dir = tmp_path / "formula_diff"
    base_csv.write_text(
        "\n".join(
            [
                "neuron,formula,iou",
                "1,(tok:A OR tok:B),0.2",
                "2,(tok:X AND tok:Y),0.9",
            ]
        ),
        encoding="utf-8",
    )
    finetuned_csv.write_text(
        "\n".join(
            [
                "neuron,formula,iou",
                "1,(tok:B OR tok:C),0.4",
                "2,(tok:A OR tok:B),0.8",
            ]
        ),
        encoding="utf-8",
    )

    formula_diff.write_formula_diff_generation(
        task="toy",
        base_csv=base_csv,
        finetuned_csv=finetuned_csv,
        output_dir=output_dir,
    )

    same_rows = (output_dir / "toy_same_neuron_structure_diff.csv").read_text(
        encoding="utf-8"
    )
    closest_buckets = json.loads(
        (output_dir / "closest_finetuned_formula_match_buckets.json").read_text(
            encoding="utf-8"
        )
    )
    matrix_rows = (
        output_dir / "toy_formula_diff_score_matrix.csv"
    ).read_text(encoding="utf-8").splitlines()

    assert "task,neuron,diff_score,base_formula,finetuned_formula,formula_diff" in same_rows
    assert "toy,1," in same_rows
    assert closest_buckets["toy"][0]["neuron_id"] == 1
    assert closest_buckets["toy"][0]["base_formula"] == "(tok:A OR tok:B)"
    assert closest_buckets["toy"][0]["lowest_diff_score"] == 0
    assert closest_buckets["toy"][0]["candidates"] == [
        {
            "neuron_id": 2,
            "formula": "(tok:A OR tok:B)",
            "diff": {"kind": "shared", "formula": "(tok:A OR tok:B)"},
        }
    ]
    assert closest_buckets["toy"][1]["neuron_id"] == 2
    assert closest_buckets["toy"][1]["lowest_diff_score"] == 6
    assert [candidate["neuron_id"] for candidate in closest_buckets["toy"][1]["candidates"]] == [1, 2]
    assert matrix_rows == [
        "base_neuron,finetuned_1,finetuned_2",
        "1,2,0",
        "2,6,6",
    ]
    assert (output_dir / "toy_formula_diff_score_heatmap.png").exists()
    assert not (output_dir / "same_neuron_structure_diff.csv").exists()
    assert not (output_dir / "closest_finetuned_formula_matches.csv").exists()
    assert not (output_dir / "toy_closest_finetuned_formula_matches.csv").exists()


def test_write_formula_diff_generation_keeps_pruned_cells_as_nulls(tmp_path: Path):
    base_csv = tmp_path / "base.csv"
    finetuned_csv = tmp_path / "finetuned.csv"
    output_dir = tmp_path / "formula_diff"
    base_csv.write_text(
        "\n".join(
            [
                "neuron,formula,iou",
                "1,(tok:A OR tok:B),0.2",
                "2,LOW_ACTS_PRUNED,0.0",
            ]
        ),
        encoding="utf-8",
    )
    finetuned_csv.write_text(
        "\n".join(
            [
                "neuron,formula,iou",
                "1,(tok:B OR tok:A),0.4",
                "2,LOW_ACTS_PRUNED,0.0",
            ]
        ),
        encoding="utf-8",
    )

    formula_diff.write_formula_diff_generation(
        task="toy",
        base_csv=base_csv,
        finetuned_csv=finetuned_csv,
        output_dir=output_dir,
    )

    same_rows = pd.read_csv(output_dir / "toy_same_neuron_structure_diff.csv")
    matrix_rows = pd.read_csv(output_dir / "toy_formula_diff_score_matrix.csv")
    closest_buckets = json.loads(
        (output_dir / "closest_finetuned_formula_match_buckets.json").read_text(
            encoding="utf-8"
        )
    )

    assert list(matrix_rows.columns) == [
        "base_neuron",
        "finetuned_1",
        "finetuned_2",
    ]
    assert matrix_rows.loc[0, "finetuned_1"] == 0
    assert pd.isna(matrix_rows.loc[0, "finetuned_2"])
    assert pd.isna(matrix_rows.loc[1, "finetuned_1"])
    assert pd.isna(matrix_rows.loc[1, "finetuned_2"])

    pruned_same_row = same_rows[same_rows["neuron"] == 2].iloc[0]
    assert pd.isna(pruned_same_row["diff_score"])
    assert pruned_same_row["base_formula"] == formula_diff.PRUNED_FORMULA
    assert pruned_same_row["finetuned_formula"] == formula_diff.PRUNED_FORMULA
    assert pd.isna(pruned_same_row["formula_diff"])

    assert closest_buckets["toy"][1] == {
        "neuron_id": 2,
        "base_formula": formula_diff.PRUNED_FORMULA,
        "lowest_diff_score": None,
        "candidates": [],
    }

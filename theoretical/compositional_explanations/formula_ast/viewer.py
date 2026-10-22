"""Build an offline AST explorer from compositional search formulas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import z3

from .analysis import parse_formula


ASSET_DIR = Path(__file__).with_name("viewer_assets")
DEFAULT_CSV = Path(__file__).parent.parent / "data" / "snli_beam_results.csv"


def display_ast(expr: z3.BoolRef) -> dict:
    """Copy supported Z3 operators without simplifying the original tree."""
    if z3.is_true(expr) or z3.is_false(expr):
        return {"type": "constant", "value": z3.is_true(expr)}
    if z3.is_const(expr) and z3.is_bool(expr):
        return {"type": "concept", "name": str(expr.decl().name())}
    operators = {z3.Z3_OP_AND: "and", z3.Z3_OP_OR: "or", z3.Z3_OP_NOT: "not"}
    operator = operators.get(expr.decl().kind())
    if operator is None:
        raise ValueError(f"Unsupported AST operator: {expr.decl().name()}")
    return {"type": operator, "children": [display_ast(child) for child in expr.children()]}


def load_records(csv_path: Path) -> list[dict]:
    records = []
    neuron_ids = set()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"neuron", "formula", "iou"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path}: expected columns neuron, formula, iou")
        for row in reader:
            try:
                neuron_id = int(row["neuron"])
                if neuron_id < 0 or neuron_id in neuron_ids:
                    raise ValueError("neuron IDs must be unique nonnegative integers")
                iou = float(row["iou"])
                if not math.isfinite(iou) or not 0 <= iou <= 1:
                    raise ValueError("IoU must be between 0 and 1")
                ast = display_ast(parse_formula(row["formula"]))
            except Exception as exc:
                raise ValueError(f"{csv_path}, row {reader.line_num}: {exc}") from exc
            neuron_ids.add(neuron_id)
            records.append({"neuronId": neuron_id, "formula": row["formula"], "iou": iou, "ast": ast})
    if not records:
        raise ValueError(f"{csv_path}: no neuron formulas")
    return records


def build_viewer(csv_path: Path, output_path: Path) -> int:
    records = load_records(csv_path)
    payload = {
        "records": records,
        "source": csv_path.name,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    # Escape '<' so imported formula text cannot close the JSON script element.
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")
    page = (ASSET_DIR / "index.html").read_text(encoding="utf-8")
    page = page.replace("__VIEWER_STYLE__", (ASSET_DIR / "viewer.css").read_text(encoding="utf-8"))
    page = page.replace("__VIEWER_SCRIPT__", (ASSET_DIR / "viewer.js").read_text(encoding="utf-8"))
    page = page.replace("__AST_DATA__", data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with neuron, formula and iou columns")
    parser.add_argument("--output", type=Path, default=Path("data/formula_ast_viewer.html"))
    args = parser.parse_args()
    if args.csv.resolve() == args.output.resolve():
        parser.error("output must differ from the source CSV")
    try:
        count = build_viewer(args.csv, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Wrote {count:,} neuron formulas to {args.output.resolve()}")


if __name__ == "__main__":
    main()

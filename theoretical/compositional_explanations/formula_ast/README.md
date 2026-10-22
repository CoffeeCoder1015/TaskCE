# Neuron formula viewer

Build a standalone HTML viewer from the saved neuron formulas. Run from the repository root with the project Python environment.

```sh
python -m theoretical.compositional_explanations.formula_ast.viewer
```

Open `data/formula_ast_viewer.html` directly in a desktop browser. The HTML contains its data, styles and JavaScript, so it needs no server or network connection. To rebuild another CSV or choose the destination:

```sh
python -m theoretical.compositional_explanations.formula_ast.viewer --csv path/to/results.csv --output path/to/viewer.html
```

The CSV must have `neuron`, `formula` and `iou` columns, unique nonnegative integer neuron IDs, and IoU values from 0 to 1. Formulas pass through the current `analysis.parse_formula` converter. Unsupported operators or invalid rows fail the build with row context. The viewer preserves the resulting Z3 tree without simplification and records the source filename and CSV SHA-256 in its source disclosure. Rebuild after changing the CSV.

Concept clicks toggle a shared Boolean assignment. Unlisted concepts are false. Operator clicks inspect their calculation. The other-neuron list evaluates every formula under that same assignment, excludes the displayed neuron, and links to its trees without changing inputs. These are formula outputs, not measured neural model activations or coactivation frequencies across a dataset.

`viewer.py` loads and validates the CSV and bundles the page. `viewer_assets/index.html`, `viewer.css` and `viewer.js` own the interface. Edit those sources and rebuild the HTML. Generated output belongs in the ignored `data/` directory.

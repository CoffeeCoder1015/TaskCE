# Formula AST experiment

Open `notebook.ipynb` with this directory as the working directory and the
project's Python environment as the kernel. Set `csv_path` to an existing
compositional search output (for example, `snli_beam_results.csv`), then run
the cells to load it. No model, dataset download, or GPU is needed.

`load_compexp(path)` returns a pandas DataFrame with the original columns and
row order, plus `ast`. Neuron IDs remain the CSV's IDs, rather than row indices.
The frozen `FormulaNode` exposes `kind`, `value`, and `children`. Atoms retain
their complete feature names; `AND`, `OR`, and `NOT` preserve written nesting
and child order. `True` and `False` are Boolean constants. Empty formulas and
`LOW_ACTS_PRUNED` have `ast=None`; their original formula text distinguishes
them. Malformed formulas raise an error identifying the neuron.

The notebook stops at loading and inspection, leaving the experiment open for
subsequent analysis. Z3 is available as `import z3` through the project's
`z3-solver` dependency; no solver query or AST translation is imposed here.

The AST reconstructs the formula saved in the CSV. It cannot recover the search
ancestry or token IDs that the CSV does not contain. Like the search rendering
and existing formula-diff parser, this grammar treats whitespace and parentheses
as delimiters; feature names containing those characters are not escaped by
the producer and cannot be reconstructed unambiguously.

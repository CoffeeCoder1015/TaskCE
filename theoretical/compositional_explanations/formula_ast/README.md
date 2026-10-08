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

The notebook translates each usable AST with `to_z3` and adds it to one
`z3.Solver`, available as `solver` for subsequent analysis. Every complete
feature name denotes the same Boolean variable across all formulas. Each
formula defines a Boolean named `neuron_<neuron_id>` through equality, so the
neuron is true exactly when its formula is true. Pruned and empty rows
contribute no constraint. IoU and class weights remain metadata and do not
affect the assertions.

The final cell copies the constraints into `enumeration_solver`, records each
satisfying neuron pattern, and excludes that pattern before solving again.
Enumeration stops at `unsat`. An `unknown` result raises an error rather than
reporting partial enumeration as complete. The original `solver` remains
available with its constraints intact.

`activation_patterns` is a Boolean DataFrame whose columns are the CSV neuron
IDs with usable formulas. Each row is a distinct satisfying pattern, regardless
of how many feature assignments produce it. Enumeration has no cutoff and
stores all patterns in memory. With no usable formulas, the projection has one
empty pattern. Re-running the cell starts enumeration from the original
constraints again.

The AST reconstructs the formula saved in the CSV. It cannot recover the search
ancestry or token IDs that the CSV does not contain. Like the search rendering
and existing formula-diff parser, this grammar treats whitespace and parentheses
as delimiters; feature names containing those characters are not escaped by
the producer and cannot be reconstructed unambiguously.

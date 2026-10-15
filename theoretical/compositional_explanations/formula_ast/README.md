# Formula AST experiment

Open `notebook.ipynb` with this directory as the working directory and the project's Python environment as the kernel. Set `csv_path` to an existing compositional search CSV, then run the cells.

The notebook parses each formula into a Z3 Boolean expression and constrains the corresponding `neuron_<id>` variable to equal it. The enumeration cell records each distinct neuron pattern and one satisfying input witness in `activation_patterns`, a DataFrame with `active_neurons` and `inputs` list columns. Enumeration stops at `unsat` or the existing 11-pattern limit. Counts from a stopped run describe only the enumerated patterns. The JSON file `../data/activation_patterns.json` contains the rows as a bare list of objects with `active_neurons` and `inputs` fields.

The matrix cell builds `relationship_matrix`, a symmetric `int64` DataFrame with every original neuron ID on both axes. Each off-diagonal value counts enumerated patterns containing both neurons. The diagonal is zero. The cell writes the full matrix to `../data/neuron_relationships.csv` and displays the submatrix for neurons observed in the enumerated patterns. Re-running the cell rebuilds the matrix from `activation_patterns` before saving, so it does not add counts from an earlier run.

The final cell plots the full relationship matrix as a PNG at 300 DPI, with at most 20 evenly spaced original neuron IDs on each axis. The colorbar shows the number of enumerated patterns containing each pair.

Load the saved matrix with `pd.read_csv("../data/neuron_relationships.csv", index_col="neuron")`. Pandas reads the neuron ID column labels as strings, so convert them with `matrix.columns = matrix.columns.astype(int)` when you need integer column labels. Run the solver setup cell before re-enumerating because enumeration adds blocking clauses to `solver`.

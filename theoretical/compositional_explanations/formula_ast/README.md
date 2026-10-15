# Formula AST experiment

Open `notebook.ipynb` with this directory as the working directory and the project's Python environment as the kernel. Set `csv_path` to an existing compositional search CSV, then run the cells.

The notebook parses each formula into a Z3 Boolean expression and constrains the corresponding `neuron_<id>` variable to equal it. The final cell enumerates distinct neuron patterns, recording each pattern's active original integer neuron IDs and one satisfying input witness. `activation_patterns` is a DataFrame with `active_neurons` and `inputs` list columns.

After enumeration reaches `unsat`, the notebook writes `../data/activation_patterns.json`. The file contains the `neuron_ids` universe once and a `patterns` list with the same rows as the DataFrame. Enumeration has no cutoff. The final cell adds blocking clauses to `solver`; run the solver setup cell again before repeating enumeration.

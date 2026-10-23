# LevelSearch baseline

`algorithm.py` carries LevelSearch and its setup from `../search/`, excluding the
unrelated `Search` function and its `tensor_key` helper. `formula.py` and
`__init__.py` are unchanged copies.
The original API, helpers, process-pool setup, defaults, validation behavior,
prints, comments, and formatting are retained.

The previous vectorized implementation remains in `legacy/` for reference.
The analysis caller and the separate `../search/batched_level.py` prototype are unchanged.

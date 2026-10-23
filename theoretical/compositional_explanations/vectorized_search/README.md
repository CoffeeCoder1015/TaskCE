# Vectorized search plumbing

`search_all(activation_vectors, feature_vectors, device=None, config=None)` moves
the activation matrix and shared concept masks to one device, then calls
`LevelSearch` once. The expected shapes are `activation_vectors[examples,
neurons]` and one boolean concept mask of shape `[examples]` per formula.
`LevelSearch` currently loops over neurons serially and returns one
`(best_formula, best_score)` pair per column; `search_all` adds each column's
index as a `SearchResult`. This is plumbing for the input and output contract,
not the GPU algorithm. The serial body is a mock awaiting the user's PyTorch /
Triton rewrite.

`legacy/` keeps the previous vectorized implementation for reference. The
analysis caller still imports `../search/algorithm.py`; that original search
and the separate `../search/batched_level.py` prototype are unchanged.

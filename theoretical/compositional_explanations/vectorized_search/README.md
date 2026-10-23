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

To profile the existing first-neuron search, run from the repository root:

```sh
python theoretical/compositional_explanations/profile_first_neuron.py --activation <activation.pt> --output-dir <profile-output> --profile
```

The CLI prepares the configured SNLI inputs, then profiles the complete `search_all` call on CUDA, including host-to-device conversion and the final CUDA synchronization. It writes the source hash and Git HEAD with `result.json`, plus HTML, JSON, text, and Speedscope profiles. Install the optional sampler with `uv pip install pyinstrument==5.1.3`; without `--profile`, Pyinstrument is not imported. This is a host wall-clock profile and does not break down individual GPU kernels.

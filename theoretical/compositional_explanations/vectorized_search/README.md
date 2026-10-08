# Vectorized Boolean formula search: construction checkpoint

This is a deliberately incomplete algorithm scaffold, not a runnable search variant. Source is syntax-valid; GPU execution and numerical behavior have not been verified. Missing algorithm parts raise explicit `NotImplementedError` where a coordinator or selector has been introduced. The final working implementation appears only after final assembly.

## Construction so far

1. Packed GPU forward scoring is set up: little-endian packed words, population counts, float32 IoU, atomic [batch, feature] scoring, and composition [batch, parent, retained feature, operation] scoring. The operation order is AND, OR, AND NOT. No beam or search coordinator exists yet; their concepts are introduced when the atomic seed needs them.

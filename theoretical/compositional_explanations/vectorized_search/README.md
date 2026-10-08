# Vectorized Boolean formula search

This package reimplements the Boolean beam-search contract of `new_search`
with a direct batch coordinator and separate packed scoring kernels. It is an
independent implementation reviewed against that contract. The numerical
operations were compared by reading the source; CUDA execution and full runtime
equivalence remain unverified.

`algorithm.py` owns search policy, semantic selection, ancestry, and rendering.
`kernels.py` owns packing and IoU scoring. `search_all` is the public entry point;
`SearchConfig` and `SearchResult` are exported alongside it. Existing search
packages and callers are unchanged.

## Inputs and device ownership

Call `search_all(activation_vectors, feature_vectors, device, config=None)`.
The activation matrix must be a dense CPU `torch.bool` tensor shaped
`[examples, neurons]`. Features must be a list of tuples containing a SymPy
`Symbol` and a one-dimensional NumPy Boolean array of the same example length.
Configuration fields must be positive native Python integers, excluding Boolean
values. The device argument is required and must name a CUDA device, for example
`"cuda:0"`. Validation does not query CUDA availability or initialize CUDA.

A search with neurons requires at least one example and feature. Zero neurons
returns an empty list after validation, before GPU allocation. Results use
activation column indices in ascending order. The producer's separate original
neuron-ID mapping remains the caller's responsibility.

One calling process owns the chosen device. Inputs are packed and transferred
once, then neuron batches run sequentially. There is no worker count or process
pool. Batching bounds intermediate search tensors; it does not bound total input
storage because all packed neuron and feature vectors remain on the device.

## How the search proceeds

The effective beam width is the smaller of the requested width and feature
count. Atomic IoU scores identify each neuron's retained features: only strictly
positive scores survive. The retained tensor width is the largest positive
feature count in the batch, or the effective beam width if larger. Unoccupied
retained slots have negative-infinity scores. Initial atomic ranking uses
`torch.topk`; ties retain its unspecified ordering, including any dependence on
batch-wide retained width.

The batch loop keeps the whole search visible: score atoms, form the initial
beam, record a strictly better winner, score all compositions, select meanings,
record ancestry, and advance using the selected vectors. Each appended atom
increases formula length by one. Best scores persist across levels; equal scores
never replace the earlier winner. Neurons with perfect scores or no occupied
parents do not expand. A neuron with no positive atomic feature returns
`LOW_ACTS_PRUNED` and score `0.0`.

Each occupied parent combines with each retained feature in this fixed order:

| Operation index | Boolean meaning |
| --- | --- |
| 0 | parent AND feature |
| 1 | parent OR feature |
| 2 | parent AND NOT feature |

Positive atomic retention also restricts negated features. For example, if target
`T = {1}` and feature `B = {2}`, `B` has zero atomic IoU and cannot be appended,
even when subtracting `B` from parent `A = {1, 2}` would improve the score.
This restriction is part of the search contract.

Semantic selection ranks candidates by descending score with stable ties.
Equal-score order is parent slot, retained feature slot, then operation index.
The selector excludes the empty vector, **every original atomic meaning**
(including atoms not retained for that neuron), and current parent meanings.
For duplicate candidate meanings, only the earliest ranked candidate survives.
There is no historical set of meanings from earlier levels, so an earlier
composite may reappear once it leaves the current beam.

Selection groups excluded and candidate vectors with `torch.unique`, then maps
candidate meaning IDs into score order. An `amin` reduction finds the first rank
for each meaning. The smallest eligible ranks fill the beam. Only winning vectors
are gathered from the original candidate order. Insufficient unique meanings
leave zero vectors and negative-infinity scores in filler slots. Finite scores
are the occupancy invariant; filler indices are safe to gather but never become
winning formulas.

## Shapes and formula identity

Let `B` be the neuron batch size, `F` the original feature count, `K` the
effective beam width, `R` the batch retained width, and `W = ceil(examples / 32)`.

| Value | Shape | Meaning |
| --- | --- | --- |
| Packed neurons | `[B, W]` | Target Boolean vectors |
| Packed features | `[F, W]` | Original atomic meanings |
| Atomic scores | `[B, F]` | Target/atom IoU |
| Retained feature IDs | `[B, R]` | Indices into original features |
| Beam vectors | `[B, K, W]` | Current formula meanings |
| Composition scores | `[B, K, R, 3]` | Parent/feature/operator scores |
| Candidate selection | `[B, K, W]`, `[B, K]`, `[B, K]` | Vectors, scores, flat origins |
| Formula history | `[B, (maximum_length - 1) * K]` per field | Operations, parent IDs, feature IDs |

Flat origin `((parent_slot * R) + feature_slot) * 3 + operation` identifies one
candidate. The coordinator decodes it to record ancestry without recomposing
Boolean vectors. Atomic formula IDs are original feature indices `0..F-1`.
A selected slot at formula length `L + 1` receives ID
`F + (L - 1) * K + slot`; subtracting `F` addresses its history column.
History is local to the batch and filled in place. Length one allocates no
composite history columns. Only winning ancestry is reconstructed on the CPU.
SymPy builds and simplifies Boolean expressions; the renderer preserves the
established parenthesized `AND`, `OR`, and `NOT` syntax.

## Packed numerical operations

Packing uses `np.packbits(..., bitorder="little")`, zero-pads to a four-byte
boundary, and views contiguous bytes as `int32` words. Kernels use the PTX
`popc.b32` instruction. IoU is the sum of intersection popcounts divided by
`max(sum of union popcounts, 1)`, in float32. Each atomic program owns one
neuron/feature pair. Each composition program owns one neuron/parent/retained
feature triple and explicitly writes its three operator scores.

Loads mask padded words, invalid slots, and inactive neurons. Complementation
occurs only inside `parent AND NOT feature`, so zero padding remains zero.
The final operator axis in the kernel agrees with candidate materialization,
flat origins, and formula reconstruction.

## Review and repeatable static audit

The history constructs one algorithm in six commits. Checkpoints one through
five are deliberately incomplete scaffolds, not working algorithm variants.
Missing pieces raise explicit `NotImplementedError`; syntax checks do not make
those checkpoints runtime-complete. The working implementation appears only
after final assembly.

1. Packed GPU forward scoring is set up: little-endian packed words, population counts, float32 IoU, atomic [batch, feature] scoring, and composition [batch, parent, retained feature, operation] scoring. The operation order is AND, OR, AND NOT. No beam or search coordinator exists yet; their concepts are introduced when the atomic seed needs them.

2. Configuration, result shape, beam state, and formula history are introduced with the atomic seed. Atomic scoring now retains strictly positive-IoU features and seeds beam meanings and IDs. Best-winner defaults, bounded history storage, and neuron-row indices are initialized. The requested beam width is used directly at this checkpoint; expansion is explicitly unfinished.

3. The bounded-depth coordinator now records strict winner improvements, scores compositions, decodes parent/feature/operator origins, appends ancestry, and advances the beam. Semantic selection explicitly raises until the next checkpoint. Winner reconstruction remains unfinished; the coordinator is not a working search variant.

4. Semantic selection now orders scores stably, materializes Boolean compositions, excludes empty meanings, all original atomic meanings, and current parents, groups equal rows, finds each meaning's minimum rank, and ranks eligible representatives. It explicitly raises before filling vacant slots or gathering results.

5. Selection now gathers occupied meanings and makes vacant slots zero vectors with negative-infinity scores and safe origin indices. Effective beam width is capped by atomic feature count. Parents with no finite scores and neurons already at IoU 1 do not expand. Exhausted or insufficient unique meanings remain vacant and cannot become fake winning formulas. Length one skips expansion. Winner reconstruction is still explicitly unfinished.

6. Winning ancestry is reconstructed and rendered on the CPU, with LOW_ACTS_PRUNED / 0.0 for no positive atomic winner. The CUDA device entry point, sequential batch dispatch, public input guards, zero-neuron handling, and exports complete the assembly.

Positive atomic retention, semantic exclusions, ancestry, beam occupancy, and
the IoU upper bound are algorithmic rules. Public type, shape, configuration,
and CUDA device guards are assembled with the entry point in the final commit.
The selector ranks meaning IDs and gathers winners without a second full
candidate-vector allocation for score ranking.

The final source places formula history beside reconstruction, and beam state beside growth. Each scoring
kernel sits beside its host launch. These existing concepts provide precise
ways to reason about meanings, ancestry, and the scoring matrix; colocation
reduces jumps between related definitions without adding abstractions or
changing algorithm rules. Semantic selection returns a tuple of selected
vectors, scores, and flat origins directly. The transitional `CandidateSelection`
record is removed at final assembly because it merely wrapped that one return;
`Beam` and `FormulaHistory` retain persistent state and their invariants.

From the repository root, run:

```powershell
python theoretical/compositional_explanations/vectorized_search/audit_sources.py
git diff --check
```

The standard-library audit parses and compiles package source in memory, reports
function/type inventory, operator constants, package call relationships, and
whether dependencies are discoverable. It imports no search or GPU module and
runs no search algorithm. It does not compare AST equality with the earlier
implementation or claim semantic equivalence from syntax checks.

No tests were added, edited, or run. No algorithm was executed and no GPU
dependencies were installed during this rewrite. The remaining verification gap
is CUDA kernel compilation and observed end-to-end numerical/search behavior.

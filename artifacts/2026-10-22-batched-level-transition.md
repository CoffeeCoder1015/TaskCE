# Batched level search with a bounded priority queue

[Implementations and full caller](C:/Users/heroi/Code/Workspace/AIEALab/TaskCE/theoretical/compositional_explanations/search/batched_level.py)

Each neuron has its own formula history and bounded priority queue. GPU batches contain parent/concept pairs from several neurons. CPU code constructs formulas and decides which improving formulas enter those queues. Existing production callers remain unchanged.

The entry point accepts Boolean `neurons[N, E]` and `(formula, mask[E])` feature pairs on the same device.

```python
results = batched_level_search(
    neurons, feature_vectors,
    SearchConfig(beam_size=10, iou_calculation_batch_size=4096),
)
```

This usage has not been executed.

## What the queue guarantees

`Candidate` binds a formula, its actual Boolean device mask, and its raw IoU. There are no formula IDs or vector-bank coordinates to reconstruct.

`BoundedPriorityQueue.push_many` keeps the best K records offered since the last drain. Higher raw IoU wins. Ties favor earlier offers. It selects the winners on the CPU, then stacks only their masks into compact device storage. After the call, the queue retains no views into the offered temporary candidate batches. `drain` empties the queue and returns complete records in ranked order.

This is a CPU-ranked queue with GPU-backed mask storage when the inputs are on CUDA. It is not a GPU heap. SymPy formula history already consumes admissible scores on the CPU, so GPU ranking would add coordination without removing that work. Argon and the independent design review favored this arrangement. No throughput comparison has been run.

For a fixed ranking, `bestK(bestK(A) + B) = bestK(A + B)`. This is why bounded insertion can replace the unbounded child list and final pruning. Formula history still sees every improving formula before selection. A later improvement can create another record for the same formula, as in LevelSearch.

The queue copies up to K masks for each participating neuron in each chunk, including retained winners from earlier chunks. This is an explicit cost. Tensor contents must not be mutated concurrently with insertion. The ownership contract uses the ordinary current device stream and does not promise immutable tensors or coordination across streams.

## What the numerical tools mean

`SetPartition(left, right)` computes three disjoint regions of matching Boolean batches.

```python
shared = left & right
left_only = left & ~right
right_only = right & ~left
```

Those regions cover the union. They know nothing about search policy. The caller expresses the old vector exclusions as nonempty-region checks.

| Operation | Result mask | Required nonempty regions |
| --- | --- | --- |
| AND | `shared` | All three |
| OR | The union | `left_only`, `right_only` |
| AND NOT | `left_only` | `shared`, `left_only` |

AND NOT also requires different formulas. This check remains on the CPU because arbitrary input pairs need not associate equal formulas with equal masks.

`intersection_over_union(masks, targets)` reduces only the last axis. Other axes broadcast through PyTorch. Empty unions score zero. The function contains no admission rule or host transfer.

## One complete transition

`advance_level` drains every active queue and visits its parents. It saves strict improvements in the penalized best score before applying the parent symbol-count bound. Lazy Cartesian products enumerate each neuron's expandable parents and retained concepts. Round-robin interleaving gives the numerical chunks work from multiple neurons while preserving each neuron's subsequence.

Each chunk has operands `[pairs, examples]`, result masks `[pairs, 3, examples]`, and scores `[pairs, 3]`. The caller states the region checks, materializes the three results in bulk, and copies the score table to the CPU. Rejected operations have score -1. A valid zero remains distinct.

The CPU constructs formulas in AND, OR, AND NOT order. Every strict penalized improvement updates formula history before queue selection. Accepted complete records enter their owning neuron's queue in one bulk offer per chunk. At the next level, only those queues' children become parents.

`batched_level_search` also batches atomic IoU across neurons. It retains positive atoms in stable descending IoU order and seeds history with all of them, including atoms outside the beam. It then calls the complete transition up to the iteration bound.

## Preserved rules and declared differences

The implementation keeps positive atomic filtering, local vector exclusions, SymPy formula construction, formula-keyed history, strict penalized improvement, raw-IoU beam ranking, and the separate saved best. It keeps the parent symbol-count expansion bound. Children generated on the final iteration do not update the saved result until visited, matching LevelSearch's timing.

Parent traversal is now explicitly ranked. LevelSearch iterates its heap's internal array, which is not a fully ranked sequence. This difference can change which tied explanation wins or which formula-history entry is encountered first. Empty feature sets return blank explanations instead of failing during concatenation. Exact output parity is not claimed.

Duplicate seed formulas record their maximum raw IoU. Argon identified that LevelSearch's descending seed loop instead lets the lowest duplicate overwrite history. Keeping the maximum makes later strict-improvement checks compare against the strongest seed. Feature records themselves remain separate.

This remains a dense Boolean implementation. It introduces no Triton kernel, process pool, or global truth-vector deduplication. Stacking operands and targets, materializing Boolean temporaries, CPU SymPy work, and repeated queue compaction are visible costs that require profiling before optimization.

## Verification limits

Review is static only. Source inspection and in-memory syntax compilation do not establish runtime correctness or GPU throughput. No algorithm execution, tests, CUDA compilation, or benchmarks were run. The HTML is unchanged.

| Component | Function | Behavior to verify | Proposed method for an authorized run |
| --- | --- | --- | --- |
| Queue | `push_many`, `drain` | Stable best K across chunk boundaries, zero capacity, and independent mask storage. | Concrete offers with tied scores, followed by input mutation and device-memory inspection. |
| Set calculation | `SetPartition`, `intersection_over_union` | Disjoint regions, broadcast shapes, and empty union. | Small Boolean examples on CPU and CUDA. |
| Whole search | `batched_level_search`, `advance_level` | Independent neuron histories, admission rules, penalties, and saved-result timing. | Compare per-level states with LevelSearch, accounting for declared parent-order differences. Existing `tests/test_search.py` covers only the legacy path. |

"""Score packed sample masks by counting intersection and union bits."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _batched_iou_kernel(
    neuron,
    vectors,
    scores,
    word_count,
    BLOCK_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    words = tl.arange(0, BLOCK_WIDTH)
    valid_words = words < word_count
    neuron_words = tl.load(neuron + words, mask=valid_words, other=0)
    candidate_words = tl.load(
        vectors + row * word_count + words, mask=valid_words, other=0
    )
    intersection = tl.sum(libdevice.popc(neuron_words & candidate_words), axis=0)
    union = tl.sum(libdevice.popc(neuron_words | candidate_words), axis=0)
    # Round division like eager float32 IoU to preserve score ties.
    score = tl.div_rn(intersection.to(tl.float32), tl.maximum(union, 1).to(tl.float32))
    tl.store(scores + row, score)


def batched_iou(neuron: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Score contiguous int32 [word] against [row, word]; return float32 [row]."""
    row_count, word_count = vectors.shape
    scores = torch.empty(row_count, dtype=torch.float32, device=vectors.device)
    _batched_iou_kernel[(row_count,)](
        neuron,
        vectors,
        scores,
        word_count,
        BLOCK_WIDTH=triton.next_power_of_2(word_count),
        num_warps=4,
    )
    return scores

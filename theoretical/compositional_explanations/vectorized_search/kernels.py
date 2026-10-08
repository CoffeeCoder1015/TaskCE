"""Packed Boolean scoring; composition axis order is AND, OR, AND NOT."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def popcount32(words):
    return tl.inline_asm_elementwise(
        asm="popc.b32 $0, $1;",
        constraints="=r,r",
        args=[words],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def packed_iou(target_words, candidate_words):
    intersection = tl.sum(popcount32(target_words & candidate_words), axis=0)
    union = tl.sum(popcount32(target_words | candidate_words), axis=0)
    return intersection.to(tl.float32) / tl.maximum(union, 1).to(tl.float32)


@triton.jit
def score_atomic_kernel(
    neurons,
    features,
    scores,
    feature_count,
    word_count,
    BLOCK_WIDTH: tl.constexpr,
):
    neuron = tl.program_id(0)
    feature = tl.program_id(1)
    words = tl.arange(0, BLOCK_WIDTH)
    within_vector = words < word_count
    target_words = tl.load(
        neurons + neuron * word_count + words,
        mask=within_vector,
        other=0,
    )
    feature_words = tl.load(
        features + feature * word_count + words,
        mask=within_vector,
        other=0,
    )
    tl.store(
        scores + neuron * feature_count + feature,
        packed_iou(target_words, feature_words),
    )


@triton.jit
def score_composition_kernel(
    neurons,
    parents,
    features,
    retained_features,
    parent_valid,
    retained_valid,
    active,
    scores,
    beam_size,
    retained_count,
    word_count,
    BLOCK_WIDTH: tl.constexpr,
):
    neuron = tl.program_id(0)
    parent_slot = tl.program_id(1)
    feature_slot = tl.program_id(2)
    parent_offset = neuron * beam_size + parent_slot
    retained_offset = neuron * retained_count + feature_slot
    eligible = (
        tl.load(active + neuron)
        & tl.load(parent_valid + parent_offset)
        & tl.load(retained_valid + retained_offset)
    )
    feature = tl.load(retained_features + retained_offset, mask=eligible, other=0)
    words = tl.arange(0, BLOCK_WIDTH)
    readable = (words < word_count) & eligible
    target_words = tl.load(
        neurons + neuron * word_count + words,
        mask=readable,
        other=0,
    )
    parent_words = tl.load(
        parents + parent_offset * word_count + words,
        mask=readable,
        other=0,
    )
    feature_words = tl.load(
        features + feature * word_count + words,
        mask=readable,
        other=0,
    )
    # Each program owns three adjacent scores for one parent/feature pair.
    score_offset = (parent_offset * retained_count + feature_slot) * 3
    tl.store(
        scores + score_offset,
        tl.where(
            eligible,
            packed_iou(target_words, parent_words & feature_words),
            -float("inf"),
        ),
    )
    tl.store(
        scores + score_offset + 1,
        tl.where(
            eligible,
            packed_iou(target_words, parent_words | feature_words),
            -float("inf"),
        ),
    )
    tl.store(
        scores + score_offset + 2,
        tl.where(
            eligible,
            packed_iou(target_words, parent_words & ~feature_words),
            -float("inf"),
        ),
    )


def pack_vectors(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    """Pack rows into little-bit-order int32 words, with zero tail padding."""
    packed_bytes = np.packbits(vectors, axis=1, bitorder="little")
    padding_bytes = -packed_bytes.shape[1] % 4
    if padding_bytes:
        packed_bytes = np.pad(packed_bytes, ((0, 0), (0, padding_bytes)))
    packed_words = np.ascontiguousarray(packed_bytes).view(np.int32)
    return torch.from_numpy(packed_words).to(device=device)


def score_atoms(batch_neurons: torch.Tensor, packed_features: torch.Tensor) -> torch.Tensor:
    """Score [batch, features]; a program handles one neuron/feature pair."""
    batch_size, word_count = batch_neurons.shape
    feature_count = packed_features.shape[0]
    scores = torch.empty(
        (batch_size, feature_count), dtype=torch.float32, device=batch_neurons.device
    )
    score_atomic_kernel[batch_size, feature_count](
        batch_neurons,
        packed_features,
        scores,
        feature_count,
        word_count,
        BLOCK_WIDTH=triton.next_power_of_2(word_count),
        num_warps=4,
    )
    return scores


def score_compositions(
    batch_neurons: torch.Tensor,
    parent_vectors: torch.Tensor,
    packed_features: torch.Tensor,
    retained_features: torch.Tensor,
    parent_valid: torch.Tensor,
    retained_valid: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Score [batch, parent, retained feature, 3] in AND/OR/AND NOT order.

    Invalid slots and inactive neurons receive negative infinity. Complements
    occur only inside parent AND NOT feature, preserving zero tail padding.
    """
    batch_size, beam_size, word_count = parent_vectors.shape
    retained_count = retained_features.shape[1]
    scores = torch.empty(
        (batch_size, beam_size, retained_count, 3),
        dtype=torch.float32,
        device=batch_neurons.device,
    )
    score_composition_kernel[batch_size, beam_size, retained_count](
        batch_neurons,
        parent_vectors,
        packed_features,
        retained_features,
        parent_valid,
        retained_valid,
        active,
        scores,
        beam_size,
        retained_count,
        word_count,
        BLOCK_WIDTH=triton.next_power_of_2(word_count),
        num_warps=4,
    )
    return scores

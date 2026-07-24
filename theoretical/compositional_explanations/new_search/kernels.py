"""Packed Boolean IoU kernels for compositional explanation search."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _popcount(values):
    return tl.inline_asm_elementwise(
        asm="popc.b32 $0, $1;",
        constraints="=r,r",
        args=[values],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _iou(target, candidate):
    intersection = tl.sum(
        _popcount(target & candidate),
        axis=0,
    )
    union = tl.sum(
        _popcount(target | candidate),
        axis=0,
    )
    return intersection.to(tl.float32) / tl.maximum(
        union,
        1,
    ).to(tl.float32)


@triton.jit
def _atomic_iou_kernel(
    neurons,
    features,
    scores,
    feature_count,
    packed_width,
    BLOCK_WIDTH: tl.constexpr,
):
    neuron_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    word_offsets = tl.arange(0, BLOCK_WIDTH)
    word_valid = word_offsets < packed_width

    neuron = tl.load(
        neurons + neuron_id * packed_width + word_offsets,
        mask=word_valid,
        other=0,
    )
    feature = tl.load(
        features + feature_id * packed_width + word_offsets,
        mask=word_valid,
        other=0,
    )
    tl.store(
        scores + neuron_id * feature_count + feature_id,
        _iou(neuron, feature),
    )


@triton.jit
def _composition_iou_kernel(
    neurons,
    parents,
    features,
    retained_ids,
    parent_valid,
    retained_valid,
    active,
    scores,
    beam_size,
    retained_count,
    packed_width,
    BLOCK_WIDTH: tl.constexpr,
):
    neuron_id = tl.program_id(0)
    parent_slot = tl.program_id(1)
    retained_slot = tl.program_id(2)

    retained_offset = (
        neuron_id * retained_count
        + retained_slot
    )
    candidate_valid = (
        tl.load(active + neuron_id)
        & tl.load(
            parent_valid + neuron_id * beam_size + parent_slot
        )
        & tl.load(retained_valid + retained_offset)
    )
    feature_id = tl.load(
        retained_ids + retained_offset,
        mask=candidate_valid,
        other=0,
    )

    word_offsets = tl.arange(0, BLOCK_WIDTH)
    word_valid = (
        word_offsets < packed_width
    ) & candidate_valid
    neuron = tl.load(
        neurons + neuron_id * packed_width + word_offsets,
        mask=word_valid,
        other=0,
    )
    parent = tl.load(
        parents
        + (
            neuron_id * beam_size
            + parent_slot
        ) * packed_width
        + word_offsets,
        mask=word_valid,
        other=0,
    )
    feature = tl.load(
        features + feature_id * packed_width + word_offsets,
        mask=word_valid,
        other=0,
    )

    score_offset = (
        (
            neuron_id * beam_size
            + parent_slot
        ) * retained_count
        + retained_slot
    ) * 3
    invalid_score = -float("inf")
    tl.store(
        scores + score_offset,
        tl.where(
            candidate_valid,
            _iou(neuron, parent & feature),
            invalid_score,
        ),
    )
    tl.store(
        scores + score_offset + 1,
        tl.where(
            candidate_valid,
            _iou(neuron, parent | feature),
            invalid_score,
        ),
    )
    tl.store(
        scores + score_offset + 2,
        tl.where(
            candidate_valid,
            _iou(neuron, parent & ~feature),
            invalid_score,
        ),
    )


def pack_vectors(
    vectors: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    packed_bytes = np.packbits(
        vectors,
        axis=1,
        bitorder="little",
    )
    byte_padding = -packed_bytes.shape[1] % 4
    if byte_padding:
        packed_bytes = np.pad(
            packed_bytes,
            ((0, 0), (0, byte_padding)),
        )
    packed_words = np.ascontiguousarray(
        packed_bytes,
    ).view(np.int32)
    return torch.from_numpy(packed_words).to(
        device=device,
    )


def atomic_iou_scores(
    neurons: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    neuron_count, packed_width = neurons.shape
    feature_count = features.shape[0]
    scores = torch.empty(
        (neuron_count, feature_count),
        dtype=torch.float32,
        device=neurons.device,
    )
    _atomic_iou_kernel[
        neuron_count,
        feature_count,
    ](
        neurons,
        features,
        scores,
        feature_count,
        packed_width,
        BLOCK_WIDTH=triton.next_power_of_2(packed_width),
        num_warps=4,
    )
    return scores


def composition_iou_scores(
    neurons: torch.Tensor,
    parents: torch.Tensor,
    features: torch.Tensor,
    retained_ids: torch.Tensor,
    parent_valid: torch.Tensor,
    retained_valid: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    neuron_count, beam_size, packed_width = parents.shape
    retained_count = retained_ids.shape[1]
    scores = torch.empty(
        (
            neuron_count,
            beam_size,
            retained_count,
            3,
        ),
        dtype=torch.float32,
        device=neurons.device,
    )
    _composition_iou_kernel[
        neuron_count,
        beam_size,
        retained_count,
    ](
        neurons,
        parents,
        features,
        retained_ids,
        parent_valid,
        retained_valid,
        active,
        scores,
        beam_size,
        retained_count,
        packed_width,
        BLOCK_WIDTH=triton.next_power_of_2(packed_width),
        num_warps=4,
    )
    return scores

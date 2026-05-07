import pytest
import torch

from capture.capturer import (
    latest_checkpoint,
    verify_padded_batch_shape,
)


def test_latest_checkpoint_uses_numeric_checkpoint_suffix(tmp_path):
    task_dir = tmp_path / "snli"
    task_dir.mkdir()
    checkpoint_9 = task_dir / "checkpoint-9"
    checkpoint_10 = task_dir / "checkpoint-10"
    checkpoint_9.mkdir()
    checkpoint_10.mkdir()

    assert latest_checkpoint([str(checkpoint_9), str(checkpoint_10)]) == str(checkpoint_10)


def test_latest_checkpoint_falls_back_to_lexicographic_for_unknown_names(tmp_path):
    task_dir = tmp_path / "snli"
    task_dir.mkdir()
    first = task_dir / "alpha"
    second = task_dir / "beta"
    first.mkdir()
    second.mkdir()

    assert latest_checkpoint([str(second), str(first)]) == str(second)


def test_verify_padded_batch_shape_accepts_matching_batch_size():
    tokenized = {"input_ids": torch.zeros((2, 5), dtype=torch.long)}
    verify_padded_batch_shape(tokenized, [[1, 2, 3], [4, 5]])


def test_verify_padded_batch_shape_rejects_joined_batch():
    tokenized = {"input_ids": torch.zeros((1, 8), dtype=torch.long)}
    with pytest.raises(ValueError, match="batch size"):
        verify_padded_batch_shape(tokenized, [[1, 2, 3], [4, 5]])

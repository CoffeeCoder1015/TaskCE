import pytest
import torch

from capture.capturer import (
    latest_checkpoint,
    resolve_capture_layer,
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


class FakeLayer:
    pass


class FakeModel:
    def __init__(self, module_names):
        self.modules = [(name, FakeLayer()) for name in module_names]

    def named_modules(self):
        yield "", self
        yield from self.modules


def test_resolve_capture_layer_accepts_exact_module_path():
    model = FakeModel(["model.layers.8.feed_forward"])

    module, path = resolve_capture_layer(model, "model.layers.8.feed_forward")

    assert module is model.modules[0][1]
    assert path == "model.layers.8.feed_forward"


def test_resolve_capture_layer_accepts_peft_prefixed_module_path():
    model = FakeModel(["base_model.model.model.layers.8.feed_forward"])

    module, path = resolve_capture_layer(model, "model.layers.8.feed_forward")

    assert module is model.modules[0][1]
    assert path == "base_model.model.model.layers.8.feed_forward"


def test_resolve_capture_layer_rejects_ambiguous_suffix_matches():
    model = FakeModel(
        [
            "left.model.layers.8.feed_forward",
            "right.model.layers.8.feed_forward",
        ]
    )

    with pytest.raises(KeyError, match="matched multiple module paths"):
        resolve_capture_layer(model, "model.layers.8.feed_forward")

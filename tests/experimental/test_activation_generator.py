import pytest

torch = pytest.importorskip("torch")

import experimental.model as model_module
from experimental.generators.activations import (
    generate_final_token_activations,
)
from experimental.model import CaptureIdentity, WrappedModel


class FakeTokenizer:
    def pad(self, encoded, *, padding, return_tensors):
        assert padding is True
        assert return_tensors == "pt"
        rows = encoded["input_ids"]
        width = max(len(row) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row + [0] * (width - len(row)) for row in rows],
                dtype=torch.long,
            )
        }


class CaptureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.first = torch.nn.Identity()
        self.second = torch.nn.Identity()
        self.forward_calls = 0

    def forward(self, input_ids):
        self.forward_calls += 1
        states = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        return self.second(self.first(states))


def capture_identity():
    return CaptureIdentity(
        prefix=("creator", "model", "base"),
        dataset="snli",
    )


def test_final_token_generator_hooks_all_layers_in_one_dataset_pass(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(model_module, "DATA_DIRECTORY", tmp_path)
    raw_model = CaptureModel()
    wrapped = WrappedModel(raw_model)

    saved_paths = generate_final_token_activations(
        model=wrapped,
        tokenizer=FakeTokenizer(),
        data={"input_ids": [[1, 2], [3, 4], [5, 6]]},
        layers=("first", "second"),
        identity=capture_identity(),
        batch_size=2,
    )

    expected = torch.tensor([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    assert raw_model.forward_calls == 2
    assert set(saved_paths) == {"first", "second"}
    assert torch.equal(wrapped.captures["first"], expected)
    assert torch.equal(wrapped.captures["second"], expected)
    assert torch.equal(
        torch.load(saved_paths["first"], weights_only=False),
        expected,
    )
    assert not raw_model.first._forward_hooks
    assert not raw_model.second._forward_hooks

import torch

from capture.capturer import CapturedResults
from capture.saving import (
    load_captured_activations,
    save_captured_activations,
)


def test_save_and_load_captured_activations_round_trips_capture_output_shape(tmp_path):
    snli_base_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    snli_finetuned_states = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    claim_base_states = torch.tensor([[9.0, 10.0]])
    captured_results = {
        "snli": {
            "base": CapturedResults(
                states=snli_base_states,
                labels=["entailment", "neutral"],
                layer="model.layers.1",
            ),
            "finetuned": CapturedResults(
                states=snli_finetuned_states,
                labels=["entailment", "neutral"],
                layer="model.layers.1",
            ),
        },
        "claim": {
            "base": CapturedResults(
                states=claim_base_states,
                labels=["supports"],
                layer="model.layers.1",
            ),
        },
    }

    saved_path = save_captured_activations(captured_results, tmp_path)
    loaded_results = load_captured_activations(saved_path)

    assert saved_path == tmp_path / "captured_activations" / "captured_results.pt"
    assert set(loaded_results) == {"snli", "claim"}
    assert set(loaded_results["snli"]) == {"base", "finetuned"}
    assert set(loaded_results["claim"]) == {"base"}
    assert isinstance(loaded_results["snli"]["base"], CapturedResults)
    assert torch.equal(loaded_results["snli"]["base"].states, snli_base_states)
    assert loaded_results["snli"]["base"].labels == ["entailment", "neutral"]
    assert loaded_results["snli"]["base"].layer == "model.layers.1"
    assert torch.equal(loaded_results["snli"]["finetuned"].states, snli_finetuned_states)
    assert loaded_results["snli"]["finetuned"].labels == ["entailment", "neutral"]
    assert torch.equal(loaded_results["claim"]["base"].states, claim_base_states)
    assert loaded_results["claim"]["base"].labels == ["supports"]


def test_load_captured_activations_accepts_output_dir(tmp_path):
    captured_results = {
        "snli": {
            "base": CapturedResults(
                states=torch.tensor([[1.0, 2.0]]),
                labels=["entailment"],
                layer="model.layers.1",
            ),
        },
    }

    save_captured_activations(captured_results, tmp_path)
    loaded_results = load_captured_activations(tmp_path)

    assert torch.equal(
        loaded_results["snli"]["base"].states,
        captured_results["snli"]["base"].states,
    )

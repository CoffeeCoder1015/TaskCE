"""Procedural generators for raw layer activations."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from experimental.model import CaptureIdentity, WrappedModel


def append_final_token(current, _module, _inputs, output):
    """Append one final-token activation tensor for the current inference batch."""
    if current is None:
        current = []
    current.append(
        output[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
    )
    return current


def padded_inputs(model: WrappedModel, tokenizer, input_ids):
    """Pad one prepared batch and move it to the wrapped model's input device."""
    inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt",
    )
    input_device = next(model.model.parameters()).device
    return {
        name: value.to(input_device)
        for name, value in inputs.items()
    }


def generate_final_token_activations(
    *,
    model: WrappedModel,
    tokenizer: Any,
    data: dict[str, Any],
    layers: Sequence[str],
    identity: CaptureIdentity,
    batch_size: int = 32,
) -> dict[str, Path]:
    """Capture one final-token vector per example for every requested layer."""
    resolved_layers = []
    try:
        resolved_layers = [
            model.hook(layer, append_final_token)
            for layer in layers
        ]
        model.print_hook_report()

        input_ids = data["input_ids"]
        for start in range(0, len(input_ids), batch_size):
            inputs = padded_inputs(
                model,
                tokenizer,
                input_ids[start:start + batch_size],
            )
            with torch.inference_mode():
                model.model(**inputs)
    finally:
        model.remove_hooks()

    for layer in resolved_layers:
        model.captures[layer] = torch.cat(model.captures[layer], dim=0)

    return model.save(identity)

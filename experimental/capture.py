from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from typing import Any

import torch

from hook_recorder import HookRecorder


@contextmanager
def capture(
    model: torch.nn.Module,
    dataset: Any,
    *,
    tokenizer: Any,
    layers: list[str],
    batch_size: int = 32,
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[Iterator[tuple[list[int], dict[str, list[torch.Tensor]]]]]:
    """Yield an iterator of example indices and layer outputs, one batch at a time."""
    input_ids = dataset["input_ids"]
    device = next(model.parameters()).device
    model.eval()
    with HookRecorder(model, layers) as recorder:
        def batches():
            for indices, inputs in prepared_batches(input_ids, tokenizer, batch_size):
                with torch.no_grad():
                    model(**{name: tensor.to(device) for name, tensor in inputs.items()})
                if on_progress is not None:
                    on_progress(indices[-1] + 1)
                yield indices, recorder.take_outputs()

        with closing(batches()) as stream:
            yield stream


def prepared_batches(
    input_ids: Any, tokenizer: Any, batch_size: int
) -> Iterator[tuple[list[int], dict[str, torch.Tensor]]]:
    """Pad pretokenized rows in order, retaining their dataset positions."""
    for start in range(0, len(input_ids), batch_size):
        stop = min(start + batch_size, len(input_ids))
        inputs = tokenizer.pad(
            {"input_ids": input_ids[start:stop]},
            padding=True,
            return_tensors="pt",
        )
        yield list(range(start, stop)), inputs

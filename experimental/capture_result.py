from __future__ import annotations

import torch


class CaptureResult(dict[tuple[str, int], list[torch.Tensor]]):
    """In-memory layer outputs indexed by (layer, example)."""

    def write(
        self,
        example_indices: list[int],
        outputs: dict[str, list[torch.Tensor]],
    ) -> None:
        """Index the recorder's CPU tensors by example without copying storage."""
        for layer, invocations in outputs.items():
            for position, index in enumerate(example_indices):
                key = (layer, index)
                self[key] = [tensor[position] for tensor in invocations]

from __future__ import annotations

from types import TracebackType
from typing import Self

import torch
from torch.utils.hooks import RemovableHandle


class HookRecorder:
    _modules: dict[str, torch.nn.Module]
    _handles: dict[str, RemovableHandle]
    _outputs: dict[str, list[torch.Tensor]]

    def __init__(self, model: torch.nn.Module, layers: list[str]) -> None:
        self._modules = {}
        self._handles = {}
        self._outputs = {}


        modules = dict(model.named_modules())

        for layer in layers:
            if layer in modules:
                module = modules[layer]
                self._modules[layer] = module
                continue

            suffix_matches = [
                (path, module)
                for path, module in modules.items()
                if path.endswith(f".{layer}")
            ]
            if len(suffix_matches) == 1:
                resolved_layer = suffix_matches[0][0]
                module = suffix_matches[0][1]
                self._modules[resolved_layer] = module
                continue

            # Error
            if suffix_matches:
                raise KeyError(
                    f"Layer {layer!r} matched multiple module paths: "
                    f"{[path for path, _module in suffix_matches]}"
                )
            raise KeyError(layer)

    def get_hook(self,resolved_layer):
        def record_output(module, inputs, output) -> None:
            self._outputs.setdefault(resolved_layer,[])
            self._outputs[resolved_layer].append(output.detach().to(device="cpu", dtype=torch.float32))

        return record_output

    def __enter__(self) -> Self:
        try:
            for layer, module in self._modules.items():
                hook_function = self.get_hook(layer)
                hook = module.register_forward_hook(hook_function)
                self._handles[layer] = hook
        except Exception:
            self.remove_hooks()
            raise

        return self

    def take_outputs(self) -> dict[str, list[torch.Tensor]]:
        current_outputs = self._outputs
        self._outputs = {layer: [] for layer in self._modules}
        return current_outputs

    def remove_hooks(self) -> None:
        for handle in self._handles.values():
            handle.remove()
        self._handles.clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.remove_hooks()

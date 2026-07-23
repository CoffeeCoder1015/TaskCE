"""Instrumentation for an already-constructed model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import torch


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def validate_path_component(value: str, field: str) -> None:
    """Reject values that could change the capture directory structure."""
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    if not value or value in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty directory name.")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be one relative path component.")
    if Path(value).is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"{field} must be one relative path component.")


@dataclass(frozen=True)
class CaptureIdentity:
    """Describe the caller-defined path prefix and dataset for one capture."""

    prefix: tuple[str, ...]
    dataset: str

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, tuple):
            raise TypeError("Capture identity prefix must be a tuple of strings.")
        if not self.prefix:
            raise ValueError("Capture identity prefix cannot be empty.")
        for index, component in enumerate(self.prefix):
            validate_path_component(component, f"Capture identity prefix[{index}]")
        validate_path_component(self.dataset, "Capture identity dataset")


class WrappedModel:
    """Own a model's active hooks, inference, and layer captures."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.hooked_layers: set[str] = set()
        self.hook_handles: dict[str, Any] = {}
        self.captures: dict[str, Any] = {}

    def resolve_layer(self, layer: str) -> tuple[str, torch.nn.Module]:
        """Resolve an exact module path or one unambiguous path suffix."""
        if not isinstance(layer, str) or not layer:
            raise TypeError("A layer must be identified by a non-empty string path.")

        try:
            return layer, self.model.get_submodule(layer)
        except AttributeError:
            pass

        suffix_matches = [
            path
            for path, _module in self.model.named_modules()
            if path and path.endswith(f".{layer}")
        ]
        if len(suffix_matches) == 1:
            resolved_path = suffix_matches[0]
            return resolved_path, self.model.get_submodule(resolved_path)
        if suffix_matches:
            raise KeyError(
                f"Layer {layer!r} matched multiple module paths: {suffix_matches}"
            )
        raise KeyError(layer)

    def hook(
        self,
        layer: str,
        capture: Callable[[Any, torch.nn.Module, tuple[Any, ...], Any], Any],
    ) -> str:
        """Update a layer's capture whenever the resolved module executes."""
        resolved_path, module = self.resolve_layer(layer)
        if resolved_path in self.hook_handles:
            raise ValueError(f"Layer {resolved_path!r} is already hooked.")

        def record_output(module, inputs, output) -> None:
            current = self.captures.get(resolved_path)
            self.captures[resolved_path] = capture(
                current,
                module,
                inputs,
                output,
            )

        handle = module.register_forward_hook(record_output)
        self.hook_handles[resolved_path] = handle
        self.hooked_layers.add(resolved_path)
        return resolved_path

    def save(self, identity: CaptureIdentity) -> dict[str, Path]:
        """Save each layer's capture without interpreting its structure."""
        saved_paths = {}
        for layer, capture in self.captures.items():
            output_path = DATA_DIRECTORY.joinpath(
                *identity.prefix,
                layer,
                f"{identity.dataset}_activation.pt",
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(capture, output_path)
            saved_paths[layer] = output_path

        return saved_paths

    def clear_captures(self) -> None:
        """Discard captured values while leaving installed hooks active."""
        self.captures.clear()

    def remove_hooks(self) -> None:
        """Remove every installed hook while preserving existing captures."""
        for handle in self.hook_handles.values():
            handle.remove()
        self.hook_handles.clear()
        self.hooked_layers.clear()

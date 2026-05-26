"""Activation capture utilities for TaskCE experiments.

The capture package runs model forward passes over configured datasets and
records layer activations for base and fine-tuned checkpoints. It also provides
lightweight postprocessing helpers for turning raw activations into binary
neuron-feature matrices used by downstream search and analysis steps.
"""
from .captureConfig import CaptureConfig
from .capturer import Capture, CapturedResults
from .saving import (
    load_captured_activations,
    save_captured_activations,
)

__all__ = [
    "Capture",
    "CaptureConfig",
    "CapturedResults",
    "load_captured_activations",
    "save_captured_activations",
]

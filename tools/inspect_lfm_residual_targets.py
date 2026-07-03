"""Inspect candidate residual-stream capture targets for LFM-style models.

This script is intentionally laptop-safe by default: it only downloads/loads
model config unless --load-model is passed, and it only runs a forward pass when
--probe-forward is passed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "block_ff_dim",
    "block_auto_adjust_ff_dim",
    "block_ffn_dim_multiplier",
    "vocab_size",
    "max_position_embeddings",
    "layer_types",
)


@dataclass(frozen=True)
class ModuleRecord:
    path: str
    class_name: str
    parameter_shape: str | None = None
    trainable: bool | None = None


@dataclass(frozen=True)
class LayerContainerRecord:
    path: str
    count: int
    layer_class: str


def shape_string(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    if len(shape) == 0:
        return "scalar"
    return "x".join(str(dim) for dim in shape)


def config_record(config: Any) -> dict[str, Any]:
    return {
        field: getattr(config, field, None)
        for field in CONFIG_FIELDS
        if hasattr(config, field)
    }


def dtype_from_name(dtype_name: str):
    dtypes = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[dtype_name]


def tensor_summary(value: Any) -> dict[str, Any]:
    return {
        "type": type(value).__name__,
        "shape": list(getattr(value, "shape", ())),
        "dtype": str(getattr(value, "dtype", "")),
        "device": str(getattr(value, "device", "")),
    }


def output_summary(value: Any) -> Any:
    """Return a JSON-safe summary of hook input/output structure."""
    if isinstance(value, torch.Tensor):
        return tensor_summary(value)
    if isinstance(value, tuple):
        return [output_summary(item) for item in value]
    if isinstance(value, list):
        return [output_summary(item) for item in value]
    if isinstance(value, dict):
        return {str(key): output_summary(item) for key, item in value.items()}
    if hasattr(value, "to_tuple"):
        return output_summary(value.to_tuple())
    return {"type": type(value).__name__}


def find_layer_containers(model: Any) -> list[LayerContainerRecord]:
    records = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and name.endswith("layers"):
            first_layer = module[0] if len(module) else None
            records.append(
                LayerContainerRecord(
                    path=name,
                    count=len(module),
                    layer_class=type(first_layer).__name__ if first_layer else "",
                )
            )
    return records


def direct_parameter_record(path: str, module: Any) -> ModuleRecord:
    params = list(module.parameters(recurse=False))
    if not params:
        return ModuleRecord(path=path, class_name=type(module).__name__)
    return ModuleRecord(
        path=path,
        class_name=type(module).__name__,
        parameter_shape=shape_string(params[0]),
        trainable=bool(params[0].requires_grad),
    )


def layer_child_records(layer_path: str, layer: Any) -> list[ModuleRecord]:
    records = []
    for child_name, child_module in layer.named_children():
        child_path = f"{layer_path}.{child_name}"
        records.append(direct_parameter_record(child_path, child_module))
        for sub_name, sub_module in child_module.named_children():
            records.append(direct_parameter_record(f"{child_path}.{sub_name}", sub_module))
    return records


def candidate_target_paths(container: LayerContainerRecord, layer_index: int) -> list[str]:
    layer_path = f"{container.path}.{layer_index}"
    return [
        layer_path,
        f"{layer_path}.operator_norm",
        f"{layer_path}.input_layernorm",
        f"{layer_path}.conv",
        f"{layer_path}.self_attn",
        f"{layer_path}.ffn_norm",
        f"{layer_path}.feed_forward",
    ]


def load_model(model_id: str, dtype_name: str, device_map: str, lora_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_from_name(dtype_name),
        device_map=device_map,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def inspect_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer = load_model(
        args.model_id,
        args.dtype,
        args.device_map,
        args.lora_path,
    )
    try:
        containers = find_layer_containers(model)
        selected_container = containers[0] if containers else None
        selected_layer = None
        child_records: list[ModuleRecord] = []
        candidates: list[str] = []

        if selected_container is not None:
            modules = dict(model.named_modules())
            layer_index = resolve_layer_index(args.layer_index, selected_container.count)
            selected_layer_path = f"{selected_container.path}.{layer_index}"
            selected_layer = selected_layer_path
            child_records = layer_child_records(selected_layer_path, modules[selected_layer_path])
            candidates = [
                path
                for path in candidate_target_paths(selected_container, layer_index)
                if path in modules
            ]

        probe = {}
        if args.probe_forward:
            probe = probe_targets(model, tokenizer, candidates, args.prompt)

        return {
            "model_id": args.model_id,
            "lora_path": args.lora_path,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "config": config_record(model.config),
            "layer_containers": [asdict(record) for record in containers],
            "selected_layer": selected_layer,
            "selected_layer_children": [asdict(record) for record in child_records],
            "candidate_capture_targets": candidates,
            "probe": probe,
        }
    finally:
        del model


def resolve_layer_index(layer_index: int, layer_count: int) -> int:
    if layer_count == 0:
        raise ValueError("cannot resolve a layer index for an empty layer container")
    resolved = layer_index if layer_index >= 0 else layer_count + layer_index
    if resolved < 0 or resolved >= layer_count:
        raise ValueError(
            f"layer index {layer_index} resolves to {resolved}, outside 0..{layer_count - 1}"
        )
    return resolved


def probe_targets(
    model: Any,
    tokenizer: Any,
    target_paths: list[str],
    prompt: str,
) -> dict[str, Any]:
    modules = dict(model.named_modules())
    captured: dict[str, Any] = {}
    handles = []

    def make_hook(path: str):
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            captured[path] = {
                "input": output_summary(inputs),
                "output": output_summary(output),
            }

        return hook

    for path in target_paths:
        handles.append(modules[path].register_forward_hook(make_hook(path)))

    try:
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_device = next(model.parameters()).device
        tokenized = {key: value.to(input_device) for key, value in tokenized.items()}
        with torch.inference_mode():
            model(**tokenized)
    finally:
        for handle in handles:
            handle.remove()

    return captured


def config_only(args: argparse.Namespace) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(args.model_id)
    return {
        "model_id": args.model_id,
        "config": config_record(config),
        "note": (
            "Pass --load-model to inspect module paths; pass --probe-forward "
            "to run one hook probe."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect LFM residual-stream hook targets without running the full "
            "TaskCE pipeline."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Load model weights and list candidate module paths.",
    )
    parser.add_argument(
        "--probe-forward",
        action="store_true",
        help="Run one prompt through hooks on the candidate target paths.",
    )
    parser.add_argument("--layer-index", type=int, default=-2)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument(
        "--prompt",
        default="Premise: A person is running.\nHypothesis: Someone is moving.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a concise text report.",
    )
    return parser.parse_args()


def print_text_report(record: dict[str, Any]) -> None:
    print(f"Model: {record['model_id']}")
    if record.get("lora_path"):
        print(f"LoRA: {record['lora_path']}")
    print("\nConfig:")
    for key, value in record["config"].items():
        print(f"  {key}: {value}")

    if "layer_containers" not in record:
        print(f"\n{record['note']}")
        return

    print("\nLayer containers:")
    for container in record["layer_containers"]:
        print(
            f"  {container['path']}: {container['count']} layers "
            f"({container['layer_class']})"
        )

    print(f"\nSelected layer: {record['selected_layer']}")
    print("\nSelected layer children:")
    for child in record["selected_layer_children"]:
        suffix = ""
        if child["parameter_shape"] is not None:
            state = "trainable" if child["trainable"] else "frozen"
            suffix = f" shape={child['parameter_shape']} {state}"
        print(f"  {child['path']}: {child['class_name']}{suffix}")

    print("\nCandidate capture targets:")
    for path in record["candidate_capture_targets"]:
        print(f"  {path}")

    if record["probe"]:
        print("\nProbe output summaries:")
        print(json.dumps(record["probe"], indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.probe_forward and not args.load_model:
        raise SystemExit("--probe-forward requires --load-model")

    record = inspect_model(args) if args.load_model else config_only(args)
    if args.json:
        print(json.dumps(record, indent=2, sort_keys=True))
    else:
        print_text_report(record)


if __name__ == "__main__":
    main()

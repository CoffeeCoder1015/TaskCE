import gc
from dataclasses import dataclass
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from capture.captureConfig import CaptureConfig
from capture.lora_checkpoints import latest_checkpoint, latest_task_lora_checkpoints

@dataclass
class CapturedResults:
    states: torch.Tensor
    labels: Any
    layer: str

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        state_summary = (
            f"Tensor(shape={tuple(self.states.shape)}, "
            f"dtype={self.states.dtype}, device={self.states.device})"
        )
        return (
            f"{type(self).__name__}("
            f"states={state_summary}, "
            f"labels={len(self.labels)} labels, "
            f"layer={self.layer!r})"
        )


def resolve_capture_layer(model, layer):
    # Return the layer to be captured on the model either by name or by index
    named_layers = [
        (name, module)
        for name, module in model.named_modules()
        if name
    ]
    layer_by_name = dict(named_layers)

    if isinstance(layer, str):
        resolved_path = resolve_capture_layer_path(layer_by_name, layer)
        resolved_module = layer_by_name[resolved_path]
    else:
        resolved_path, resolved_module = named_layers[layer]

    return resolved_module, resolved_path


def resolve_capture_layer_path(layer_by_name, layer):
    if layer in layer_by_name:
        return layer

    suffix_matches = [
        name
        for name in layer_by_name
        if name.endswith(f".{layer}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise KeyError(
            f"Capture layer {layer!r} matched multiple module paths: "
            f"{suffix_matches}"
        )

    raise KeyError(layer)


def verify_padded_batch_shape(tokenized, batch_input_ids):
    actual_batch_size = tokenized["input_ids"].shape[0]
    expected_batch_size = len(batch_input_ids)
    if actual_batch_size != expected_batch_size:
        raise ValueError(
            "Padded tokenizer output changed the capture batch size: "
            f"expected {expected_batch_size}, got {actual_batch_size}."
        )


def capture_task_activations(model_id, tokenizer, task, layer, lora_path=None, batch_size=32):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    print(
        f"{task.name}: formatting={task.need_formatting}, "
        f"tokenizing={task.need_tokenizing}"
    )
    formatted_dataset = (
        task.dataset.map(task.data_formatter)
        if task.need_formatting
        else task.dataset
    )
    tokenized_dataset = (
        formatted_dataset.map(
            lambda example: {
                "input_ids": tokenizer.apply_chat_template(
                    example["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                )
            }
        )
        if task.need_tokenizing
        else formatted_dataset
    )

    labels = formatted_dataset[task.label_field]
    input_ids = tokenized_dataset["input_ids"]

    states = []
    activations = {}
    capture_layer, resolved_layer_path = resolve_capture_layer(model, layer)

    def capture_hook(module, inputs, output):
        activations[resolved_layer_path] = output.detach()

    hook_handle = capture_layer.register_forward_hook(capture_hook)

    try:
        print(f"\nStarting {task.name} activation capture.")
        for i in tqdm(
            range(0, len(input_ids), batch_size),
            desc=f"Capturing {task.name}",
        ):
            batch_input_ids = input_ids[i:i + batch_size]

            # Pad only the current batch so each forward has a rectangular tensor.
            tokenized = tokenizer.pad(
                {"input_ids": batch_input_ids},
                padding=True,
                return_tensors="pt",
            )
            verify_padded_batch_shape(tokenized, batch_input_ids)

            # With device_map="auto", putting inputs on the first parameter device is the
            # most direct way to let Accelerate dispatch the rest through model hooks.
            input_device = next(model.parameters()).device
            tokenized = {
                key: value.to(input_device)
                for key, value in tokenized.items()
            }

            activations.clear()
            with torch.inference_mode():
                model(**tokenized)

            # PCA wants one vector per example. Left padding plus add_generation_prompt
            # makes the final sequence position the comparable prompt-final position.
            batch_states = activations[resolved_layer_path][:, -1, :]
            states.append(batch_states.to("cpu", dtype=torch.float32))

        print(f"{task.name} activation capture finished.")

        if states:
            states = torch.cat(states, dim=0)
        else:
            states = torch.empty((0, 0), dtype=torch.float32)

        return CapturedResults(
            states=states,
            labels=labels,
            layer=resolved_layer_path,
        )
    finally:
        # The hook must always be removed so future captures do not append stale values.
        hook_handle.remove()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def Capture(
    model_id: str,
    lora_dir: str | None,
    tasks: list[CaptureConfig],
    layer: int | str,
    batch_size: int = 32,
    lora_remote: bool = False,
    lora_token: str | bool | None = None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    results = {}

    print("[Base model activation capture]")
    for task in tasks:
        results[task.name] = {
            "base": capture_task_activations(
                model_id=model_id,
                tokenizer=tokenizer,
                task=task,
                layer=layer,
                batch_size=batch_size,
            )
        }

    # lora_dir=None gives a base-only capture path for quick PCA experiments.
    if lora_dir is None:
        return results

    print("[Finetuned model activation capture]")
    latest_checkpoints = latest_task_lora_checkpoints(
        lora_dir,
        remote=lora_remote,
        token=lora_token,
    )

    for task in tasks:
        if task.name not in latest_checkpoints:
            print(f"Skipping {task.name}: no LoRA checkpoint found.")
            continue

        results[task.name]["finetuned"] = capture_task_activations(
            model_id=model_id,
            tokenizer=tokenizer,
            task=task,
            layer=layer,
            lora_path=latest_checkpoints[task.name],
            batch_size=batch_size,
        )

    return results

import gc
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from capture.captureConfig import CaptureConfig


def resolve_capture_layer(model, layer):
    # Use the model's own named modules as the source of truth.
    # The empty name is the whole model, so skip it and expose every real named module.
    named_layers = [
        (name, module)
        for name, module in model.named_modules()
        if name
    ]
    named_modules = dict(named_layers)

    # A manual module path should mean exactly what the caller wrote.
    if isinstance(layer, str):
        if layer in named_modules:
            print(f"Capturing manual layer path: {layer}")
            return named_modules[layer], layer, None

        candidate_paths = [name for name, module in named_layers][:50]
        raise ValueError(
            f"Layer path '{layer}' was not found. Candidate paths include: {candidate_paths}"
        )

    if not named_layers:
        raise ValueError("The model does not expose any named modules to hook.")

    # None means the default requested behavior: second-to-last named module.
    if layer is None:
        layer_index = len(named_layers) - 2
    else:
        layer_index = layer

    if not isinstance(layer_index, int):
        raise TypeError("layer must be None, an int layer index, or a string module path.")

    if layer_index < 0 or layer_index >= len(named_layers):
        raise IndexError(
            f"Layer index {layer_index} is out of range for the model's "
            f"{len(named_layers)} named modules."
        )

    resolved_path, resolved_module = named_layers[layer_index]
    print(
        f"Capturing named module index {layer_index}; resolved path: {resolved_path}"
    )
    return resolved_module, resolved_path, layer_index


def capture_task_activations(model_id, tokenizer, task, layer, lora_path=None, batch_size=32):
    # This mirrors evaluation model loading, but we do not create a generation pipeline.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()

    capture_layer, resolved_layer_path, resolved_layer_index = resolve_capture_layer(model, layer)
    formatted_dataset = task.dataset.map(task.data_formatter)
    prompts = formatted_dataset["prompt"]

    # The formatter should tag each example with the correct downstream label.
    # The existing repo formatters write this to "answer", and CaptureConfig keeps that default.
    if task.label_field not in formatted_dataset.column_names:
        raise ValueError(
            f"Formatted dataset for task '{task.name}' does not contain "
            f"label field '{task.label_field}'."
        )
    labels = formatted_dataset[task.label_field]

    states = []
    captured_batch_vectors = []

    def capture_hook(module, inputs, output):
        # Transformer blocks often return either a tensor or a tuple with the tensor first.
        # The analysis target is one vector per prompt, so slice the final token immediately.
        activation = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(activation):
            raise TypeError(
                f"Captured output from {resolved_layer_path} is not a tensor: "
                f"{type(activation)}"
            )
        captured_batch_vectors.clear()
        captured_batch_vectors.append(
            activation.detach()[:, -1, :].to("cpu", dtype=torch.float32)
        )

    hook_handle = capture_layer.register_forward_hook(capture_hook)

    try:
        print(f"\nStarting {task.name} activation capture.")
        for i in tqdm(
            range(0, len(prompts), batch_size),
            desc=f"Capturing {task.name}",
        ):
            batch_prompts = prompts[i:i + batch_size]

            # Match the reference script: chat-template the prompt and ask for model inputs.
            tokenized = tokenizer.apply_chat_template(
                batch_prompts,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )

            # With device_map="auto", putting inputs on the first parameter device is the
            # most direct way to let Accelerate dispatch the rest through model hooks.
            input_device = next(model.parameters()).device
            tokenized = {
                key: value.to(input_device)
                for key, value in tokenized.items()
            }

            captured_batch_vectors.clear()
            with torch.inference_mode():
                model(**tokenized)

            if not captured_batch_vectors:
                raise RuntimeError(
                    f"The hook for {resolved_layer_path} did not capture an activation."
                )

            # PCA wants one vector per example. Left padding plus add_generation_prompt
            # makes the final sequence position the comparable prompt-final position.
            states.append(captured_batch_vectors[0])

        print(f"{task.name} activation capture finished.")

        if states:
            states = torch.cat(states, dim=0)
        else:
            states = torch.empty((0, 0), dtype=torch.float32)

        return {
            "states": states,
            "labels": labels,
            "prompts": prompts,
            "layer": resolved_layer_path,
            "layer_index": resolved_layer_index,
            "model_id": model_id,
            "lora_path": lora_path,
        }
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
    layer: int | str | None = None,
    batch_size: int = 32,
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
    task_paths = {
        path: os.path.join(lora_dir, path)
        for path in os.listdir(lora_dir)
        if os.path.isdir(os.path.join(lora_dir, path))
    }
    task_and_checkpts = {
        task_name: [
            os.path.join(task_path, path)
            for path in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, path))
        ]
        for task_name, task_path in task_paths.items()
    }

    # Keep the eval convention: use the lexicographically latest checkpoint per task.
    latest_checkpoints = {
        task_name: sorted(checkpoints)[-1]
        for task_name, checkpoints in task_and_checkpts.items()
        if checkpoints
    }

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

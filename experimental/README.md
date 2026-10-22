# Activation capture

`main.py` invokes the new capture API and separately saves final-token matrices.

- `main.py` captures the base model and latest SNLI LoRA checkpoint, then saves each variant separately.
- `capture.py` contains the capture loop and its padding and inference helpers.
- `prepare_snli.py` loads SNLI, formats prompts, and returns the tokenized dataset.
- `prepare_lora.py` downloads the adapter snapshot and selects the latest checkpoint per task.
- `hook_recorder.py` defines `HookRecorder`, which owns hooks and buffered layer outputs.
- `capture_result.py` defines `CaptureResult`, an in-memory dictionary with a batch-writing helper.

## Running capture

Run `python experimental/main.py` from the repository root using the project environment. It requires PyTorch, Transformers, PEFT, Datasets, and Accelerate, plus access to the adapter repository. The entry point defaults to 128 SNLI validation examples and two feed-forward layers. It captures the base LFM model and the latest SNLI adapter from `LORA_REPO`, loading a fresh model for each. Edit `MODEL_ID`, `LORA_REPO`, `DATASET_SPLIT`, and `LAYERS` in `main.py` for your run.

```python
from capture import capture

with capture(
    model,
    {"input_ids": tokenized_examples},
    tokenizer=tokenizer,
    layers=["model.layers.8.feed_forward"],
    batch_size=32,
) as result:
    outputs = result["model.layers.8.feed_forward", 17]
```

Input examples are already tokenized. The tokenizer controls padding, and example IDs are input row positions. Selected modules emit tensors with a leading batch dimension. Each list contains one tensor per invocation, including every token and existing padding. Supply exact module paths, as listed in `main.py`. Hooks are installed once, and all selected layers share each forward pass. Capture runs in evaluation mode without gradient tracking and leaves the model in evaluation mode.

Entering `with capture(...) as result` completes inference and removes the hooks before the caller's block begins. All captured tensors stay in CPU memory. The result is an ordinary dictionary and remains usable after the block.

`main.py` selects the final token only during saving. This aggregation uses left-padded inputs and one sequence-by-neuron output per example. It saves float32 matrices under `data/<model-id>/base/<layer>/snli_activation.pt` and `data/<model-id>/snli/<checkpoint>/<layer>/snli_activation.pt`. Adapter layer names include the PEFT wrapper prefix. General capture does not impose this final-token policy or float32 conversion.

## Results

```python
outputs = result["model.layers.8", 17]
outputs[0].relu_()
```

Each key is `(layer, example_index)`. Each value is a list of tensors, one per module invocation for that example, preserving padding and dtype. `write` indexes the recorder's CPU tensors using row views. Reading, assigning, and editing values use ordinary dictionary and tensor behavior without extra copies. Saving is explicit.

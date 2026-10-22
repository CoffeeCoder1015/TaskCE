# Activation capture

`main.py` invokes the new capture API and separately saves final-token matrices.

- `main.py` captures the base model and latest SNLI LoRA checkpoint, then saves each variant separately.
- `capture.py` contains the capture loop and its padding and inference helpers.
- `prepare_snli.py` loads SNLI, formats prompts, and returns the tokenized dataset.
- `prepare_lora.py` downloads the adapter snapshot and selects the latest checkpoint per task.
- `hook_recorder.py` defines `HookRecorder`, which owns hooks and buffered layer outputs.

## Running capture

Run `python experimental/main.py` from the repository root using the project environment. It requires PyTorch, Transformers, PEFT, Datasets, Accelerate, and Rich, plus access to the adapter repository. The entry point uses the full SNLI validation split (10,000 examples), capturing `model.embedding_norm` immediately before `lm_head` and the feed-forward outputs of layers 14 down to 8. It captures the base LFM model and the latest SNLI adapter from `LORA_REPO`, loading a fresh model for each. Edit `MODEL_ID`, `LORA_REPO`, `DATASET_SPLIT`, and `LAYERS` in `main.py` for your run.

The Rich terminal display updates one inference-progress row per model. Counts advance after a whole batch is recorded across all selected layers. Once both runs are saved, it prints the model architecture tree once, with captured modules in green.

```python
from capture import capture

with capture(
    model,
    {"input_ids": tokenized_examples},
    tokenizer=tokenizer,
    layers=["model.layers.8.feed_forward"],
    batch_size=32,
) as batches:
    for indices, outputs in batches:
        for layer, invocations in outputs.items():
            final_tokens = invocations[0][:, -1].clone()
            # Process or save these final tokens before advancing to the next batch.
```

Input examples are already tokenized. The tokenizer controls padding, and example IDs are input row positions. Selected modules emit tensors with a leading batch dimension. Each list contains one tensor per invocation, including every token and existing padding. Supply exact module paths, as listed in `main.py`. Hooks are installed once, and all selected layers share each forward pass. Capture runs in evaluation mode without gradient tracking and leaves the model in evaluation mode.

Entering `with capture(...) as batches` installs the hooks. Each iteration runs one forward pass and yields `(indices, outputs)` for that batch. Leaving the context closes the iterator and removes the hooks, including on early exit or failure. Capture does not accumulate batches; the caller owns any outputs it retains.

`on_progress`, when supplied, receives the cumulative number of examples recorded after each batch. The capture code does not render terminal output.

`main.py` copies the final-token slice from each batch, releasing full-token outputs before advancing inference. It concatenates those smaller chunks in dataset order and saves float32 matrices under `data/<model-id>/base/<layer>/snli_activation.pt` and `data/<model-id>/snli/<checkpoint>/<layer>/snli_activation.pt`. This uses left-padded inputs and one module invocation per example. Adapter layer names include the PEFT wrapper prefix. General capture does not impose this final-token policy.

## Batch outputs

```python
indices                                      # Dataset row positions for this batch.
outputs["model.layers.8.feed_forward"][0]      # First invocation: [batch, tokens, neurons].
```

`outputs` is a dictionary keyed by resolved module path. Each value is a list of float32 CPU tensors, one per module invocation, preserving every token and padding. Rows correspond to `indices`. Keeping a batch or a view into it keeps its tensor storage alive. Copy reduced slices when retaining them, as in `main.py`. Saving is explicit.

"""Run the SNLI compositional analysis for every captured layer."""

from analysis import DATA_DIRECTORY, activation_path, run
from vectorized_search.algorithm import search_all


# Configuration
model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
task_name = "snli"
dataset_name = "snli"
checkpoint_name = None
layers = [
    "model.embedding_norm",
    "model.layers.14.feed_forward",
    "model.layers.13.feed_forward",
    "model.layers.12.feed_forward",
    "model.layers.11.feed_forward",
    "model.layers.10.feed_forward",
    "model.layers.9.feed_forward",
    "model.layers.8.feed_forward",
]

for layer in layers:
    finetuned_path = activation_path(
        model_id,
        dataset_name,
        layer,
        task_name=task_name,
        checkpoint_name=checkpoint_name,
    )
    output_path = DATA_DIRECTORY / layer / "snli_beam_results.csv"
    print(f"Running {layer} from {finetuned_path}", flush=True)
    result_path = run(
        task_name,
        finetuned_path,
        search_fn=search_all,
        output_path=output_path,
    )
    print(f"Saved results to {result_path}", flush=True)

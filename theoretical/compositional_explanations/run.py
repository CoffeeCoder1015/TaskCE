from analysis import DATA_DIRECTORY, activation_path, run
from vectorized_search.algorithm import search_all

# Configuration
model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
task_name = "snli"
dataset_name = "snli"
layer = "model.layers.8.feed_forward"
checkpoint_name = None
finetuned_path = activation_path(
    model_id,
    dataset_name,
    layer,
    task_name=task_name,
    checkpoint_name=checkpoint_name,
)
output_path = DATA_DIRECTORY / f"{task_name}_beam_results.csv"
search_fn = search_all

result_path = run(
    task_name,
    finetuned_path,
    search_fn=search_fn,
    output_path=output_path,
)

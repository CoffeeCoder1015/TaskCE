from analysis import (
    activation_path,
    load_activation,
    load_dataset,
)
from construction.tokenizer.orchestration import (
    get_tokenizer,
)
from construction.vectors import (
    construct_feature_vectors,
)
from postprocessing import (
    prune_min_acts,
    threshold,
)
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

activations = load_activation(finetuned_path)
dataset = load_dataset("stanfordnlp/snli",split="validation")

feature_columns = ("premise", "hypothesis")
feature_tokenizer = get_tokenizer(
    "spacy-pos-snli-features",
    dataset,
    feature_columns,
    enable_pos=False,
)
features = construct_feature_vectors(
    dataset,
    feature_tokenizer,
    feature_columns,
)

binary_activations = threshold(activations, alpha=0.055)
searchable_activations = prune_min_acts(
    binary_activations,
    min_acts=500,
)
search_results = search_all(
    searchable_activations.matrix,
    features,
)

for result in search_results:
    print(result)
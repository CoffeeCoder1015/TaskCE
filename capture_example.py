from collections import Counter

from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from capture import Capture, CaptureConfig


SNLI_LABELS = ["entailment", "neutral", "contradiction"]

SNLI_SYSTEM_PROMPT = {
    "role": "system",
    "content": """Determine the relationship between the `premise`and `hypothesis` and respond with an answer.
You must respond with an answer of `entailment`, `neutral` or `contradiction`""",
}

def format_snli_for_capture(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["prompt"] = [
        SNLI_SYSTEM_PROMPT,
        {"role": "user", "content": test_example},
    ]

    example["answer"] = SNLI_LABELS[example["label"]]
    return example


def plot_pca(capture_result, output_path, title):
    states = capture_result["states"]
    labels = capture_result["labels"]

    print("Captured states shape:", tuple(states.shape))
    print("Captured label counts:", Counter(labels))
    print("Captured layer:", capture_result["layer"])

    # PCA expects a 2D array shaped like [examples, activation_dimension].
    # The capture package already returns CPU float32 tensors, so NumPy conversion is direct.
    states_for_pca = states.numpy()
    labels_for_plot = list(labels)

    # Reduce the activation vectors to two coordinates so the label groups can be visualized.
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(states_for_pca)

    print("PCA output shape:", pca_points.shape)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    # Plot each label separately so the legend and colors line up with the task labels.
    for label in SNLI_LABELS:
        x_values = [
            pca_points[i, 0]
            for i, example_label in enumerate(labels_for_plot)
            if example_label == label
        ]
        y_values = [
            pca_points[i, 1]
            for i, example_label in enumerate(labels_for_plot)
            if example_label == label
        ]
        plt.scatter(x_values, y_values, label=label, alpha=0.7, s=18)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


snli_examples = load_dataset("snli", split="validation")

capture_results = Capture(
    model_id="LiquidAI/LFM2.5-1.2B-Thinking",
    lora_dir="../multitune/output", # Assumed to be used in conjunction with https://github.com/CoffeeCoder1015/multitune
    tasks=[
        CaptureConfig(
            name="snli",
            dataset=snli_examples,
            data_formatter=format_snli_for_capture,
        )
    ],
    layer=-2,
    batch_size=256,
)

plot_pca(
    capture_results["snli"]["base"],
    "snli_activation_pca_base.png",
    "SNLI base activation PCA",
)

if "finetuned" in capture_results["snli"]:
    plot_pca(
        capture_results["snli"]["finetuned"],
        "snli_activation_pca_finetuned.png",
        "SNLI finetuned activation PCA",
    )

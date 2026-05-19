from pathlib import Path

import pandas as pd


NEURON_SEARCH_RESULT_COLUMNS = [
    "neuron",
    "formula",
    "iou",
    "weight_ent",
    "weight_neut",
    "weight_contr",
]
PRUNED_NEURON_FORMULA = "LOW_ACTS_PRUNED"


def build_neuron_search_results_dataframe(
    search_results,
    kept_neuron_ids,
    total_neuron_count,
    classification_weights,
):
    kept_neuron_ids = neuron_id_list(kept_neuron_ids)
    kept_neuron_id_set = set(kept_neuron_ids)

    records = []
    for result in search_results:
        neuron_id = kept_neuron_ids[result.activation_index]
        records.append(
            neuron_search_result_record(
                neuron_id=neuron_id,
                formula=result.best_formula,
                iou=result.best_score,
                classification_weights=classification_weights,
            )
        )

    for neuron_id in range(int(total_neuron_count)):
        if neuron_id in kept_neuron_id_set:
            continue
        records.append(
            neuron_search_result_record(
                neuron_id=neuron_id,
                formula=PRUNED_NEURON_FORMULA,
                iou=0.0,
                classification_weights=classification_weights,
            )
        )

    return pd.DataFrame(
        records,
        columns=NEURON_SEARCH_RESULT_COLUMNS,
    ).sort_values("iou", ascending=False, ignore_index=True)


def save_neuron_search_results_csv(dataframe, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)
    return dataframe


def neuron_search_result_record(neuron_id, formula, iou, classification_weights):
    weights = classification_weights[int(neuron_id)]
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    weight_ent, weight_neut, weight_contr = weights

    return {
        "neuron": int(neuron_id),
        "formula": formula,
        "iou": float(iou),
        "weight_ent": float(weight_ent),
        "weight_neut": float(weight_neut),
        "weight_contr": float(weight_contr),
    }


def neuron_id_list(neuron_ids):
    if hasattr(neuron_ids, "detach"):
        neuron_ids = neuron_ids.detach().cpu()
    if hasattr(neuron_ids, "tolist"):
        neuron_ids = neuron_ids.tolist()
    return [int(neuron_id) for neuron_id in neuron_ids]

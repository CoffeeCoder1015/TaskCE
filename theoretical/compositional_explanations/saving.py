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
    weight_column_names=None,
):
    weight_column_names = list(weight_column_names or NEURON_SEARCH_RESULT_COLUMNS[3:])
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
                weight_column_names=weight_column_names,
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
                weight_column_names=weight_column_names,
            )
        )

    return pd.DataFrame(
        records,
        columns=["neuron", "formula", "iou", *weight_column_names],
    ).sort_values("iou", ascending=False, ignore_index=True)


def save_neuron_search_results_csv(dataframe, output_csv_path):
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_csv_path, index=False)
    return dataframe


def neuron_search_result_record(
    neuron_id,
    formula,
    iou,
    classification_weights,
    weight_column_names=None,
):
    weight_column_names = list(weight_column_names or NEURON_SEARCH_RESULT_COLUMNS[3:])
    weights = classification_weights[int(neuron_id)]
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    if len(weights) != len(weight_column_names):
        raise ValueError(
            "classification weight count does not match weight column count: "
            f"{len(weights)} != {len(weight_column_names)}"
        )

    record = {
        "neuron": int(neuron_id),
        "formula": formula,
        "iou": float(iou),
    }
    record.update(
        {
            column_name: float(weight)
            for column_name, weight in zip(weight_column_names, weights, strict=True)
        }
    )
    return record


def neuron_id_list(neuron_ids):
    if hasattr(neuron_ids, "detach"):
        neuron_ids = neuron_ids.detach().cpu()
    if hasattr(neuron_ids, "tolist"):
        neuron_ids = neuron_ids.tolist()
    return [int(neuron_id) for neuron_id in neuron_ids]

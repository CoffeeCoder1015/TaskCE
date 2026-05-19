from types import SimpleNamespace

import pandas as pd

from analysis import build_neuron_search_results_dataframe, save_neuron_search_results_csv


def test_build_neuron_search_results_dataframe_includes_found_and_pruned_neurons():
    classification_weights = [
        [0.10, 0.20, 0.30],
        [1.10, 1.20, 1.30],
        [2.10, 2.20, 2.30],
        [3.10, 3.20, 3.30],
    ]
    search_results = [
        SimpleNamespace(activation_index=1, best_formula="tok:B", best_score=0.4),
        SimpleNamespace(activation_index=0, best_formula="tok:A", best_score=0.9),
    ]

    dataframe = build_neuron_search_results_dataframe(
        search_results=search_results,
        kept_neuron_ids=[2, 0],
        total_neuron_count=4,
        classification_weights=classification_weights,
    )

    assert dataframe.to_dict("records") == [
        {
            "neuron": 2,
            "formula": "tok:A",
            "iou": 0.9,
            "weight_ent": 2.1,
            "weight_neut": 2.2,
            "weight_contr": 2.3,
        },
        {
            "neuron": 0,
            "formula": "tok:B",
            "iou": 0.4,
            "weight_ent": 0.1,
            "weight_neut": 0.2,
            "weight_contr": 0.3,
        },
        {
            "neuron": 1,
            "formula": "LOW_ACTS_PRUNED",
            "iou": 0.0,
            "weight_ent": 1.1,
            "weight_neut": 1.2,
            "weight_contr": 1.3,
        },
        {
            "neuron": 3,
            "formula": "LOW_ACTS_PRUNED",
            "iou": 0.0,
            "weight_ent": 3.1,
            "weight_neut": 3.2,
            "weight_contr": 3.3,
        },
    ]


def test_save_neuron_search_results_csv_creates_parent_directory_and_writes_expected_columns(tmp_path):
    output_path = tmp_path / "nested" / "results.csv"
    dataframe = pd.DataFrame(
        [
            {
                "neuron": 1,
                "formula": "tok:A",
                "iou": 0.75,
                "weight_ent": 1.1,
                "weight_neut": 1.2,
                "weight_contr": 1.3,
            }
        ]
    )

    saved_dataframe = save_neuron_search_results_csv(
        dataframe=dataframe,
        output_csv_path=output_path,
    )

    assert output_path.exists()
    loaded_dataframe = pd.read_csv(output_path)
    assert list(loaded_dataframe.columns) == [
        "neuron",
        "formula",
        "iou",
        "weight_ent",
        "weight_neut",
        "weight_contr",
    ]
    assert loaded_dataframe.to_dict("records") == saved_dataframe.to_dict("records")

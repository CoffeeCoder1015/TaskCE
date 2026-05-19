from collections import Counter

import pandas as pd

from analysis.ablation import run_ablation
from analysis.ablation_analysis import run_ablation_analysis


def write_search_csv(path):
    pd.DataFrame(
        [
            {
                "neuron": 1,
                "formula": "tok:good",
                "iou": 0.9,
                "weight_ent": 1.0,
                "weight_neut": 0.0,
                "weight_contr": 0.0,
            },
            {
                "neuron": 2,
                "formula": "tok:bad",
                "iou": 0.01,
                "weight_ent": 0.0,
                "weight_neut": 1.0,
                "weight_contr": 0.0,
            },
            {
                "neuron": 3,
                "formula": "LOW_ACTS_PRUNED",
                "iou": 0.0,
                "weight_ent": 0.0,
                "weight_neut": 0.0,
                "weight_contr": 1.0,
            },
        ]
    ).to_csv(path, index=False)


def test_run_ablation_runs_baseline_and_cumulative_groups(tmp_path):
    result_csv = tmp_path / "search.csv"
    write_search_csv(result_csv)
    analysis_result = run_ablation_analysis(result_csv, output_dir=tmp_path)
    calls = []

    def inference_engine(neuron_ids):
        calls.append(neuron_ids)
        if neuron_ids is None:
            return Counter({"success": 8, "fail": 2, "reject": 0})
        return Counter({"success": 8 - len(neuron_ids), "fail": 2 + len(neuron_ids), "reject": 0})

    result = run_ablation(
        analysis_result,
        inference_engine=inference_engine,
        output_dir=tmp_path,
    )

    assert calls == [
        None,
        [1],
        [2],
        [2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
    ]
    assert set(result.result_paths) == {"good", "bad", "iou_ranked"}
    assert set(result.plot_paths) == {"good_bad", "combined"}

    good_df = pd.read_csv(tmp_path / "ablation_cumulative_good.csv")
    bad_df = pd.read_csv(tmp_path / "ablation_cumulative_bad.csv")
    iou_df = pd.read_csv(tmp_path / "ablation_cumulative_iou.csv")

    assert good_df["accuracy"].tolist() == [0.7]
    assert bad_df["n_neurons"].tolist() == [1, 2]
    assert iou_df["n_neurons"].tolist() == [1, 2, 3]
    assert iou_df["accuracy_delta"].tolist()[-1] == -0.3

import os

import pandas as pd

from theoretical.ablation.analysis import plot_ablation_results


def write_ablation_csv(path, accuracies):
    pd.DataFrame(
        {
            "percentile": [0.1, 1.0],
            "accuracy": accuracies,
        }
    ).to_csv(path, index=False)


def test_plot_ablation_results_saves_expected_pngs(tmp_path):
    good_path = tmp_path / "good.csv"
    bad_path = tmp_path / "bad.csv"
    iou_path = tmp_path / "iou.csv"
    write_ablation_csv(good_path, [0.8, 0.7])
    write_ablation_csv(bad_path, [0.81, 0.79])
    write_ablation_csv(iou_path, [0.8, 0.68])

    paths = plot_ablation_results(good_path, bad_path, iou_path, tmp_path)

    assert set(paths) == {"good_bad", "combined"}
    assert os.path.getsize(paths["good_bad"]) > 0
    assert os.path.getsize(paths["combined"]) > 0

import pandas as pd
import pytest

from analysis.ablation_analysis import (
    SEARCH_RESULT_COLUMNS,
    run_ablation_analysis,
)


def write_search_csv(path):
    pd.DataFrame(
        [
            {
                "neuron": 10,
                "formula": "tok:A",
                "iou": 0.8,
                "weight_ent": 1.0,
                "weight_neut": -0.5,
                "weight_contr": 0.2,
            },
            {
                "neuron": 11,
                "formula": "LOW_ACTS_PRUNED",
                "iou": 0.0,
                "weight_ent": 0.1,
                "weight_neut": 0.2,
                "weight_contr": 0.3,
            },
            {
                "neuron": 12,
                "formula": "tok:B",
                "iou": 0.01,
                "weight_ent": -2.0,
                "weight_neut": 0.5,
                "weight_contr": 0.1,
            },
        ]
    ).to_csv(path, index=False)


def test_run_ablation_analysis_writes_reports_and_ranked_groups(tmp_path):
    result_csv = tmp_path / "search.csv"
    write_search_csv(result_csv)

    result = run_ablation_analysis(result_csv, output_dir=tmp_path)

    assert result.good_neurons["neuron"].tolist() == [10]
    assert result.bad_neurons["neuron"].tolist() == [12, 11]
    assert result.iou_ranked_neurons["neuron"].tolist() == [10, 12, 11]
    assert (tmp_path / "ablation_analysis_report.txt").exists()
    assert not (tmp_path / "ablation_group_summary.csv").exists()
    assert not (tmp_path / "ablation_iou_bands.csv").exists()
    assert not (tmp_path / "weight_contributions.csv").exists()

    report = (tmp_path / "ablation_analysis_report.txt").read_text()
    assert "ABLATION PRE-RUN ANALYSIS" in report
    assert "GROUP SUMMARY" in report
    assert "IOU BAND BOUNDARIES" in report
    assert "entailment" in report
    assert "0.7" in report


def test_run_ablation_analysis_requires_expected_columns(tmp_path):
    result_csv = tmp_path / "bad.csv"
    pd.DataFrame({"neuron": [1], "formula": ["tok:A"]}).to_csv(result_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        run_ablation_analysis(result_csv, output_dir=tmp_path)

    assert SEARCH_RESULT_COLUMNS == [
        "neuron",
        "formula",
        "iou",
        "weight_ent",
        "weight_neut",
        "weight_contr",
    ]

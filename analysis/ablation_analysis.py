from dataclasses import dataclass
from io import StringIO
from math import ceil
from pathlib import Path

import pandas as pd


SEARCH_RESULT_COLUMNS = [
    "neuron",
    "formula",
    "iou",
    "weight_ent",
    "weight_neut",
    "weight_contr",
]
PRUNED_FORMULA = "LOW_ACTS_PRUNED"
DEFAULT_PERCENTILES = tuple(step / 10 for step in range(1, 11))


@dataclass(frozen=True)
class AblationAnalysisConfig:
    iou_min: float = 0.05
    garbage_formulas: tuple[str, ...] = ()
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES


@dataclass(frozen=True)
class AblationAnalysisResult:
    output_dir: Path
    good_neurons: pd.DataFrame
    bad_neurons: pd.DataFrame
    iou_ranked_neurons: pd.DataFrame


def run_ablation_analysis(result_csv_path, output_dir=None, config=None):
    config = config or AblationAnalysisConfig()
    result_csv_path = Path(result_csv_path)
    output_dir = Path(output_dir) if output_dir is not None else result_csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: load the search result CSV produced after feature search.
    search_results = pd.read_csv(result_csv_path)
    missing_columns = [
        column for column in SEARCH_RESULT_COLUMNS if column not in search_results.columns
    ]
    if missing_columns:
        raise ValueError(f"search result CSV missing required columns: {missing_columns}")
    search_results = search_results[SEARCH_RESULT_COLUMNS].copy()

    # Stage 2: split the search rows into the ranked groups used by ablation.
    garbage_formulas = set(config.garbage_formulas)
    garbage_formulas.add(PRUNED_FORMULA)
    garbage_mask = search_results["formula"].isin(garbage_formulas)
    good_neurons = search_results[
        (~garbage_mask) & (search_results["iou"] >= config.iou_min)
    ].sort_values("iou", ascending=False, ignore_index=True)
    bad_neurons = search_results[
        garbage_mask | (search_results["iou"] < config.iou_min)
    ].sort_values("iou", ascending=False, ignore_index=True)
    iou_ranked_neurons = search_results.sort_values(
        "iou",
        ascending=False,
        ignore_index=True,
    )

    # Stage 3: build summary tables for threshold review before expensive ablations.
    group_summary = pd.DataFrame(
        [
            {
                "group": "good",
                "n_neurons": len(good_neurons),
                "iou_min": good_neurons["iou"].min() if len(good_neurons) else 0.0,
                "iou_max": good_neurons["iou"].max() if len(good_neurons) else 0.0,
                "iou_mean": good_neurons["iou"].mean() if len(good_neurons) else 0.0,
            },
            {
                "group": "bad",
                "n_neurons": len(bad_neurons),
                "iou_min": bad_neurons["iou"].min() if len(bad_neurons) else 0.0,
                "iou_max": bad_neurons["iou"].max() if len(bad_neurons) else 0.0,
                "iou_mean": bad_neurons["iou"].mean() if len(bad_neurons) else 0.0,
            },
            {
                "group": "iou_ranked",
                "n_neurons": len(iou_ranked_neurons),
                "iou_min": iou_ranked_neurons["iou"].min(),
                "iou_max": iou_ranked_neurons["iou"].max(),
                "iou_mean": iou_ranked_neurons["iou"].mean(),
            },
        ]
    )

    # Stage 4: build IoU band boundaries for the same percentile schedule as ablation.
    band_records = []
    for group_name, group_frame in [
        ("good", good_neurons),
        ("bad", bad_neurons),
        ("iou_ranked", iou_ranked_neurons),
    ]:
        previous_end = 0
        for percentile in config.percentiles:
            end = max(1, ceil(len(group_frame) * percentile)) if len(group_frame) else 0
            if end == previous_end:
                continue
            band = group_frame.iloc[previous_end:end]
            band_records.append(
                {
                    "group": group_name,
                    "percentile": percentile,
                    "start_index": previous_end,
                    "end_index": end,
                    "n_neurons": len(band),
                    "iou_min": band["iou"].min() if len(band) else 0.0,
                    "iou_max": band["iou"].max() if len(band) else 0.0,
                }
            )
            previous_end = end
    iou_bands = pd.DataFrame(band_records)

    # Stage 5: build weight contribution analysis by class.
    weight_contributions = search_results.copy()
    weight_contributions["total_weight"] = (
        weight_contributions["weight_ent"].abs()
        + weight_contributions["weight_neut"].abs()
        + weight_contributions["weight_contr"].abs()
    )
    weight_contributions["pct_total"] = (
        weight_contributions["total_weight"].rank(pct=True) * 100
    )
    weight_contributions["pct_ent"] = (
        weight_contributions["weight_ent"].rank(pct=True) * 100
    )
    weight_contributions["pct_neut"] = (
        weight_contributions["weight_neut"].rank(pct=True) * 100
    )
    weight_contributions["pct_contr"] = (
        weight_contributions["weight_contr"].rank(pct=True) * 100
    )
    dominant_columns = ["weight_ent", "weight_neut", "weight_contr"]
    dominant_names = {
        "weight_ent": "entailment",
        "weight_neut": "neutral",
        "weight_contr": "contradiction",
    }
    weight_contributions["dominant_class"] = (
        weight_contributions[dominant_columns].abs().idxmax(axis=1).map(dominant_names)
    )
    weight_contributions = weight_contributions.sort_values(
        "total_weight",
        ascending=False,
        ignore_index=True,
    )

    # Stage 6: print and save the original-style analysis report.
    report = StringIO()
    report.write("=" * 80 + "\n")
    report.write("ABLATION PRE-RUN ANALYSIS\n")
    report.write("=" * 80 + "\n\n")
    report.write(f"Search result CSV: {result_csv_path}\n")
    report.write(f"Total neurons: {len(search_results)}\n")
    report.write(f"Zero IoU neurons: {(search_results['iou'] == 0).sum()}\n")
    report.write(f"IoU > 0 neurons: {(search_results['iou'] > 0).sum()}\n\n")
    report.write("IoU distribution:\n")
    report.write(search_results["iou"].describe().to_string())
    report.write("\n\nTop 10 formulas:\n")
    report.write(search_results["formula"].value_counts().head(10).to_string())
    report.write("\n\n")
    report.write("=" * 80 + "\n")
    report.write("GROUP SUMMARY\n")
    report.write("=" * 80 + "\n")
    report.write(group_summary.to_string(index=False))
    report.write("\n\n")
    report.write("=" * 80 + "\n")
    report.write("IOU BAND BOUNDARIES\n")
    report.write("=" * 80 + "\n")
    report.write(iou_bands.to_string(index=False))
    report.write("\n\n")
    report.write("=" * 80 + "\n")
    report.write("TOP WEIGHT CONTRIBUTIONS\n")
    report.write("=" * 80 + "\n")
    report.write(
        weight_contributions[
            [
                "neuron",
                "formula",
                "iou",
                "weight_ent",
                "weight_neut",
                "weight_contr",
                "total_weight",
                "dominant_class",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    report.write("\n")

    report_text = report.getvalue()
    print(report_text)

    report_path = output_dir / "ablation_analysis_report.txt"
    report_path.write_text(report_text)

    return AblationAnalysisResult(
        output_dir=output_dir,
        good_neurons=good_neurons,
        bad_neurons=bad_neurons,
        iou_ranked_neurons=iou_ranked_neurons,
    )

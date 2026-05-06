import gc
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import ablation_plotter

CLASS_TOKEN_IDS = {
    "entailment": 806,
    "neutral": 25919,
    "contradiction": 10913,
}
CLASS_NAMES = ["entailment", "neutral", "contradiction"]
CLASS_IDS_LIST = [CLASS_TOKEN_IDS[label] for label in CLASS_NAMES]
LABEL_MAP = ["entailment", "neutral", "contradiction"]

ABLATION_IOU_MIN = 0.05
GARBAGE_FEATURES = {
    "pre:tok:in",
    "((pre:tag:nn OR pre:tag:nns) OR pre:tok:man)",
}
CUMULATIVE_PERCENTILES = np.linspace(0.1, 1.0, 10)


def canonicalize_result_columns(df):
    df = df.copy()
    rename_map = {
        "feature": "formula",
        "w_entail": "weight_ent",
        "w_neutral": "weight_neut",
        "w_contra": "weight_contr",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def compute_differential_bands(df_sorted, percentiles):
    if len(df_sorted) == 0:
        return []

    results = []
    prev_end_idx = 0
    for i, pct in enumerate(percentiles):
        end_idx = int(np.ceil(len(df_sorted) * pct))
        band = df_sorted.iloc[prev_end_idx:end_idx]
        start_pct = percentiles[i - 1] if i > 0 else 0.0
        results.append(
            {
                "band_label": f"{int(start_pct * 100)}-{int(pct * 100)}%",
                "start_idx": prev_end_idx,
                "end_idx": end_idx,
                "n_neurons": len(band),
                "iou_min": band["iou"].min() if len(band) else 0.0,
                "iou_max": band["iou"].max() if len(band) else 0.0,
            }
        )
        prev_end_idx = end_idx
    return results


def print_band_examples(title, df, percentiles=CUMULATIVE_PERCENTILES):
    sorted_df = df.sort_values("iou", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total neurons: {len(sorted_df)}")
    for band in compute_differential_bands(sorted_df, percentiles):
        band_df = sorted_df.iloc[band["start_idx"]:band["end_idx"]]
        if band_df.empty:
            continue
        sample = band_df.nlargest(5, "iou")[["neuron", "formula", "iou"]]
        print(
            f"\nBand {band['band_label']} "
            f"({band['n_neurons']} neurons, IoU {band['iou_min']:.4f}-{band['iou_max']:.4f})"
        )
        print(sample.to_string(index=False))


def print_stats(df):
    print("\n" + "=" * 80)
    print("NEURON STATS")
    print("=" * 80)
    print(f"Total neurons: {len(df)}")
    print(f"Zero IoU neurons: {(df['iou'] == 0).sum()}")
    print(f"IoU > 0 neurons: {(df['iou'] > 0).sum()}")
    print("\nIoU distribution:")
    print(df["iou"].describe().to_string())
    print("\nTop 10 formulas:")
    print(df["formula"].value_counts().head(10).to_string())

    pruned_mask = (
        df["was_pruned"].astype(bool)
        if "was_pruned" in df.columns
        else pd.Series(False, index=df.index)
    )
    garbage_mask = df["formula"].isin(GARBAGE_FEATURES) | pruned_mask
    if pruned_mask.any():
        print(f"\nPruned low-activation neurons recovered: {int(pruned_mask.sum())}")
    good = df[~garbage_mask & (df["iou"] >= ABLATION_IOU_MIN)].copy()
    bad = df[garbage_mask | (df["iou"] < ABLATION_IOU_MIN)].copy()

    for label, group in [("GOOD", good), ("BAD", bad)]:
        print("\n" + "=" * 80)
        print(f"{label} NEURONS")
        print("=" * 80)
        print(f"Count: {len(group)}")
        if group.empty:
            continue
        print("\nIoU distribution:")
        print(group["iou"].describe().to_string())
        weight_cols = ["weight_ent", "weight_neut", "weight_contr"]
        if all(col in group.columns for col in weight_cols):
            print("\nWeight distributions:")
            print(group[weight_cols].describe().to_string())

    print_band_examples("IOU BAND EXAMPLES (ALL NEURONS)", df)
    return good, bad


def compute_percentile_ranks(df):
    df = df.copy()
    df["total_weight"] = (
        df["weight_ent"] + df["weight_neut"] + df["weight_contr"]
    )
    df["pct_total"] = df["total_weight"].rank(pct=True) * 100
    df["pct_ent"] = df["weight_ent"].rank(pct=True) * 100
    df["pct_neut"] = df["weight_neut"].rank(pct=True) * 100
    df["pct_contr"] = df["weight_contr"].rank(pct=True) * 100
    abs_cols = df[["weight_ent", "weight_neut", "weight_contr"]].abs()
    df["dominant_class"] = abs_cols.idxmax(axis=1).map(
        {
            "weight_ent": "entailment",
            "weight_neut": "neutral",
            "weight_contr": "contradiction",
        }
    )
    return df


def print_weight_contributions_by_band(df):
    ranked = compute_percentile_ranks(df)
    print("\n" + "=" * 80)
    print("WEIGHT CONTRIBUTION ANALYSIS")
    print("=" * 80)
    for title, weight_col in [
        ("TOTAL", "total_weight"),
        ("ENTAILMENT", "weight_ent"),
        ("NEUTRAL", "weight_neut"),
        ("CONTRADICTION", "weight_contr"),
    ]:
        print(f"\n--- {title} CONTRIBUTORS ---")
        sorted_df = ranked.sort_values(weight_col, ascending=False).reset_index(drop=True)
        prev_end_idx = 0
        for i, pct in enumerate(CUMULATIVE_PERCENTILES):
            end_idx = int(np.ceil(len(sorted_df) * pct))
            band_df = sorted_df.iloc[prev_end_idx:end_idx]
            start_pct = CUMULATIVE_PERCENTILES[i - 1] if i > 0 else 0.0
            prev_end_idx = end_idx
            if band_df.empty:
                continue
            sample = band_df.head(5)[
                ["neuron", "formula", "weight_ent", "weight_neut", "weight_contr", "dominant_class"]
            ]
            print(
                f"\nBand {int(start_pct * 100)}-{int(pct * 100)}% "
                f"({weight_col} {band_df[weight_col].min():.4f}-{band_df[weight_col].max():.4f})"
            )
            print(sample.to_string(index=False))
    return ranked


def load_model_with_lora(model_id, lora_path=None, dtype=torch.bfloat16, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
    )
    if lora_path is not None and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def resolve_layer(model, layer):
    named_layers = [(name, module) for name, module in model.named_modules() if name]
    if isinstance(layer, str):
        return dict(named_layers)[layer]
    return named_layers[layer][1]


def make_ablation_hook(neuron_indices):
    indices = torch.as_tensor(neuron_indices, dtype=torch.long)

    def hook_fn(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        local_indices = indices.to(tensor.device)
        patched = tensor.clone()
        patched[:, -1, local_indices] = 0.0
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    return hook_fn


def prepare_dataset(dataset, data_formatter):
    if "prompt" in dataset.column_names:
        return dataset
    return dataset.map(data_formatter)


def run_inference(model, tokenizer, dataset, layer, ablate_neurons=None, batch_size=32, desc="Inference"):
    handle = None
    if ablate_neurons:
        handle = resolve_layer(model, layer).register_forward_hook(
            make_ablation_hook(ablate_neurons)
        )

    predictions = []
    logits_by_class = []
    labels = []
    correct = 0
    input_device = next(model.parameters()).device

    try:
        with torch.inference_mode():
            for i in tqdm(range(0, len(dataset), batch_size), desc=desc, leave=False):
                batch = dataset[i:i + batch_size]
                tokenized = tokenizer.apply_chat_template(
                    batch["prompt"],
                    add_generation_prompt=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(input_device)
                out = model(**tokenized)
                nli_logits = out.logits[:, -1, CLASS_IDS_LIST].float().cpu().numpy()
                logits_by_class.append(nli_logits)

                pred_indices = nli_logits.argmax(axis=1)
                for pred_idx, label_idx in zip(pred_indices, batch["label"]):
                    pred = CLASS_NAMES[int(pred_idx)]
                    label = LABEL_MAP[int(label_idx)]
                    predictions.append(pred)
                    labels.append(label)
                    correct += pred == label
    finally:
        if handle is not None:
            handle.remove()

    logits_by_class = np.concatenate(logits_by_class, axis=0)
    accuracy = correct / len(predictions) if predictions else 0.0
    return predictions, logits_by_class, labels, accuracy


def run_ablation_group(
    group_name,
    ranked_neurons,
    model,
    tokenizer,
    dataset,
    layer,
    base_preds,
    base_logits_mean,
    base_acc,
    output_path,
):
    records = []
    if not ranked_neurons:
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        return df

    for pct in tqdm(CUMULATIVE_PERCENTILES, desc=f"Cumulative ablations ({group_name})"):
        n_neurons = max(1, int(np.ceil(len(ranked_neurons) * pct)))
        neurons_to_ablate = ranked_neurons[:n_neurons]
        preds, logits, _, acc = run_inference(
            model,
            tokenizer,
            dataset,
            layer,
            ablate_neurons=neurons_to_ablate,
            desc=f"{group_name} {int(pct * 100)}%",
        )
        logits_mean = logits.mean(axis=0)
        records.append(
            {
                "group": group_name,
                "percentile": pct,
                "n_neurons": n_neurons,
                "accuracy": acc,
                "accuracy_delta": acc - base_acc,
                "prediction_flips": sum(p != b for p, b in zip(preds, base_preds)),
                "logit_entail_delta": logits_mean[0] - base_logits_mean[0],
                "logit_neutral_delta": logits_mean[1] - base_logits_mean[1],
                "logit_contra_delta": logits_mean[2] - base_logits_mean[2],
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    return df


def print_ablation_comparison(good_res, bad_res, iou_res):
    print("\n" + "=" * 80)
    print("ABLATION RESULTS: Good vs Bad vs IoU Ranked")
    print("=" * 80)
    if good_res.empty or bad_res.empty or iou_res.empty:
        print("One or more ablation groups are empty; skipping comparison table.")
        return
    print(f"{'Pct':<5} | {'Good(N)':<8} | {'Good dAcc':<10} | {'Bad(N)':<8} | {'Bad dAcc':<10} | {'IoU(N)':<8} | {'IoU dAcc':<10}")
    print("-" * 90)
    for i in range(len(iou_res)):
        g = good_res.iloc[i]
        b = bad_res.iloc[i]
        r = iou_res.iloc[i]
        pct = int(r["percentile"] * 100)
        print(
            f"{pct:>3d}%  | {int(g['n_neurons']):>6d} | {g['accuracy_delta'] * 100:>+8.2f}% | "
            f"{int(b['n_neurons']):>6d} | {b['accuracy_delta'] * 100:>+8.2f}% | "
            f"{int(r['n_neurons']):>6d} | {r['accuracy_delta'] * 100:>+8.2f}%"
        )


def run_ablation_pipeline(
    result_csv,
    model_id,
    lora_path,
    dataset,
    data_formatter,
    layer,
    output_dir,
):
    df = canonicalize_result_columns(pd.read_csv(result_csv))
    good_df, bad_df = print_stats(df)
    ranked_df = print_weight_contributions_by_band(df)

    ranked_all = df.sort_values("iou", ascending=False)["neuron"].astype(int).tolist()
    ranked_good = good_df.sort_values("iou", ascending=False)["neuron"].astype(int).tolist()
    ranked_bad = bad_df.sort_values("iou", ascending=False)["neuron"].astype(int).tolist()
    if not ranked_good:
        print("No good neurons found after filtering; ablation will still run bad and IoU groups if available.")

    formatted_dataset = prepare_dataset(dataset, data_formatter)
    model, tokenizer = load_model_with_lora(model_id, lora_path=lora_path)
    try:
        print("\nRunning baseline inference for ablation.")
        base_preds, base_logits, _, base_acc = run_inference(
            model,
            tokenizer,
            formatted_dataset,
            layer,
            desc="Baseline",
        )
        print(f"Baseline accuracy: {base_acc:.4f} ({base_acc * 100:.2f}%)")
        base_logits_mean = base_logits.mean(axis=0)

        good_path = os.path.join(output_dir, "ablation_cumulative_good.csv")
        bad_path = os.path.join(output_dir, "ablation_cumulative_bad.csv")
        iou_path = os.path.join(output_dir, "ablation_cumulative_iou.csv")

        good_res = run_ablation_group(
            "good",
            ranked_good,
            model,
            tokenizer,
            formatted_dataset,
            layer,
            base_preds,
            base_logits_mean,
            base_acc,
            good_path,
        )
        bad_res = run_ablation_group(
            "bad",
            ranked_bad,
            model,
            tokenizer,
            formatted_dataset,
            layer,
            base_preds,
            base_logits_mean,
            base_acc,
            bad_path,
        )
        iou_res = run_ablation_group(
            "iou_ranked",
            ranked_all,
            model,
            tokenizer,
            formatted_dataset,
            layer,
            base_preds,
            base_logits_mean,
            base_acc,
            iou_path,
        )
        print_ablation_comparison(good_res, bad_res, iou_res)
        plot_paths = ablation_plotter.plot_ablation_results(
            good_path,
            bad_path,
            iou_path,
            output_dir,
        )
        print("Saved ablation plots:")
        for name, path in plot_paths.items():
            print(f"  {name}: {path}")

        weight_path = os.path.join(output_dir, "weight_contributions.csv")
        ranked_df.to_csv(weight_path, index=False)
        print(f"Saved weight contributions: {weight_path}")
        return good_path, bad_path, iou_path, weight_path
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

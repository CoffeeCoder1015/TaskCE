from analysis import run_ablation, run_ablation_analysis
from analysis.ablation_inference import AblationInferenceEngine


RESULT_CSV_PATH = "results/snli_beam_results.csv"
OUTPUT_DIR = "results"


def build_inference_engine():
    raise NotImplementedError(
        "Configure and return an AblationInferenceEngine with the model, task, layer, "
        "and optional LoRA checkpoint before running ablation.py."
    )


if __name__ == "__main__":
    analysis_result = run_ablation_analysis(
        result_csv_path=RESULT_CSV_PATH,
        output_dir=OUTPUT_DIR,
    )
    inference_engine = build_inference_engine()
    run_ablation(
        analysis_result=analysis_result,
        inference_engine=inference_engine,
        output_dir=OUTPUT_DIR,
    )

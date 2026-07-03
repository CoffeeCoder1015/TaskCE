from collections import Counter
import gc

from evaluation.evalConfig import EvalConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from peft import PeftModel

from capture.lora_checkpoints import latest_task_lora_checkpoints


def evaluate_predictions(prediction, answer):
    if prediction is None:
        return "reject"
    if prediction == answer:
        return "success"
    return "fail"


def print_results(task_name, stats):
    total = sum(stats.values())
    accuracy = stats["success"] / total if total > 0 else 0

    print(f"\n\033[94m{task_name} results\033[0m")
    print("\033[94mSuccess:\033[0m", stats["success"])
    print("\033[94mFail:\033[0m", stats["fail"])
    print("\033[94mReject:\033[0m", stats["reject"])
    print("\033[94mAccuracy:\033[0m", accuracy)


def eval_task(model_id, tokenizer, task, lora_path=None, batch_size=128):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)

    tg_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    formatted_dataset = task.dataset.map(task.data_formatter)
    prompts = formatted_dataset["prompt"]
    answers = formatted_dataset["answer"]

    responses_raw = []

    print(f"\nStarting {task.name} inference.")
    for i in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"Generating {task.name}",
    ):
        batch = prompts[i:i + batch_size]
        out = tg_pipeline(
            batch,
            batch_size=batch_size,
            return_full_text=False,
        )
        responses_raw.extend(out)
    print(f"{task.name} inference finished.")

    response_chats = [resp[0]["generated_text"] for resp in responses_raw]
    predictions = [task.eval_fn(response_chat) for response_chat in response_chats]
    results = [
        evaluate_predictions(prediction, answer)
        for prediction, answer in zip(predictions, answers)
    ]

    stats = Counter(results)

    del tg_pipeline, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def Evaluate(
    model_id: str,
    lora_dir: str,
    tasks: list[EvalConfig],
    *,
    lora_remote: bool = False,
    lora_token: str | bool | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"


    print("[Base model evaluation]")
    for task in tasks:
        stats = eval_task(model_id, tokenizer, task)
        print_results(task.name, stats)

    print("[Finetuned model evaluation]")
    latest_checkpoints = latest_task_lora_checkpoints(
        lora_dir,
        remote=lora_remote,
        token=lora_token,
    )

    for task in tasks:
        if task.name not in latest_checkpoints:
            print(f"Skipping {task.name}: no LoRA checkpoint found.")
            continue

        stats = eval_task(model_id, tokenizer, task, lora_path=latest_checkpoints[task.name])
        print_results(task.name,stats)

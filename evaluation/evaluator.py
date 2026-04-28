from collections import Counter
import os

from evaluation.evalConfig import EvalConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


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


def Evaluate(model_id: str,lora_dir: str, tasks: list[EvalConfig]):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tg_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


    print("[Base model evaluation]")
    for task in tasks:
        formatted_dataset = task.dataset.map(task.data_formatter)
        prompts = formatted_dataset["prompt"]
        answers = formatted_dataset["answer"]

        responses_raw = []
        batch_size = 8

        # -------   Actual inference stage ---------
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
        # -------   Actual inference stage ---------

        response_chats = [resp[0]["generated_text"] for resp in responses_raw]
        predictions = [task.eval_fn(response_chat) for response_chat in response_chats]
        results = [
            evaluate_predictions(prediction, answer)
            for prediction, answer in zip(predictions, answers)
        ]

        stats = Counter(results)
        print_results(task.name, stats)

    print("[Finetuned model evaluation]")
    task_paths = {
        path: os.path.join(lora_dir, path)
        for path in os.listdir(lora_dir)
        if os.path.isdir(os.path.join(lora_dir, path))
    }
    task_and_checkpts = {
        task_name: [
            os.path.join(task_path, path)
            for path in os.listdir(task_path)
            if os.path.isdir(os.path.join(task_path, path))
        ]
        for task_name, task_path in task_paths.items()
    }
    print(task_and_checkpts)

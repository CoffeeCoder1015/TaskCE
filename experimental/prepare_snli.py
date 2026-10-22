from datasets import load_dataset

def format_snli(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `premise` and "
                "`hypothesis` and respond with `entailment`, `neutral`, "
                "or `contradiction`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
    ]
    return example


def get_prepared_snli(tokenizer, split="validation"):
    snli = load_dataset("stanfordnlp/snli", split=split)
    snli = snli.map(format_snli)
    return snli.map(
        lambda example: {
            "input_ids": tokenizer.apply_chat_template(
                example["prompt"],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        }
    )

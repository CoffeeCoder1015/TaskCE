import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from experimental.lora_checkpoints import latest_task_lora_checkpoint


model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
lora_dir = "Heroi/multitune-lora-backup"
lora_remote = True
lora_token = os.environ.get("HF_TOKEN")


def print_layers(label, model):
    print(f"\n================ {label} ================\n")

    for index, (name, module) in enumerate(model.named_modules()):
        if name:
            print(f"{index:5}  {name:<90}  {type(module).__name__}")


base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.eval()

print_layers("BASE MODEL", base_model)

del base_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


lora_path = latest_task_lora_checkpoint(
    lora_dir,
    "snli",
    remote=lora_remote,
    token=lora_token,
)

if lora_path is None:
    raise FileNotFoundError("No LoRA checkpoint found for snli.")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

finetuned_model = PeftModel.from_pretrained(base_model, lora_path)
finetuned_model.eval()

print_layers(f"FINE-TUNED MODEL ({lora_path})", finetuned_model)

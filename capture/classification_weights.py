import gc
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from capture.capturer import latest_checkpoint


def classification_weights_from_lm_head(model, class_token_ids):
    class_ids = list(class_token_ids.values())
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)
    return lm_head_weight[class_ids].T


def latest_task_lora_checkpoint(lora_dir, task_name):
    task_dir = os.path.join(lora_dir, task_name)
    if not os.path.isdir(task_dir):
        return None

    checkpoints = [
        os.path.join(task_dir, path)
        for path in os.listdir(task_dir)
        if os.path.isdir(os.path.join(task_dir, path))
    ]
    return latest_checkpoint(checkpoints)


def get_classification_weights(model_id, lora_dir, task_name, class_token_ids):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_path = latest_task_lora_checkpoint(lora_dir, task_name)
    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path)
        if hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

    try:
        return classification_weights_from_lm_head(model, class_token_ids)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from capture.lora_checkpoints import latest_task_lora_checkpoint


def classification_weights_from_lm_head(model, class_token_ids):
    class_ids = list(class_token_ids.values())
    lm_head_weight = model.lm_head.weight.detach().cpu().to(torch.float32)
    return lm_head_weight[class_ids].T


def get_classification_weights(
    model_id,
    lora_dir,
    task_name,
    class_token_ids,
    *,
    lora_remote=False,
    lora_token=None,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_path = latest_task_lora_checkpoint(
        lora_dir,
        task_name,
        remote=lora_remote,
        token=lora_token,
    )
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

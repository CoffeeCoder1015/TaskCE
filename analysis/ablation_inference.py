from collections import Counter
from dataclasses import dataclass
import gc

from peft import PeftModel
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AblationTaskConfig:
    name: str
    dataset: object
    data_formatter: object


def make_ablation_hook(neuron_ids):
    def ablation_hook(module, inputs, output):
        ablated_output = output.clone()
        ablated_output[:, -1, neuron_ids] = 0.0
        return ablated_output

    return ablation_hook

class AblationInferenceEngine:
    def __init__(
        self,
        model_id,
        task,
        layer,
        class_token_ids,
        lora_path=None,
        batch_size=128,
        dtype=torch.bfloat16,
        device_map="auto",
    ):
        self.task = task
        self.layer = layer
        self.batch_size = batch_size
        self.class_names = list(class_token_ids)
        self.class_ids_list = [class_token_ids[label] for label in self.class_names]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        if lora_path is not None:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        named_layers = [(name, module) for name, module in self.model.named_modules() if name]
        if isinstance(layer, str):
            self.ablation_layer = dict(named_layers)[layer]
        else:
            self.ablation_layer = named_layers[layer][1]

        formatted_dataset = task.dataset.map(task.data_formatter)
        self.prompts = formatted_dataset["prompt"]
        self.answers = formatted_dataset["answer"]

    def __call__(self, neuron_ids):
        hook_handle = None
        if neuron_ids is not None:
            hook_handle = self.ablation_layer.register_forward_hook(
                make_ablation_hook(neuron_ids)
            )

        try:
            responses_raw = []
            for index in tqdm(
                range(0, len(self.prompts), self.batch_size),
                desc=f"Ablating {self.task.name}",
                leave=False,
            ):
                batch = self.prompts[index:index + self.batch_size]
                tokenized = self.tokenizer.apply_chat_template(
                    batch,
                    add_generation_prompt=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)
                with torch.inference_mode():
                    out = self.model(**tokenized)
                class_logits = out.logits[:, -1, self.class_ids_list]
                predictions = class_logits.argmax(dim=1).tolist()
                responses_raw.extend(self.class_names[prediction] for prediction in predictions)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        results = []
        for prediction, answer in zip(responses_raw, self.answers):
            if prediction is None:
                results.append("reject")
            elif prediction == answer:
                results.append("success")
            else:
                results.append("fail")
        return Counter(results)

    def close(self):
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

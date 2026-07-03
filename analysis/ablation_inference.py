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


def resolve_ablation_layer(model, layer):
    named_layers = [
        (name, module)
        for name, module in model.named_modules()
        if name
    ]
    layer_by_name = dict(named_layers)

    if isinstance(layer, str):
        resolved_path = resolve_ablation_layer_path(layer_by_name, layer)
        resolved_module = layer_by_name[resolved_path]
    else:
        resolved_path, resolved_module = named_layers[layer]

    return resolved_module, resolved_path


def resolve_ablation_layer_path(layer_by_name, layer):
    if layer in layer_by_name:
        return layer

    suffix_matches = [
        name
        for name in layer_by_name
        if name.endswith(f".{layer}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if suffix_matches:
        raise KeyError(
            f"Ablation layer {layer!r} matched multiple module paths: "
            f"{suffix_matches}"
        )

    raise KeyError(layer)


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
        self.ablation_layer, self.resolved_layer = resolve_ablation_layer(self.model, layer)

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

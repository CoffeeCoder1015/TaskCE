"""Train, persist, and load compositional feature tokenizers."""

from collections.abc import Iterator, Sequence
from pathlib import Path

from tokenizers.trainers import WordLevelTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .tokenizer import (
    BASE_SPECIAL_TOKENS,
    SPACY_POS_TAG_TOKENS,
    attach_spacy_pretokenizer,
    build_tokenizer,
    detach_spacy_pretokenizer,
)


TOKENIZER_DIRECTORY = (
    Path(__file__).resolve().parents[2] / "resources" / "tokenizers"
)


def tokenizer_corpus(dataset, columns: Sequence[str]) -> Iterator[str]:
    """Yield one training document per dataset row."""
    for example in dataset:
        yield "\n".join(str(example[column]) for column in columns)


def get_tokenizer(
    name: str,
    dataset,
    columns: Sequence[str],
    *,
    enable_pos: bool = False,
):
    """Load a tokenizer resource, training it once when it is absent."""
    resource_name = f"{name}_pos" if enable_pos else name
    tokenizer_path = TOKENIZER_DIRECTORY / resource_name

    if not (tokenizer_path / "tokenizer.json").exists():
        tokenizer = build_tokenizer(enable_pos=enable_pos)
        special_tokens = list(BASE_SPECIAL_TOKENS)
        if enable_pos:
            special_tokens.extend(SPACY_POS_TAG_TOKENS)
        trainer = WordLevelTrainer(special_tokens=special_tokens)
        tokenizer.train_from_iterator(
            tokenizer_corpus(dataset, columns),
            trainer=trainer,
        )
        detach_spacy_pretokenizer(tokenizer)

        tokenizer_options = {
            "tokenizer_object": tokenizer,
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        if enable_pos:
            tokenizer_options["additional_special_tokens"] = (
                SPACY_POS_TAG_TOKENS
            )
        tokenizer = PreTrainedTokenizerFast(**tokenizer_options)
        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    attach_spacy_pretokenizer(
        tokenizer.backend_tokenizer,
        enable_pos=enable_pos,
    )
    return tokenizer

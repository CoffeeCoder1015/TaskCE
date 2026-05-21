from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers.trainers import WordLevelTrainer

from .corpus import build_corpus_from_dataset
from .tokenizer import (
    SPACY_POS_TAG_TOKENS,
    TOKENIZER_SPECIAL_TOKENS,
    attach_spacy_pretokenizer,
    build_tokenizer,
    detach_spacy_pretokenizer,
)


TOKENIZER_DIR = Path("tokenizers")


def get_tokenizer(name, formatted_dataset, enable_pos=False):
    if enable_pos:
        name = f"{name}_pos"
    tokenizer_path = TOKENIZER_DIR / name

    if not (tokenizer_path / "tokenizer.json").exists():
        tokenizer = build_tokenizer(enable_pos=enable_pos)
        trainer = WordLevelTrainer(
            special_tokens=TOKENIZER_SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(
            build_corpus_from_dataset(formatted_dataset),
            trainer=trainer,
        )
        detach_spacy_pretokenizer(tokenizer)
        
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        if enable_pos:
            tokenizer.add_special_tokens({"additional_special_tokens": SPACY_POS_TAG_TOKENS})

        tokenizer_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    attach_spacy_pretokenizer(tokenizer.backend_tokenizer, enable_pos=enable_pos)
    return tokenizer

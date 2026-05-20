from datasets import load_dataset
from corpus import build_corpus_from_dataset
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.models import WordLevel
from tokenizers import pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast
import spacy


class SpacyPretokenizer:
    def __init__(self,enable_pos=False):
        self.enable_pos = enable_pos
        self.nlp = spacy.load("en_core_web_sm")

    def spacy_split(self, i: int, string: NormalizedString):
        del i # this is unused for our purposes
        doc = self.nlp(str(string))

        splits = []
        for token in doc:
            start,end = token.idx , token.idx + len(token.text)
            split = string[start:end]

            splits.append(split)

            if self.enable_pos:
                pos = f"<{token.tag_}>"
                splits.append(NormalizedString(pos))

        return splits

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.spacy_split)


def build_tokenizer():
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(SpacyPretokenizer(True))
    return tokenizer


if __name__ == "__main__":
    # Demo: train the spaCy-backed tokenizer on a tiny formatted SNLI slice.
    demo_tokenizer = build_tokenizer()
    print(demo_tokenizer.pre_tokenizer.pre_tokenize_str("Hello world"))

    demo_trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
    )

    def demo_to_text(example):
        demo_text = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        example["text"] = demo_text
        return example

    demo_dataset = load_dataset("snli", split="validation[:100]")
    demo_dataset = demo_dataset.map(demo_to_text)

    demo_train_data = build_corpus_from_dataset(demo_dataset)
    demo_tokenizer.train_from_iterator(demo_train_data, trainer=demo_trainer)

    demo_hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=demo_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    demo_encoded = demo_hf_tokenizer("a man is walking", add_special_tokens=False)
    print(demo_hf_tokenizer.tokenize("a man is walking"))
    print(demo_encoded["input_ids"])

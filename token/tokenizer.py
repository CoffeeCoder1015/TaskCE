from datasets import load_dataset
from huggingface_hub import split_tf_state_dict_into_shards
from corpus import build_corpus_from_dataset
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.models import WordLevel
from tokenizers import pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
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


tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(SpacyPretokenizer(True))
print(tokenizer.pre_tokenizer.pre_tokenize_str("Hello world"))

trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"],
)


def to_text(example):
    test_example = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
    example["text"] = test_example
    return example


snli = load_dataset("snli", split="validation[:100]")
snli = snli.map(to_text)

train_data = build_corpus_from_dataset(snli)
tokenizer.train_from_iterator(train_data, trainer=trainer)

encoding = tokenizer.encode("a man is walking")
print(encoding.tokens)
print(encoding.ids)

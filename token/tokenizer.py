from datasets import load_dataset
from corpus import build_corpus_from_dataset
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.models import WordLevel
from tokenizers import pre_tokenizers
from tokenizers.trainers import WordLevelTrainer
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast
import spacy


BASE_SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
SPACY_POS_TAG_TOKENS = [
    "<$>",      # symbol, currency
    "<''>",     # closing quotation mark
    "<,>",      # punctuation mark, comma
    "<-LRB->",  # left round bracket
    "<-RRB->",  # right round bracket
    "<.>",      # punctuation mark, sentence closer
    "<:>",      # punctuation mark, colon or ellipsis
    "<ADD>",    # email
    "<AFX>",    # affix
    "<CC>",     # conjunction, coordinating
    "<CD>",     # cardinal number
    "<DT>",     # determiner
    "<EX>",     # existential there
    "<FW>",     # foreign word
    "<HYPH>",   # punctuation mark, hyphen
    "<IN>",     # conjunction, subordinating or preposition
    "<JJ>",     # adjective (English), other noun-modifier (Chinese)
    "<JJR>",    # adjective, comparative
    "<JJS>",    # adjective, superlative
    "<LS>",     # list item marker
    "<MD>",     # verb, modal auxiliary
    "<NFP>",    # superfluous punctuation
    "<NN>",     # noun, singular or mass
    "<NNP>",    # noun, proper singular
    "<NNPS>",   # noun, proper plural
    "<NNS>",    # noun, plural
    "<PDT>",    # predeterminer
    "<POS>",    # possessive ending
    "<PRP>",    # pronoun, personal
    "<PRP$>",   # pronoun, possessive
    "<RB>",     # adverb
    "<RBR>",    # adverb, comparative
    "<RBS>",    # adverb, superlative
    "<RP>",     # adverb, particle
    "<SYM>",    # symbol
    "<TO>",     # infinitival "to"
    "<UH>",     # interjection
    "<VB>",     # verb, base form
    "<VBD>",    # verb, past tense
    "<VBG>",    # verb, gerund or present participle
    "<VBN>",    # verb, past participle
    "<VBP>",    # verb, non-3rd person singular present
    "<VBZ>",    # verb, 3rd person singular present
    "<WDT>",    # wh-determiner
    "<WP>",     # wh-pronoun, personal
    "<WP$>",    # wh-pronoun, possessive
    "<WRB>",    # wh-adverb
    "<XX>",     # unknown
    "<_SP>",    # whitespace
    "<``>",     # opening quotation mark
]
TOKENIZER_SPECIAL_TOKENS = BASE_SPECIAL_TOKENS + SPACY_POS_TAG_TOKENS


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
        special_tokens=TOKENIZER_SPECIAL_TOKENS,
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
        additional_special_tokens=SPACY_POS_TAG_TOKENS,
    )

    demo_encoded = demo_hf_tokenizer("a man is walking", add_special_tokens=False)
    print(demo_hf_tokenizer.tokenize("a man is walking"))
    print(demo_encoded["input_ids"])

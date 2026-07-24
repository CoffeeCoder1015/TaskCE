"""Tokenizer primitives for compositional feature construction."""

import spacy
from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase


BASE_SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
SPACY_POS_TAG_TOKENS = [
    "<$>",
    "<''>",
    "<,>",
    "<-LRB->",
    "<-RRB->",
    "<.>",
    "<:>",
    "<ADD>",
    "<AFX>",
    "<CC>",
    "<CD>",
    "<DT>",
    "<EX>",
    "<FW>",
    "<HYPH>",
    "<IN>",
    "<JJ>",
    "<JJR>",
    "<JJS>",
    "<LS>",
    "<MD>",
    "<NFP>",
    "<NN>",
    "<NNP>",
    "<NNPS>",
    "<NNS>",
    "<PDT>",
    "<POS>",
    "<PRP>",
    "<PRP$>",
    "<RB>",
    "<RBR>",
    "<RBS>",
    "<RP>",
    "<SYM>",
    "<TO>",
    "<UH>",
    "<VB>",
    "<VBD>",
    "<VBG>",
    "<VBN>",
    "<VBP>",
    "<VBZ>",
    "<WDT>",
    "<WP>",
    "<WP$>",
    "<WRB>",
    "<XX>",
    "<_SP>",
    "<``>",
]


class SpacyPretokenizer:
    """Split text with spaCy and optionally emit POS tags as adjacent tokens."""

    def __init__(self, enable_pos: bool = False):
        self.enable_pos = enable_pos
        disabled_components = ["parser", "ner", "lemmatizer"]
        if not enable_pos:
            disabled_components.append("tagger")
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=disabled_components,
        )

    def split(self, _index: int, text: NormalizedString):
        pieces = []
        for token in self.nlp(str(text)):
            pieces.append(text[token.idx : token.idx + len(token.text)])
            if self.enable_pos:
                pieces.append(NormalizedString(f"<{token.tag_}>"))
        return pieces

    def pre_tokenize(self, pretokenized: PreTokenizedString):
        pretokenized.split(self.split)


def attach_spacy_pretokenizer(tokenizer, *, enable_pos: bool = False):
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
        SpacyPretokenizer(enable_pos=enable_pos)
    )
    return tokenizer


def detach_spacy_pretokenizer(tokenizer):
    """Replace the runtime-only custom pretokenizer before serialization."""
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def build_tokenizer(*, enable_pos: bool = False):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()
    return attach_spacy_pretokenizer(tokenizer, enable_pos=enable_pos)

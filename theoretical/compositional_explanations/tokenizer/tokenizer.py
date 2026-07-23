from tokenizers import NormalizedString, PreTokenizedString, Tokenizer
from tokenizers.models import WordLevel
from tokenizers import pre_tokenizers
from tokenizers.normalizers import Lowercase
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
TOKENIZER_SPECIAL_TOKENS = BASE_SPECIAL_TOKENS


class SpacyPretokenizer:
    def __init__(self,enable_pos=False):
        self.enable_pos = enable_pos
        disable = [] if enable_pos else ["tagger"]
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer", *disable])

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


def attach_spacy_pretokenizer(tokenizer,enable_pos=False):
    tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(
        SpacyPretokenizer(enable_pos=enable_pos)
    )
    return tokenizer


def detach_spacy_pretokenizer(tokenizer):
    # Python custom pretokenizers are runtime-only: HF fast tokenizers deepcopy
    # and serialize tokenizer JSON, so save/load uses WhitespaceSplit and then
    # reattaches spaCy after loading.
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def build_tokenizer(enable_pos=False):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.normalizer = Lowercase()
    return attach_spacy_pretokenizer(tokenizer, enable_pos=enable_pos)

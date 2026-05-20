from .orchestration import get_tokenizer
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

__all__ = [
    "BASE_SPECIAL_TOKENS",
    "SPACY_POS_TAG_TOKENS",
    "TOKENIZER_SPECIAL_TOKENS",
    "get_tokenizer",
]

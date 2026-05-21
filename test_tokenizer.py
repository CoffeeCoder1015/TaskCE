from WordTokenizer.orchestration import get_tokenizer
from datasets import load_dataset
from WordTokenizer.tokenizer import SPACY_POS_TAG_TOKENS

def demo():
    d = load_dataset('snli', split='validation[:10]')
    t = get_tokenizer("test_pos_1", d, enable_pos=True)
    print("Tokens with True:", len(t.additional_special_tokens))

    t2 = get_tokenizer("test_pos_1", d, enable_pos=False)
    print("Tokens with False:", len(t2.additional_special_tokens))

    # try to remove
    t2.additional_special_tokens = [x for x in t2.additional_special_tokens if x not in SPACY_POS_TAG_TOKENS]
    print("Tokens with False after remove:", len(t2.additional_special_tokens))
    
if __name__ == "__main__":
    demo()

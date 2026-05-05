SKIP_TOKENS = {"a", "an", "the", "of", ".", ",", ""}


def token_should_be_skipped(token_id: int, tokenizer, special_token_ids: set[int]) -> bool:
    if token_id in special_token_ids:
        return True

    token = tokenizer.decode([token_id]).strip().lower()
    return token in SKIP_TOKENS

import datasets

def build_corpus_from_dataset(dataset:datasets.Dataset):
    # For consistency, it will only look for the "text" column and preprocessing will be done upstream by the user
    for text in dataset["text"]:
        yield text
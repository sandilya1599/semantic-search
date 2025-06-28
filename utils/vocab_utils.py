from collections import Counter

def build_vocab(tokenized_corpus):
    """
    Build a vocabulary dictionary with token frequencies from a tokenized corpus.
    Args:
        tokenized_corpus (list of list of str): The tokenized text corpus.
    Returns:
        dict: A dictionary mapping tokens to their frequency counts.
    """
    return Counter(token for line in tokenized_corpus for token in line)

from collections import defaultdict

def build_index(docs, dictionary, stopwords):
    """
    docs: dict of {doc_id: text}
    dictionary: set of valid dictionary words (for hyphen handling)
    stopwords: set of stopwords
    """
    from src.tokenizer import tokenize
    from src.stopwords import remove_stopwords
    from src.stemmer import stem_tokens

    index = defaultdict(lambda: defaultdict(int))

    for doc_id, text in docs.items():
        tokens = tokenize(text, dictionary)
        tokens = remove_stopwords(tokens, stopwords)
        stems = stem_tokens(tokens)

        for stem in stems:
            index[stem][doc_id] += 1

    return index

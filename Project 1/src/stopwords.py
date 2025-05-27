def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r') as f:
        return set(word.strip().lower() for word in f)

def remove_stopwords(tokens, stopwords):
    return [token for token in tokens if token not in stopwords]

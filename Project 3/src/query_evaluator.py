import math
from collections import defaultdict

def preprocess_query(query, dictionary, stopwords):
    from src.tokenizer import tokenize
    from src.stopwords import remove_stopwords
    from src.stemmer import stem_tokens

    tokens = tokenize(query, dictionary)
    tokens = remove_stopwords(tokens, stopwords)
    return stem_tokens(tokens)

def score_query(query_stems, index, docs):
    N = len(docs)
    doc_scores = defaultdict(float)

    doc_max_freqs = {
        doc_id: max(freqs.values()) for doc_id, freqs in compute_doc_term_freqs(index).items()
    }

    for term in query_stems:
        if term not in index:
            continue
        docs_with_term = index[term]
        nw = len(docs_with_term)
        idf = math.log2(N / nw)

        for doc_id, freq in docs_with_term.items():
            tf = freq / doc_max_freqs[doc_id]
            doc_scores[doc_id] += tf * idf

    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:10]

def compute_doc_term_freqs(index):
    doc_freqs = defaultdict(lambda: defaultdict(int))
    for term, postings in index.items():
        for doc_id, freq in postings.items():
            doc_freqs[doc_id][term] = freq
    return doc_freqs

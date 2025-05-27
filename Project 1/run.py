from src.tokenizer import tokenize, load_dictionary
from src.stopwords import load_stopwords, remove_stopwords
from src.stemmer import stem_tokens
from src.indexer import build_index
from src.query_evaluator import preprocess_query, score_query
import os
import json

def load_all_documents(folder_path):
    docs = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), 'r', encoding='latin-1') as f:
                docs[fname] = f.read()
    return docs

def get_first_sentence(text):
    return text.strip().split('.')[0] + '.'

if __name__ == "__main__":
    dictionary = load_dictionary('data/dictionary.txt')
    stopwords = load_stopwords('data/stopwords.txt')
    docs = load_all_documents('data/wiki_docs/')

    index = build_index(docs, dictionary, stopwords)

    with open('output/index.json', 'w') as f:
        json.dump(index, f, indent=2)

    print("Index built and saved.")

    test_queries = [
        "killing incident",
        "suspect charged with murder",
        "court",
        "jury sentenced murderer to prison",
        "movie",
        "entertainment films",
        "court appeal",
        "action film producer",
        "drunk driving accusations",
        "actor appeared in movie premiere",
    ]

    for i, query in enumerate(test_queries, start=1):
        query_stems = preprocess_query(query, dictionary, stopwords)
        top_docs = score_query(query_stems, index, docs)

        print(f"\nQuery {i}: {query}")
        print(f"{'Rank':<6} {'Doc ID':<20} {'Score':<10} First Sentence")
        for rank, (doc_id, score) in enumerate(top_docs, 1):
            first_sentence = get_first_sentence(docs[doc_id])
            print(f"{rank:<6} {doc_id:<20} {score:.4f}  {first_sentence}")

from src.tokenizer import load_dictionary
from src.stopwords import load_stopwords
from src.indexer import build_index
from src.query_evaluator import preprocess_query, score_query
from src.spell_checker import SpellChecker
from src.snippet_generator import SnippetGenerator
import os
import json
import time

def load_all_documents(folder_path):
    docs = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), 'r', encoding='latin-1') as f:
                docs[fname] = f.read()
    return docs

if __name__ == "__main__":
    start_time = time.time()

    print("Loading dictionary and stopwords...")
    dictionary = load_dictionary('data/dictionary.txt')
    stopwords = load_stopwords('data/stopwords.txt')

    print("Loading wiki documents...")
    docs = load_all_documents('data/wiki_docs/')
    print(f"Loaded {len(docs)} documents.")

    print("Building index...")
    index = build_index(docs, dictionary, stopwords)
    with open('output/index.json', 'w') as f:
        json.dump(index, f, indent=2)
    print("Index built and saved.")

    print("Initializing spell checker...")
    spell_checker = SpellChecker(dictionary, 'data/query_log.txt', docs)
    print("Spell checker ready.")

    print("Initializing snippet generator...")
    snippet_generator = SnippetGenerator()
    print("Snippet generator ready.")

    # Required queries
    test_queries = [
        "sentenced to prision",
        "open cuort case",
        "entretainment group",
        "tv axtor",
        "scheduled movie screning"
    ]

    print("Running queries and generating output...")
    with open('output/final_results.txt', 'w', encoding='utf-8') as f_out:
        for i, query in enumerate(test_queries, start=1):
            print(f"\nProcessing Query {i}: '{query}'")
            corrected_query = spell_checker.correct_query(query)
            print(f"  Corrected Query: '{corrected_query}'")

            corrected_query_stems = preprocess_query(corrected_query, dictionary, stopwords)
            top_docs = score_query(corrected_query_stems, index, docs)

            f_out.write(f"\nQuery {i}: {query}\n")
            f_out.write(f"Corrected Query: {corrected_query}\n")
            f_out.write(f"{'Rank':<6} {'Doc ID':<20} {'Score':<10} Snippets\n")

            for rank, (doc_id, score) in enumerate(top_docs, 1):
                print(f"  Retrieving snippets for Doc {doc_id}...")
                snippets = snippet_generator.generate_snippets(docs[doc_id], corrected_query)

                f_out.write(f"{rank:<6} {doc_id:<20} {score:.4f}\n")
                for snippet in snippets:
                    f_out.write(f"    {snippet}\n")
            f_out.write("\n")

    end_time = time.time()
    print(f"\nAll queries processed. Total run time: {end_time - start_time:.2f} seconds.")
    print("Results saved to 'output/final_results.txt'.")

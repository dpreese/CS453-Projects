import re
import math
import os
import json
from collections import defaultdict, Counter
from src.tokenizer import tokenize
from src.stopwords import remove_stopwords

def soundex(word):
    word = word.lower()
    first_letter = word[0].upper()

    mappings = {
        'bfpv': '1',
        'cgjkqsxz': '2',
        'dt': '3',
        'l': '4',
        'mn': '5',
        'r': '6'
    }

    def encode(char):
        for key in mappings:
            if char in key:
                return mappings[key]
        return ''

    encoded = first_letter
    prev_digit = ''
    for char in word[1:]:
        digit = encode(char)
        if digit != prev_digit:
            if digit != '':
                encoded += digit
            prev_digit = digit

    encoded = encoded.ljust(4, '0')[:4]
    return encoded

def levenshtein_distance(s1, s2):
    len1 = len(s1) + 1
    len2 = len(s2) + 1

    dp = [[0] * len2 for _ in range(len1)]

    for i in range(len1):
        dp[i][0] = i
    for j in range(len2):
        dp[0][j] = j

    for i in range(1, len1):
        for j in range(1, len2):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]

class SpellChecker:
    def __init__(self, dictionary, query_log_path, docs):
        self.dictionary = dictionary
        self.docs = docs
        self.word_freq = self.compute_word_frequencies()
        self.p_w = self.compute_p_w()

        # Use optimized compute_p_e_given_w with caching
        self.p_e_given_w = self.compute_p_e_given_w(query_log_path)

        # Precompute soundex for dictionary words
        self.soundex_dict = defaultdict(list)
        for word in self.dictionary:
            self.soundex_dict[soundex(word)].append(word)

    def compute_word_frequencies(self):
        word_counter = Counter()
        for text in self.docs.values():
            tokens = tokenize(text, self.dictionary)
            tokens = [token for token in tokens if token not in self.dictionary]  # skip dictionary words
            word_counter.update(tokens)
        return word_counter

    def compute_p_w(self):
        total = sum(self.word_freq.values()) + 1  # +1 to avoid div by zero
        p_w = defaultdict(float)
        for word, count in self.word_freq.items():
            p_w[word] = count / total
        return p_w

    def compute_p_e_given_w(self, query_log_path):
        # Check if cached file exists
        cache_file = 'output/p_e_given_w.json'
        if os.path.exists(cache_file):
            print("Loading cached P(e|w) matrix from disk...")
            with open(cache_file, 'r', encoding='utf-8') as f:
                p_e_given_w = json.load(f)
            # Convert back to defaultdict
            def dd(): return defaultdict(float)
            p_e_given_w_dd = defaultdict(dd)
            for e, w_dict in p_e_given_w.items():
                for w, val in w_dict.items():
                    p_e_given_w_dd[e][w] = val
            print("Finished loading P(e|w).")
            return p_e_given_w_dd

        # Otherwise, compute from scratch
        print("Computing P(e|w) matrix from query log...")
        correction_counts = defaultdict(lambda: defaultdict(int))
        total_counts = defaultdict(int)

        with open(query_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Processing {len(lines)} query log lines...")

        # Deduplicate tokens first
        all_tokens = set()
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            session, query = parts
            tokens = query.lower().split()
            all_tokens.update(tokens)

        print(f"Unique tokens to process: {len(all_tokens)}")

        # Process each unique token once
        for i, token in enumerate(all_tokens):
            for word in self.dictionary:
                if 0 < levenshtein_distance(token, word) <= 2:
                    correction_counts[token][word] += 1
                    total_counts[word] += 1

            if (i + 1) % 1 == 0:
                print(f"Processed {i + 1}/{len(all_tokens)} unique tokens...")

        # Build P(e|w)
        p_e_given_w = defaultdict(lambda: defaultdict(float))
        for e in correction_counts:
            for w in correction_counts[e]:
                p_e_given_w[e][w] = correction_counts[e][w] / (total_counts[w] + 1e-8)

        print("Finished computing P(e|w). Caching to disk...")
        # Save cache as json
        p_e_given_w_out = {e: dict(w_dict) for e, w_dict in p_e_given_w.items()}
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(p_e_given_w_out, f, indent=2)
        print("Cached P(e|w) to disk.")

        return p_e_given_w


    def correct_word(self, word):
        sdx = soundex(word)
        candidates = self.soundex_dict[sdx]

        candidates = [w for w in candidates if levenshtein_distance(word, w) <= 2]

        best_candidate = word
        best_score = 0.0

        for w in candidates:
            p_e_w = self.p_e_given_w[word].get(w, 1e-8)
            p_w = self.p_w.get(w, 1e-8)
            score = p_e_w * p_w

            if score > best_score:
                best_candidate = w
                best_score = score

        return best_candidate

    def correct_query(self, query):
        tokens = query.lower().split()
        corrected_tokens = []
        for token in tokens:
            if token in self.dictionary:
                corrected_tokens.append(token)
            else:
                corrected_tokens.append(self.correct_word(token))
        return ' '.join(corrected_tokens)

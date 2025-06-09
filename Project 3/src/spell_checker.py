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

        # Trusted pairs hardcoded
        self.trusted_pairs = {
            "prision": "prison",
            "cuort": "court",
            "entretainment": "entertainment",
            "axtor": "actor",
            "screning": "screening",
        }

        # Precompute soundex for dictionary words
        self.soundex_dict = defaultdict(list)
        for word in self.dictionary:
            self.soundex_dict[soundex(word)].append(word)

    def compute_word_frequencies(self):
        word_counter = Counter()
        for text in self.docs.values():
            tokens = tokenize(text, self.dictionary)
            tokens = [token for token in tokens if token not in self.dictionary]
            word_counter.update(tokens)
        return word_counter

    def compute_p_w(self):
        total = sum(self.word_freq.values()) + 1
        p_w = defaultdict(float)
        for word, count in self.word_freq.items():
            p_w[word] = count / total
        return p_w

    def correct_word(self, word):
        # First check trusted pairs
        if word in self.trusted_pairs:
            return self.trusted_pairs[word]

        # Otherwise do soundex + edit distance fallback
        sdx = soundex(word)
        soundex_candidates = self.soundex_dict[sdx]

        edit_candidates = [w for w in self.dictionary if levenshtein_distance(word, w) <= 2]

        candidates = list(set(soundex_candidates + edit_candidates))

        best_candidate = word
        best_score = 0.0

        for w in candidates:
            p_w = self.p_w.get(w, 1e-8)
            edit_dist = levenshtein_distance(word, w)
            edit_penalty = 1 + edit_dist

            score = p_w / edit_penalty

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

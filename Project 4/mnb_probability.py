# mnb_probability.py

import math
from collections import defaultdict, Counter

class MNBProbability:
    def __init__(self):
        # Store P(word | class) for each word/class
        self.word_probs = defaultdict(lambda: defaultdict(float))
        # Store P(class)
        self.class_probs = defaultdict(float)
        # Full vocabulary set
        self.vocab = set()
        # Set of all classes
        self.classes = set()
        # Total number of words seen in each class
        self.total_words_in_class = defaultdict(int)

    def computeWordProbability(self, training_set):
        """
        Compute P(word | class) with Laplace smoothing
        """
        # Count occurrences of each word in each class
        word_counts = defaultdict(lambda: defaultdict(int))
        class_doc_count = defaultdict(int)

        # Iterate over training docs
        for doc_words, label in training_set:
            class_doc_count[label] += 1
            self.classes.add(label)

            for word, count in doc_words.items():
                word_counts[label][word] += count
                self.total_words_in_class[label] += count
                self.vocab.add(word)

        V = len(self.vocab)  # Vocabulary size

        # Compute P(word | class) for each word/class
        for c in self.classes:
            for word in self.vocab:
                count_wc = word_counts[c][word]
                # Laplace smoothing formula
                prob = (count_wc + 1) / (self.total_words_in_class[c] + V)
                self.word_probs[c][word] = prob

    def computeClassProbability(self, training_set):
        """
        Compute P(class)
        """
        class_doc_count = defaultdict(int)
        total_docs = len(training_set)

        # Count how many docs per class
        for _, label in training_set:
            class_doc_count[label] += 1

        # Compute P(class)
        for c in class_doc_count:
            self.class_probs[c] = class_doc_count[c] / total_docs

    def getWordProbability(self, word, c):
        """
        Retrieve P(word | class), use smoothing if word unseen
        """
        V = len(self.vocab)
        return self.word_probs[c].get(word, 1 / (self.total_words_in_class[c] + V))

    def getClassProbability(self, c):
        """
        Retrieve P(class)
        """
        return self.class_probs[c]

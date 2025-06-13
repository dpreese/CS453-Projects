# mnb_classification.py

import math
from collections import defaultdict, Counter
from mnb_probability import MNBProbability

class MNBClassification:
    def __init__(self, stopwords):
        self.stopwords = stopwords
        # Set of selected features after feature selection
        self.selectedFeatures = set()
        # MNBProbability object that stores P(word|class), P(class)
        self.mnb_prob = MNBProbability()

    def _compute_information_gain(self, training_set, vocab):
        """
        Compute Information Gain (IG) for each word
        """
        # Compute P(class)
        class_counts = defaultdict(int)
        total_docs = len(training_set)

        for doc_words, label in training_set:
            class_counts[label] += 1

        P_c = {c: class_counts[c] / total_docs for c in class_counts}

        # Compute IG for each word
        IG = {}

        for word in vocab:
            # For each word, count how often it occurs in each class
            doc_with_w = defaultdict(int)
            doc_without_w = defaultdict(int)
            count_w = 0
            count_w_bar = 0

            for doc_words, label in training_set:
                if word in doc_words:
                    doc_with_w[label] += 1
                    count_w += 1
                else:
                    doc_without_w[label] += 1
                    count_w_bar += 1

            # Entropy H(C)
            H_C = -sum(P_c[c] * math.log2(P_c[c]) for c in P_c)

            # P(w) and P(~w)
            P_w = count_w / total_docs if count_w > 0 else 1e-10
            P_w_bar = count_w_bar / total_docs if count_w_bar > 0 else 1e-10

            # Conditional entropy terms
            term_w = 0.0
            if count_w > 0:
                for c in P_c:
                    P_c_given_w = (doc_with_w[c] / count_w) if count_w > 0 else 1e-10
                    if P_c_given_w > 0:
                        term_w += P_c_given_w * math.log2(P_c_given_w)

            term_w_bar = 0.0
            if count_w_bar > 0:
                for c in P_c:
                    P_c_given_w_bar = (doc_without_w[c] / count_w_bar) if count_w_bar > 0 else 1e-10
                    if P_c_given_w_bar > 0:
                        term_w_bar += P_c_given_w_bar * math.log2(P_c_given_w_bar)

            # Final IG formula
            IG[word] = H_C + P_w * term_w + P_w_bar * term_w_bar

        return IG

    def featureSelection(self, training_set, M):
        """
        Select top-M features based on IG
        """
        # Build vocabulary from training set
        vocab = set()
        for doc_words, _ in training_set:
            vocab.update(doc_words.keys())

        # Compute IG for vocab
        IG = self._compute_information_gain(training_set, vocab)

        # Select top-M features
        sorted_features = sorted(IG.items(), key=lambda x: x[1], reverse=True)
        self.selectedFeatures = set([word for word, _ in sorted_features[:M]])

    def label(self, doc_words):
        """
        Predict the label (class) of a document using MNB classifier
        """
        max_log_prob = float('-inf')
        best_class = None

        # For each class, compute log(P(class)) + sum log(P(word|class))
        for c in self.mnb_prob.classes:
            log_prob = math.log(self.mnb_prob.getClassProbability(c))

            for word, count in doc_words.items():
                # Use only selected features (if any)
                if (len(self.selectedFeatures) == 0) or (word in self.selectedFeatures):
                    pwc = self.mnb_prob.getWordProbability(word, c)
                    log_prob += count * math.log(pwc)

            # Keep track of best class
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = c

        return best_class

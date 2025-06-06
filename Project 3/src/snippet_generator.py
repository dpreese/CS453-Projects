import re
from nltk.stem import PorterStemmer

class SnippetGenerator:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def split_into_sentences(self, text):
        sentences = re.split(r'(?<=[.!?]) +', text)
        return sentences

    def compute_features(self, sentence, query_terms, position, total_sentences):
        features = {}

        # Basic cleaning
        sentence_tokens = re.findall(r'\w+', sentence.lower())

        # Required features
        query_term_count = sum(1 for token in sentence_tokens if token in query_terms)
        unique_query_terms = len(set(token for token in sentence_tokens if token in query_terms))
        density = query_term_count / max(len(sentence_tokens), 1)
        contiguous_run = self.longest_contiguous_run(sentence_tokens, query_terms)
        is_heading = 1 if sentence.isupper() else 0
        is_first_second_line = 1 if position < 2 else 0

        # Extra features
        sentence_length_score = 1 / (len(sentence_tokens) + 1)
        position_score = 1 / (position + 1)
        named_entity_score = sum(1 for word in sentence.split() if word.istitle())

        features['density'] = density
        features['contiguous_run'] = contiguous_run
        features['unique_query_terms'] = unique_query_terms
        features['total_query_terms'] = query_term_count
        features['is_first_second_line'] = is_first_second_line
        features['is_heading'] = is_heading

        features['sentence_length_score'] = sentence_length_score
        features['position_score'] = position_score
        features['named_entity_score'] = named_entity_score

        # Combine score (can adjust weights if desired)
        combined_score = (
            density +
            contiguous_run * 0.5 +
            unique_query_terms * 0.5 +
            query_term_count * 0.2 +
            is_first_second_line * 0.2 +
            is_heading * 0.3 +
            sentence_length_score * 0.5 +
            position_score * 0.3 +
            named_entity_score * 0.3
        )

        return combined_score

    def longest_contiguous_run(self, tokens, query_terms):
        max_run = 0
        current_run = 0
        for token in tokens:
            if token in query_terms:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run

    def bold_terms(self, sentence, query_terms, stemmer):
        def bold_word(word):
            stemmed = stemmer.stem(word.lower())
            return f"**{word}**" if stemmed in query_terms else word

        return ' '.join([bold_word(w) for w in sentence.split()])

    def generate_snippets(self, doc_text, corrected_query):
        sentences = self.split_into_sentences(doc_text)
        query_terms = set(self.stemmer.stem(word) for word in corrected_query.lower().split())

        sentence_scores = []
        for pos, sentence in enumerate(sentences):
            score = self.compute_features(sentence, query_terms, pos, len(sentences))
            sentence_scores.append((sentence, score, pos))

        # Sort by score (descending), pick top 2
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = sentence_scores[:2]

        # Bold terms in output
        snippets = []
        for sentence, _, _ in top_sentences:
            bolded = self.bold_terms(sentence, query_terms, self.stemmer)
            snippets.append(bolded)

        return snippets

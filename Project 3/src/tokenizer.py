import re

def load_dictionary(dict_path):
    with open(dict_path, 'r') as f:
        return set(word.strip().lower() for word in f)

def tokenize(text, dictionary):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    words = []
    for word in text.split():
        if '-' in word:
            parts = word.split('-')
            joined = ''.join(parts)
            if joined in dictionary:
                words.append(joined)
            else:
                words.extend(parts)
        else:
            words.append(word)
    return words

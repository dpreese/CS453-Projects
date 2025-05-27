import os
import math
import requests
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

# ------------------------
# Step 1: Trie Definitions
# ------------------------
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_complete = False
        self.word = None
        self.query_text = None
        self.freq = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, query):
        node = self.root
        for word in query.split():
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            if ' ' not in node.children:
                node.children[' '] = TrieNode()
            node = node.children[' ']
        node.is_complete = True
        node.query_text = query
        node.freq += 1

    def find_prefix_node(self, prefix):
        node = self.root
        for word in prefix.split():
            for char in word:
                if char not in node.children:
                    return None
                node = node.children[char]
            if ' ' not in node.children:
                return None
            node = node.children[' ']
        return node

    def collect_suggestions(self, node, prefix="", max_results=10):
        results = []

        def dfs(n, path):
            if n.is_complete:
                results.append((''.join(path).strip(), n.freq))
            for ch, child in n.children.items():
                dfs(child, path + [ch])

        dfs(node, [prefix])
        return sorted(results, key=lambda x: -x[1])[:max_results]

# -------------------------------
# Step 2: Parse AOL Query Sessions
# -------------------------------
def parse_sessions(file_path, max_session_gap=10):
    user_sessions = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            user_id, query, timestamp_str = parts
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            user_sessions[user_id].append((timestamp, query))

    sessionized = defaultdict(list)
    for user_id, queries in user_sessions.items():
        queries.sort()
        current_session = [queries[0]]
        for i in range(1, len(queries)):
            prev_time = queries[i - 1][0]
            curr_time = queries[i][0]
            if (curr_time - prev_time) > timedelta(minutes=max_session_gap):
                sessionized[user_id].append(current_session)
                current_session = [queries[i]]
            else:
                current_session.append(queries[i])
        if current_session:
            sessionized[user_id].append(current_session)

    return sessionized

# -------------------------------
# Step 3: Word Correlation (WCF)
# -------------------------------
stemmer = PorterStemmer()

def get_wcf(word1, word2):
    stemmed1 = stemmer.stem(word1.lower())
    stemmed2 = stemmer.stem(word2.lower())
    url = f"http://peacock.cs.byu.edu/CS453Proj2/?word1={stemmed1}&word2={stemmed2}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            value = float(response.text)
            return 0.0 if value == -1 else value
        return 0.0
    except requests.RequestException:
        return 0.0

# -------------------------------
# Step 4: Ranking Function
# -------------------------------
def normalize(value, max_value):
    if value == 0 or max_value == 0:
        return 0.0
    return math.log(1 + value) / math.log(1 + max_value)

def compute_suggrank(Q, SQ):
    f_raw = query_freq[SQ]
    m_raw = mod_map.get((Q, SQ), 0)
    w_raw = get_wcf(Q.split()[-1], SQ.split()[len(Q.split())]) if len(SQ.split()) > len(Q.split()) else 0.0

    f = normalize(f_raw, max_freq)
    m = normalize(m_raw, max_mod)
    w = w_raw

    min_component = min(f, m, w) if min(f, m, w) < 1 else 0.999
    numerator = f + m + w
    denominator = 1 - min_component
    return numerator / denominator if denominator != 0 else 0.0

# -------------------------------
# Step 5: Suggestion Function
# -------------------------------
def suggest_queries(query, trie, top_k=8):
    query = query.lower().strip()
    words = query.split()

    stopwords = {"a", "the", "an", "of", "in", "on", "at", "to", "for", "with"}
    while words and words[0] in stopwords:
        words = words[1:]
    if not words:
        return []

    cleaned_query = ' '.join(words)
    prefix_node = trie.find_prefix_node(cleaned_query)
    if not prefix_node:
        return []

    suggestions = trie.collect_suggestions(prefix_node, prefix=cleaned_query + " ", max_results=30)
    valid_suggestions = [sq for sq in suggestions if len(sq[0].split()) > len(words)]

    ranked = []
    for sq, _ in valid_suggestions:
        rank = compute_suggrank(cleaned_query, sq)
        ranked.append((sq, rank))

    return sorted(ranked, key=lambda x: -x[1])[:top_k]

# -------------------------------
# Step 6: Main (Example Execution)
# -------------------------------
if __name__ == "__main__":
    data_dir = "AOL-Clean-Data"  # Unzipped folder should be here
    all_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".txt")]

    combined_sessions = defaultdict(list)
    for file_path in all_files:
        file_sessions = parse_sessions(file_path)
        for user_id, sessions in file_sessions.items():
            combined_sessions[user_id].extend(sessions)

    query_freq = Counter()
    mod_map = Counter()

    for sessions in combined_sessions.values():
        for session in sessions:
            queries = [q.lower() for _, q in session]
            for i, q in enumerate(queries):
                query_freq[q] += 1
                if i + 1 < len(queries):
                    mod_pair = (queries[i], queries[i + 1])
                    mod_map[mod_pair] += 1

    max_freq = max(query_freq.values())
    max_mod = max(mod_map.values()) if mod_map else 1

    trie = Trie()
    for sessions in combined_sessions.values():
        for session in sessions:
            for _, query in session:
                trie.insert(query.lower())
    userInput = input("Please Enter your query: ")
    print(f"Suggestions for {userInput}:")
    suggestions = suggest_queries(userInput, trie)
    for s, r in suggestions:
        print(f"{s} → {r:.4f}")

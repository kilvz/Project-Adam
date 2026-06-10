import re

FACT_PATTERNS = [
    ("name", r"my name is (\w+)"),
    ("name", r"i(?:'m| am) called (\w+)"),
    ("location", r"i live in (\w[\w\s]*)"),
    ("location", r"i(?:'m| am) from (\w[\w\s]*)"),
    ("age", r"i(?:'m| am) (\d+) years? old"),
    ("preference", r"my favorite (\w+) is ([\w\s]+)"),
    ("likes", r"i like ([\w\s]+)"),
    ("dislikes", r"i don't? like ([\w\s]+)"),
]

POSITIVE_WORDS = {"love", "great", "amazing", "interesting", "cool", "nice",
    "beautiful", "wonderful", "thanks", "good", "excellent", "fantastic",
    "helpful", "perfect", "fun", "awesome", "brilliant", "fascinating"}
NEGATIVE_WORDS = {"bad", "wrong", "no", "hate", "terrible", "awful", "stupid",
    "boring", "incorrect", "useless", "pointless", "annoying", "horrible",
    "disappointing"}

NAME_PATTERNS = [
    (r"my name is (\w+)", 1),
    (r"i(?:'m| am) called (\w+)", 1),
    (r"call me (\w+)", 1),
    (r"(?:^|\s)I(?:'m| am) (\w+)(?:\s|$|\.|,)", 1),
    (r"(?:^|\s)I'm (\w+)(?:\s|$|\.|,)", 1),
]

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "it", "its",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "this", "that", "to", "of", "in", "for", "on", "and", "or", "but",
    "not", "do", "does", "did", "have", "has", "had", "be", "been",
    "with", "about", "at", "by", "from", "as", "so", "if", "then",
    "what", "why", "how", "when", "where", "who", "which", "all",
    "can", "will", "would", "should", "could"}

POS_REFS = ["this is wonderful", "I love this", "great", "amazing", "fantastic"]
NEG_REFS = ["this is terrible", "I hate this", "bad", "awful", "horrible"]


def extract_facts(text):
    if text.strip().endswith("?"):
        return []
    facts = []
    lower = text.lower()
    for cat, pattern in FACT_PATTERNS:
        for m in re.finditer(pattern, lower):
            facts.append((cat, m.group(0).strip()))
    return facts


def compute_implicit_reward(user_input, user_profile=None, embedder=None):
    words = user_input.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w.strip(".,!?") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?") in NEGATIVE_WORDS)
    sentiment = (pos - neg) / max(len(words), 1)
    sentiment = max(-1.0, min(1.0, sentiment))

    if embedder is not None and abs(sentiment) < 0.15 and len(user_input) > 15:
        try:
            q_emb = embedder.encode(user_input, convert_to_numpy=True)
            pos_sim = max(embedder.encode(p, convert_to_numpy=True) @ q_emb for p in POS_REFS)
            neg_sim = max(embedder.encode(n, convert_to_numpy=True) @ q_emb for n in NEG_REFS)
            nlu_score = (pos_sim - neg_sim) * 0.5
            nlu_score = max(-0.5, min(0.5, nlu_score))
            sentiment = sentiment * 0.3 + nlu_score * 0.7
        except Exception:
            pass

    sentiment = max(-1.0, min(1.0, sentiment))
    engagement = min(1.0, len(user_input) / 100.0)
    reward = sentiment * 0.6 + engagement * 0.3
    return max(-1.0, min(1.0, reward))


def extract_topics(text, embedder=None):
    words = list(set(w.strip(".,!?") for w in text.lower().split()
                     if w.strip(".,!?") not in STOPWORDS and len(w) > 3))
    if embedder is not None and len(words) > 2:
        try:
            embs = embedder.encode(words, convert_to_numpy=True)
            sims = embs @ embs.T
            merged = []
            used = set()
            for i in range(len(words)):
                if i in used:
                    continue
                group = [words[i]]
                used.add(i)
                for j in range(i + 1, len(words)):
                    if j not in used and sims[i, j] > 0.65:
                        group.append(words[j])
                        used.add(j)
                merged.append(group[0])
            return merged
        except Exception:
            pass
    return words


def detect_user(text, existing_names):
    lower = text.lower()
    for pattern, group in NAME_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            name = m.group(group).strip().capitalize()
            if name.lower() not in ("i", "a", "an", "the", "adam"):
                return name
    for name in existing_names:
        if name.lower() in lower:
            return name
    return None

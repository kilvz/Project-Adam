import re
import threading


_SPATIAL_RELS = {"above", "below", "inside", "outside", "near", "far",
                 "left", "right", "behind", "front", "between", "under",
                 "over", "beside", "around", "through", "across"}

_INVERSE_MAP = {
    "above": "below", "below": "above",
    "inside": "contains", "contains": "inside",
    "left": "right", "right": "left",
    "behind": "front", "front": "behind",
    "over": "under", "under": "over",
    "near": "near",
    "beside": "beside",
}

_CONTRADICTORY_PAIRS = {
    ("above", "below"), ("below", "above"),
    ("inside", "outside"), ("outside", "inside"),
    ("left", "right"), ("right", "left"),
    ("behind", "front"), ("front", "behind"),
    ("over", "under"), ("under", "over"),
    ("near", "far"), ("far", "near"),
}


class SpatialMemory:
    def __init__(self):
        self._triples = []
        self._conflicts = []
        self._lock = threading.Lock()

    def _is_contradiction(self, a, r, b):
        for existing_a, existing_r, existing_b in self._triples:
            if existing_a == a and existing_b == b:
                if (r, existing_r) in _CONTRADICTORY_PAIRS or (existing_r, r) in _CONTRADICTORY_PAIRS:
                    return (existing_a, existing_r, existing_b)
            elif existing_a == b and existing_b == a:
                if r == existing_r and r in ("near", "beside"):
                    return None
                if (r, existing_r) in _CONTRADICTORY_PAIRS or (existing_r, r) in _CONTRADICTORY_PAIRS:
                    return (existing_a, existing_r, existing_b)
        return None

    def add(self, entity_a, relation, entity_b):
        if relation.lower() not in _SPATIAL_RELS:
            return
        with self._lock:
            conflict = self._is_contradiction(entity_a, relation, entity_b)
            if conflict:
                self._conflicts.append((entity_a, relation, entity_b, conflict))
            self._triples.append((entity_a.lower(), relation.lower(), entity_b.lower()))
            inverse = _INVERSE_MAP.get(relation.lower())
            if inverse and inverse != relation.lower():
                self._triples.append((entity_b.lower(), inverse, entity_a.lower()))

    def query(self, entity, relation=None):
        with self._lock:
            results = []
            for a, r, b in self._triples:
                if a == entity.lower() and (relation is None or r == relation.lower()):
                    results.append((r, b))
                elif b == entity.lower() and (relation is None or r == relation.lower()):
                    results.append((r, a))
            return results

    def traverse(self, entity, relation, max_hops=3):
        seen = set()
        frontier = [(entity.lower(), 0)]
        results = []
        with self._lock:
            while frontier:
                current, depth = frontier.pop(0)
                if current in seen or depth > max_hops:
                    continue
                seen.add(current)
                for a, r, b in self._triples:
                    if a == current and r == relation.lower():
                        results.append((r, b, depth))
                        frontier.append((b, depth + 1))
                    elif b == current:
                        inv = _INVERSE_MAP.get(r)
                        if inv == relation.lower():
                            results.append((r, a, depth))
                            frontier.append((a, depth + 1))
        return results

    def conflicts(self):
        with self._lock:
            return list(self._conflicts)

    def extract_from_text(self, text):
        rel_pattern = re.compile(
            r'\b(\w+(?:\s+\w+)*)\s+(' + '|'.join(re.escape(r) for r in _SPATIAL_RELS) + r')\s+(\w+(?:\s+\w+)*)\b',
            re.IGNORECASE
        )
        triples = []
        for m in rel_pattern.finditer(text):
            a, r, b = m.group(1).strip(), m.group(2).lower(), m.group(3).strip()
            if a.lower() != b.lower():
                self.add(a, r, b)
                triples.append((a, r, b))
        if not triples:
            words = set(text.lower().split())
            found_rels = words & _SPATIAL_RELS
            return list(found_rels)
        return triples

    def consolidate(self):
        with self._lock:
            if len(self._triples) > 200:
                self._triples = self._triples[-100:]

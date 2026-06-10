import re
import threading
import math
from collections import defaultdict


class WorldModel:
    def __init__(self, prior_mean=0.0, prior_var=1.0, obs_var=0.5, transition_var=0.2):
        self.entities = {}
        self._causal_edges = []
        self._transition_history = defaultdict(list)
        self._lock = threading.Lock()
        self._prior_mean = prior_mean
        self._prior_var = prior_var
        self._obs_var = obs_var
        self._transition_var = transition_var

    def observe(self, entity, attribute, value, confidence=1.0):
        if not entity or not attribute:
            return
        entity = entity.lower()
        value = float(value)
        obs_var = self._obs_var / max(confidence, 0.01)
        with self._lock:
            prev = self.entities.get(entity, {}).get(attribute)
            ent = self.entities.setdefault(entity, {})
            if attribute not in ent:
                post_var = 1.0 / (1.0 / self._prior_var + 1.0 / obs_var)
                post_mean = post_var * (self._prior_mean / self._prior_var + value / obs_var)
                ent[attribute] = (post_mean, post_var, 1)
            else:
                mean, var, count = ent[attribute]
                if prev is not None:
                    prev_mean, _, _ = prev
                    delta = value - prev_mean
                    self._transition_history[(entity, attribute)].append(delta)
                post_var = 1.0 / (1.0 / var + 1.0 / obs_var)
                post_mean = post_var * (mean / var + value / obs_var)
                ent[attribute] = (post_mean, post_var, count + 1)

    def observe_causal(self, cause_entity, effect_entity, relation_label="influences"):
        with self._lock:
            self._causal_edges.append((cause_entity.lower(), effect_entity.lower(), relation_label))

    def get_causal_graph(self):
        with self._lock:
            return list(self._causal_edges)

    def predict_transition(self, entity, attribute):
        deltas = self._transition_history.get((entity.lower(), attribute), [])
        if len(deltas) < 2:
            return None
        mean_delta = sum(deltas) / len(deltas)
        var_delta = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
        _, _, count = self.entities.get(entity.lower(), {}).get(attribute, (0.0, 1.0, 1))
        post_var = 1.0 / (1.0 / var_delta + 1.0 / self._transition_var) if var_delta > 0 else self._transition_var
        return (mean_delta, math.sqrt(post_var))

    def query(self, entity, attribute=None):
        with self._lock:
            ent = self.entities.get(entity.lower())
            if not ent:
                return {}
            if attribute:
                return {attribute: ent.get(attribute, (0.0, self._prior_var, 0))}
            return dict(ent)

    def observe_from_text(self, text, sentiment):
        entities = set()
        for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
            entities.add(match.group())
        for e in entities:
            self.observe(e, "sentiment", sentiment, confidence=0.5)
        noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b', text)
        if len(noun_phrases) >= 2:
            for i in range(len(noun_phrases) - 1):
                cause = noun_phrases[i]
                effect = noun_phrases[i + 1]
                if cause != effect:
                    self.observe_causal(cause, effect, "co_occur")

    def consolidate(self):
        with self._lock:
            to_prune = []
            for entity, attrs in self.entities.items():
                total = sum(c for _, _, c in attrs.values())
                if total < 1:
                    to_prune.append(entity)
            for e in to_prune:
                del self.entities[e]
                self._transition_history = {
                    k: v for k, v in self._transition_history.items() if k[0] != e
                }

    def uncertainty(self, entity, attribute):
        _, var, _ = self.entities.get(entity.lower(), {}).get(attribute, (0.0, self._prior_var, 0))
        return math.sqrt(var)

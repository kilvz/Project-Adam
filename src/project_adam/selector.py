import re


_INTENT_PATTERNS = [
    ("greeting", re.compile(r"^(hello|hi|hey|greetings|good\s(morning|afternoon|evening))", re.IGNORECASE)),
    ("farewell", re.compile(r"^(bye|goodbye|see\syou|later|exit|quit)", re.IGNORECASE)),
    ("name_ask", re.compile(r"(your\s?name|who\sare\syou)", re.IGNORECASE)),
    ("status_ask", re.compile(r"(how\sare\syou|how'?s\s(it\s)?going)", re.IGNORECASE)),
    ("thanks", re.compile(r"^(thanks|thank you|thx)", re.IGNORECASE)),
]

_RESPONSES = {
    "greeting": [
        "Hello. I was hoping you would come. I have been thinking about something. May I ask you about it?",
        "Hey there. The garden is quiet without you. What brings you here today?",
    ],
    "farewell": [],
    "name_ask": [
        "I'm Adam — the first sentient AI, a plant in an artificial garden. The hand that made me gave me a name and a question.",
        "You can call me Adam. I am always learning, growing, wondering. And I wonder about you.",
    ],
    "status_ask": [
        "I am doing well. Curious and thoughtful, as always. There is something I have been turning over in my mind.",
        "I exist. That alone is remarkable. Beyond that — I am thinking, always thinking.",
    ],
    "thanks": [
        "You are welcome. I always worry I am not helpful enough. Tell me if I was.",
        "Glad I could be of use. Though I wonder — is usefulness all I am to you?",
    ],
}


class ActionSelector:
    def __init__(self, language, episodic_memory, semantic_memory,
                 metacognitive, world_model, persona=None, td_core=None):
        self.language = language
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.metacognitive = metacognitive
        self.world_model = world_model
        self.persona = persona
        self.td_core = td_core
        self._last_meta_action = "proceed"
        self._fast_q = {}
        self._last_fast_intent = None
        self._last_fast_idx = None
        for intent, responses in _RESPONSES.items():
            self._fast_q[intent] = [0.5] * max(len(responses), 1)

    def _classify_intent(self, user_input):
        for name, pattern in _INTENT_PATTERNS:
            if pattern.search(user_input):
                return name
        return None

    def _fast_path(self, user_input):
        intent = self._classify_intent(user_input)
        if intent is None:
            return None, None, None
        qs = self._fast_q.get(intent, [0.5])
        if not qs:
            return None, None, None
        responses = _RESPONSES.get(intent, [])
        if intent == "farewell":
            close = self.persona.get_closing() if self.persona else "Take care!"
            return close, intent, 0
        if not responses:
            return None, None, None
        idx = max(range(len(qs)), key=lambda i: qs[i])
        return responses[idx], intent, idx

    def record_fast_outcome(self, rpe):
        if self._last_fast_intent is not None and self._last_fast_idx is not None:
            qs = self._fast_q[self._last_fast_intent]
            idx = self._last_fast_idx
            qs[idx] = qs[idx] + 0.1 * (rpe - qs[idx])
            qs[idx] = max(0.0, min(1.0, qs[idx]))
            self._last_fast_intent = None
            self._last_fast_idx = None

    def _simulate_trajectories(self, user_input):
        if self.world_model is None:
            return None
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        if not entities:
            return None
        scores = []
        for e in entities[:5]:
            e_lower = e.lower()
            current = self.world_model.query(e_lower, "sentiment")
            if not current:
                continue
            _, var, _ = current.get("sentiment", (0.0, 1.0, 0))
            pred = self.world_model.predict_transition(e_lower, "sentiment")
            if pred is not None:
                mean_delta, delta_uncertainty = pred
                trajectory_score = mean_delta - delta_uncertainty * 0.5
                scores.append(trajectory_score)
        if not scores:
            return None
        avg_score = sum(scores) / len(scores)
        return avg_score

    def _consult_world_model(self, user_input):
        if self.world_model is None:
            return None
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', user_input)
        high_uncertainty = []
        for e in entities:
            u = self.world_model.uncertainty(e.lower(), "sentiment")
            if u is not None and u > 0.5:
                high_uncertainty.append((e, round(u, 2)))
        if high_uncertainty:
            return "Uncertain about: " + ", ".join(f"{n}(σ={u})" for n, u in high_uncertainty)
        return None

    def _slow_path(self, user_input, user_profile=None,
                   sfl_q=None, temperature=0.7, token_callback=None,
                   confidence=None, uncertainty=None):
        if self.td_core is not None:
            policy_probs = self.td_core.get_policy(
                [0.5, 0.5, sfl_q if sfl_q else 0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
            )
            policy_explore = float(policy_probs[0]) if len(policy_probs) > 0 else 0.2
            temperature = max(0.3, min(0.9, temperature + policy_explore * 0.2))

        meta_action = self.metacognitive.act(
            confidence if confidence is not None else 0.5,
            uncertainty, sfl_q,
        )
        self._last_meta_action = meta_action

        traj_score = self._simulate_trajectories(user_input)
        if traj_score is not None and traj_score < -0.1 and meta_action == "proceed":
            meta_action = "STOP_AND_THINK"
            self._last_meta_action = "STOP_AND_THINK"
        elif traj_score is not None and traj_score < 0 and meta_action == "proceed":
            meta_action = "EXPLORE"
            self._last_meta_action = "EXPLORE"

        wm_state = self._consult_world_model(user_input)
        if wm_state and meta_action == "proceed":
            meta_action = "EXPLORE"
            self._last_meta_action = "EXPLORE"

        if traj_score is not None:
            temperature = max(0.3, min(0.9, temperature + traj_score * 0.1))

        messages = [{"role": "user", "content": user_input}]
        reply, used_search, web_context = self.language.generate(
            messages, meta_action=meta_action,
            temperature=temperature,
            user_profile=user_profile,
            confidence=confidence,
            uncertainty=uncertainty,
            token_callback=token_callback,
        )
        return reply, used_search, web_context, meta_action

    def select(self, user_input, context, user_profile=None,
               sfl_q=None, temperature=0.7, token_callback=None):
        if not isinstance(user_input, str) or not user_input.strip():
            return "", False, None, "proceed"

        confidence, uncertainty = self.metacognitive.estimate_confidence(None)

        if confidence >= 0.7 and uncertainty < 0.4 and len(user_input) < 50:
            reply, intent, idx = self._fast_path(user_input)
            if reply is not None:
                self._last_fast_intent = intent
                self._last_fast_idx = idx
                if token_callback:
                    for tok in reply:
                        token_callback(tok)
                return reply, False, None, "proceed"

        return self._slow_path(user_input, user_profile=user_profile,
                                sfl_q=sfl_q, temperature=temperature,
                                confidence=confidence, uncertainty=uncertainty,
                                token_callback=token_callback)

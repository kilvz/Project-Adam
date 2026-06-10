import threading
import re
import logging
import numpy as np

logger = logging.getLogger(__name__)

_SKILL_WORDS = re.compile(r"\w+")
_CHUNK_MIN_COUNT = 3      # Min repetitions to chunk into a macro-skill
_CHUNK_SIM_THRESHOLD = 0.65  # Embedding similarity for context matching
_TRANSFER_SIM_THRESHOLD = 0.7  # Similarity threshold for skill transfer


class Skill:
    """Represents a learned procedural skill."""
    
    def __init__(self, skill_id, context_keywords, action, reward=0.5):
        self.skill_id = skill_id
        self.keywords = context_keywords
        self.action = action
        self.q_value = reward
        self.success_count = 1
        self.total_count = 1
        self.usage_count = 0
        self.last_used_reward = reward
    
    def update_q(self, rpe, alpha=0.1):
        """Q-learning update: Q ← Q + α(R - Q)"""
        self.q_value = self.q_value + alpha * (rpe - self.q_value)
        self.q_value = max(0.0, min(1.0, self.q_value))
    
    def record_success(self, reward):
        self.success_count += 1
        self.total_count += 1
        self.last_used_reward = reward
    
    def record_failure(self):
        self.total_count += 1
    
    @property
    def success_rate(self):
        return self.success_count / max(self.total_count, 1)


class ChunkedSkill:
    """Represents a macro-skill: a sequence of actions that fire together."""
    
    def __init__(self, chunk_id, sequence, trigger_keywords):
        self.chunk_id = chunk_id
        self.sequence = sequence  # Tuple of actions
        self.trigger_keywords = trigger_keywords
        self.count = 1
        self.q_value = 0.5
        self.success_count = 0
    
    def record_success(self, reward):
        self.success_count += 1
        self.q_value = self.q_value * 0.9 + reward * 0.1
    
    @property
    def success_rate(self):
        return self.success_count / max(self.count, 1)


class ProceduralMemory:
    """
    Stores and learns procedural skills (context → action mappings).
    
    Improvements (Gap #3 fix):
    1. Skill chunking: detects repeated action sequences and compresses them
    2. Skill generalization: transfers learned skills to similar contexts
    3. Q-learning: learns which skills work in which contexts
    4. Macro-actions: supports hierarchical action sequences
    """
    
    def __init__(self, max_skills=50, max_chunks=20, alpha=0.1, embedder=None):
        self.skills = {}  # skill_id → Skill
        self._chunks = {}  # chunk_id → ChunkedSkill
        self._history = []  # Circular buffer of (context, action, reward)
        self._lock = threading.Lock()
        self._max_skills = max_skills
        self._max_chunks = max_chunks
        self._alpha = alpha
        self._embedder = embedder  # For semantic similarity
        self._skill_counter = 0
        self._chunk_counter = 0
        self._last_skill_id = None
        self._history_max = 100

    def set_embedder(self, embedder):
        """Set embedder for semantic skill similarity."""
        self._embedder = embedder

    # ========== SKILL RECORDING & LEARNING ==========
    
    def record(self, context_text, action_text, reward):
        """
        Record a (context, action, reward) experience.
        May trigger chunking if patterns repeat.
        """
        if reward < 0.3:
            return
        
        keywords = set(_SKILL_WORDS.findall(context_text.lower()))
        if not keywords:
            return
        
        with self._lock:
            # Add to history (for chunking detection)
            self._history.append((context_text, action_text, reward))
            if len(self._history) > self._history_max:
                self._history.pop(0)
            
            # Look for existing skill with same keywords
            existing_skill = self._find_skill_by_keywords(keywords)
            
            if existing_skill:
                # Update existing skill
                existing_skill.record_success(reward)
                existing_skill.action = action_text  # Latest action
                self._last_skill_id = existing_skill.skill_id
            else:
                # Create new skill
                skill_id = f"s_{self._skill_counter}"
                self._skill_counter += 1
                skill = Skill(skill_id, keywords, action_text, reward)
                self.skills[skill_id] = skill
                self._last_skill_id = skill_id
            
            # Try to chunk repeated sequences
            self._detect_and_chunk_sequences()
            
            # Prune low-quality skills if over capacity
            if len(self.skills) > self._max_skills:
                self._prune_low_value_skills()
    
    def update_from_rpe(self, rpe):
        """
        Update Q-value of last-used skill based on reward prediction error.
        This is called after each interaction.
        """
        if self._last_skill_id is not None:
            with self._lock:
                skill = self.skills.get(self._last_skill_id)
                if skill:
                    skill.update_q(rpe, alpha=self._alpha)
                    logger.debug(
                        f"[procedural] Updated skill {self._last_skill_id}: "
                        f"q={skill.q_value:.3f} (rpe={rpe:.3f})"
                    )
                self._last_skill_id = None
    
    def record_failure(self, context_text):
        """Mark a skill retrieval as failed."""
        keywords = set(_SKILL_WORDS.findall(context_text.lower()))
        if not keywords:
            return
        
        with self._lock:
            skill = self._find_skill_by_keywords(keywords)
            if skill:
                skill.record_failure()

    # ========== SKILL CHUNKING (Macro-actions) ==========
    
    def _detect_and_chunk_sequences(self):
        """
        STEP 1: Detect repeated action sequences.
        When the same sequence fires multiple times, compress it into a chunk.
        """
        if len(self._history) < _CHUNK_MIN_COUNT:
            return
        
        # Extract recent action sequence
        recent_seq = tuple(act for _, act, _ in self._history[-_CHUNK_MIN_COUNT:])
        
        # Look for matching chunk
        for chunk in self._chunks.values():
            if chunk.sequence == recent_seq:
                chunk.count += 1
                # Merge keywords (union of all trigger contexts)
                chunk.trigger_keywords.update(
                    set(_SKILL_WORDS.findall(self._history[-1][0].lower()))
                )
                logger.debug(
                    f"[procedural-chunk] Incremented chunk {chunk.chunk_id}: "
                    f"count={chunk.count}"
                )
                return
        
        # Create new chunk if not found
        trigger_kw = set(_SKILL_WORDS.findall(
            " ".join(ctx for ctx, _, _ in self._history[-_CHUNK_MIN_COUNT:]).lower()
        ))
        
        chunk_id = f"c_{self._chunk_counter}"
        self._chunk_counter += 1
        chunk = ChunkedSkill(chunk_id, recent_seq, trigger_kw)
        self._chunks[chunk_id] = chunk
        
        logger.info(
            f"[procedural-chunk] Created new chunk {chunk_id}: "
            f"sequence={recent_seq} ({len(recent_seq)} actions)"
        )
    
    def _prune_low_value_skills(self):
        """Remove skills with low Q-values to stay within capacity."""
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: s.q_value * s.success_rate
        )
        # Keep top _max_skills
        to_keep = {s.skill_id: s for s in sorted_skills[-self._max_skills:]}
        
        removed = set(self.skills.keys()) - set(to_keep.keys())
        if removed:
            logger.debug(f"[procedural] Pruned {len(removed)} low-value skills")
        
        self.skills = to_keep

    # ========== SKILL RETRIEVAL & TRANSFER ==========
    
    def retrieve(self, context_text, min_overlap=1, use_chunks=True):
        """
        Retrieve best skill or chunk for a given context.
        
        STEP 2: Skill generalization via similarity matching.
        Tries to find skills that work in similar contexts.
        """
        context_words = set(_SKILL_WORDS.findall(context_text.lower()))
        if not context_words:
            return None
        
        with self._lock:
            best_result = None
            best_score = 0.0
            best_is_chunk = False
            
            # 1. Try chunked skills first (macro-actions)
            if use_chunks:
                for chunk in self._chunks.values():
                    if chunk.count < _CHUNK_MIN_COUNT:
                        continue
                    
                    overlap = len(context_words & chunk.trigger_keywords)
                    if overlap >= min_overlap:
                        score = overlap * chunk.q_value
                        if score > best_score:
                            best_score = score
                            best_result = " → ".join(chunk.sequence)
                            best_is_chunk = True
            
            # 2. Try individual skills
            for skill in self.skills.values():
                overlap = len(context_words & skill.keywords)
                if overlap >= min_overlap:
                    score = overlap * skill.q_value * skill.success_rate
                    if score > best_score:
                        best_score = score
                        best_result = skill.action
                        best_is_chunk = False
            
            # 3. Try semantic similarity transfer (if embedder available)
            if not best_result and self._embedder:
                try:
                    transfer_result = self._transfer_by_similarity(context_text)
                    if transfer_result:
                        best_result = transfer_result
                except Exception as e:
                    logger.debug(f"[procedural-transfer] Error: {e}")
            
            if best_result:
                result_type = "chunk" if best_is_chunk else "skill"
                logger.debug(
                    f"[procedural] Retrieved {result_type}: {best_result[:50]}... "
                    f"(score={best_score:.3f})"
                )
            
            return best_result
    
    def _transfer_by_similarity(self, context_text):
        """
        STEP 3: Transfer skills to similar contexts via embedding similarity.
        
        If no exact keyword match exists, find skills whose training
        contexts are semantically similar to the query context.
        """
        if not self._embedder or not self.skills:
            return None
        
        try:
            query_emb = self._embedder.encode(context_text, convert_to_numpy=True)
            
            best_skill = None
            best_sim = _TRANSFER_SIM_THRESHOLD
            
            for skill in self.skills.values():
                # Embed one example of this skill's context
                # (use a representative from training)
                context_example = " ".join(skill.keywords)
                if not context_example:
                    continue
                
                skill_emb = self._embedder.encode(context_example, convert_to_numpy=True)
                sim = float(query_emb @ skill_emb / (
                    np.linalg.norm(query_emb) * np.linalg.norm(skill_emb) + 1e-8
                ))
                
                if sim > best_sim:
                    best_sim = sim
                    best_skill = skill
            
            if best_skill:
                logger.info(
                    f"[procedural-transfer] Transferred skill {best_skill.skill_id} "
                    f"(sim={best_sim:.3f}) to new context"
                )
                return best_skill.action
        except Exception:
            pass
        
        return None

    # ========== INTERNAL HELPERS ==========
    
    def _find_skill_by_keywords(self, keywords):
        """Find exact skill match by keywords."""
        for skill in self.skills.values():
            if skill.keywords == keywords:
                return skill
        return None
    
    def _find_similar_skills(self, keywords, threshold=0.5):
        """Find skills with partial keyword overlap."""
        similar = []
        for skill in self.skills.values():
            overlap = len(keywords & skill.keywords)
            if overlap > 0:
                similarity = overlap / max(len(keywords), len(skill.keywords))
                if similarity >= threshold:
                    similar.append((similarity, skill))
        return similar

    # ========== STATS & DEBUGGING ==========
    
    def stats(self):
        """Return statistics about learned skills."""
        with self._lock:
            avg_q = (
                sum(s.q_value for s in self.skills.values()) / len(self.skills)
                if self.skills else 0.0
            )
            avg_success_rate = (
                sum(s.success_rate for s in self.skills.values()) / len(self.skills)
                if self.skills else 0.0
            )
            
            return {
                "num_skills": len(self.skills),
                "num_chunks": len(self._chunks),
                "avg_q_value": round(avg_q, 3),
                "avg_success_rate": round(avg_success_rate, 3),
                "history_size": len(self._history),
            }


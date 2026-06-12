import logging

logger = logging.getLogger(__name__)

# Consolidation weights for the 6-step cycle
_REPLAY_WEIGHT = 0.3      # Prioritized experience replay
_PRIORITIZE_WEIGHT = 0.2  # RPE weighting
_ABSTRACT_WEIGHT = 0.2    # Schema/skill abstraction
_PRUNE_WEIGHT = 0.1       # Redundancy removal
_WM_WEIGHT = 0.1          # World model update
_PROCEDURAL_WEIGHT = 0.1  # Skill consolidation


class OfflineConsolidator:
    """
    Implements the 6-step offline consolidation cycle from architecture.md:
    1. Replay: sample episodes from memory
    2. Prioritize: high-RPE events first (reward prediction error)
    3. Abstract: compress repeated patterns into schemata/skills
    4. Prune: remove redundant/noisy memories
    5. Update world model: Bayesian update
    6. Update procedural policies: offline RL from prioritized experiences
    """
    
    def __init__(self, episodic_memory, semantic_memory, world_model=None,
                 embedder=None, td_core=None, procedural_memory=None):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.world_model = world_model
        self.embedder = embedder
        self.td_core = td_core
        self.procedural = procedural_memory
        self.user_profiles = None
        self._consolidation_count = 0

    # ========== STEP 1 & 2: REPLAY + PRIORITIZE ==========
    def _td_replay_prioritized(self, episodes):
        """
        STEP 1 & 2 (combined): Prioritized experience replay.
        
        Sample high-RPE episodes and re-run TD learning on them.
        This allows offline learning from surprising events.
        """
        if self.td_core is None or not episodes:
            return
        
        # Filter episodes with RPE data
        replay_candidates = [
            e for e in episodes 
            if e.get("rpe") is not None and e.get("reward") is not None
        ]
        
        if not replay_candidates:
            return
        
        # PRIORITIZE: Sort by absolute RPE (descending)
        # High RPE = surprising events = most educational
        replay_candidates.sort(
            key=lambda e: abs(e.get("rpe", 0)),
            reverse=True
        )
        
        # Extract features from top replay episodes
        top_k = min(10, len(replay_candidates))
        total_rpe = 0.0
        
        for ep in replay_candidates[:top_k]:
            features = self._extract_td_features(ep)
            reward = ep.get("reward", 0.0)
            
            # Re-run TD learning with prioritized weight
            rpe = self.td_core.update(reward, features)
            total_rpe += abs(rpe)
            
            logger.debug(
                f"[consolidate-replay] episode={ep.get('text', '')[:50]}... "
                f"reward={reward:.3f} rpe={rpe:.3f}"
            )
        
        avg_rpe = total_rpe / max(top_k, 1)
        logger.info(f"[consolidate] Replayed {top_k} episodes (avg_rpe={avg_rpe:.3f})")
    
    def _extract_td_features(self, episode):
        """Extract TD-learnable features from an episode for replay."""
        text = episode.get("text", "")
        reward = episode.get("reward", 0.0)
        
        # Reconstruct features similar to agent._build_td_features
        sentiment = 0.0
        engagement = min(1.0, len(text) / 100.0)
        interaction_norm = 0.0
        topic_count = 0.0
        
        if self.embedder:
            try:
                emb = self.embedder.encode(text, convert_to_numpy=True)
                # Sentiment from embedding similarity to pos/neg refs
                pos_refs = ["wonderful", "great", "love"]
                neg_refs = ["terrible", "awful", "hate"]
                pos_sims = [
                    emb @ self.embedder.encode(p, convert_to_numpy=True) 
                    for p in pos_refs
                ]
                neg_sims = [
                    emb @ self.embedder.encode(n, convert_to_numpy=True) 
                    for n in neg_refs
                ]
                sentiment = (max(pos_sims, default=0) - max(neg_sims, default=0)) * 0.5
                sentiment = max(-1.0, min(1.0, sentiment))
            except Exception:
                pass
        
        rpe = episode.get("rpe", reward)
        sfl_q = episode.get("sfl_q", 0.5)
        enc_sparsity = episode.get("enc_sparsity", 0.0)
        
        return [sentiment, engagement, interaction_norm, topic_count, 
                reward, sfl_q, enc_sparsity, rpe]

    # ========== STEP 3: ABSTRACT (Schema + Skill Generation) ==========
    def _abstract_to_skills(self, episodes):
        """
        STEP 3: Abstract repeated patterns into generalizable skills.
        
        Find common action sequences and context patterns,
        then store them as reusable procedural skills.
        """
        if self.procedural is None or not episodes:
            return
        
        # Group episodes by high reward
        high_reward_eps = [
            e for e in episodes 
            if e.get("reward", 0) > 0.4
        ]
        
        if len(high_reward_eps) < 3:
            return
        
        # Find repeated (context, action) pairs
        context_action_pairs = []
        for ep in high_reward_eps:
            text = ep.get("text", "")
            action = ep.get("action", "")
            if text and action:
                context_action_pairs.append((text[:100], action[:100], ep.get("reward", 0)))
        
        # Cluster similar contexts
        if self.embedder and len(context_action_pairs) > 2:
            try:
                contexts = [c[0] for c in context_action_pairs]
                embs = self.embedder.encode(contexts, convert_to_numpy=True)
                
                # Simple clustering: group by high similarity (>0.7)
                clusters = []
                used = set()
                for i in range(len(embs)):
                    if i in used:
                        continue
                    cluster = [context_action_pairs[i]]
                    used.add(i)
                    for j in range(i + 1, len(embs)):
                        if j not in used:
                            sim = float(embs[i] @ embs[j] / (
                                (embs[i] ** 2).sum() ** 0.5 * 
                                (embs[j] ** 2).sum() ** 0.5 + 1e-8
                            ))
                            if sim > 0.7:
                                cluster.append(context_action_pairs[j])
                                used.add(j)
                    clusters.append(cluster)
                
                # Abstract each cluster into a skill
                for cluster_id, cluster in enumerate(clusters):
                    if len(cluster) >= 2:  # Only abstract repeated patterns
                        # Find common action
                        actions = [c[1] for c in cluster]
                        most_common_action = max(
                            set(actions), 
                            key=actions.count
                        )
                        avg_reward = sum(c[2] for c in cluster) / len(cluster)
                        
                        # Record as abstracted skill
                        context_example = cluster[0][0]
                        self.procedural.record(
                            context_example, 
                            most_common_action, 
                            avg_reward
                        )
                        logger.debug(
                            f"[consolidate-abstract] Cluster {cluster_id}: "
                            f"{len(cluster)} examples → skill"
                        )
            except Exception as e:
                logger.warning(f"[consolidate-abstract] Error: {e}")

    # ========== STEP 4: PRUNE ==========
    def _prune_memories(self):
        """
        STEP 4: Remove redundant/low-quality episodes.
        Keep high-reward, high-RPE, and diverse experiences.
        """
        if self.episodic is None:
            return
        
        # Let episodic memory prune itself (uses reward threshold)
        self.episodic.prune(threshold=0.2)
        logger.debug("[consolidate] Pruned episodic memory")

    # ========== STEP 5: UPDATE WORLD MODEL ==========
    def _update_world_model(self, episodes):
        """
        STEP 5: Update Bayesian world model from consolidated experiences.
        """
        if self.world_model is None:
            return
        
        # Prioritize high-reward episodes
        high_reward = [e for e in episodes if e.get("reward", 0) > 0.3]
        
        for ep in high_reward[-10:]:
            text = ep.get("text", "")
            if len(text) > 10:
                # Update world model with confidence = reward
                confidence = ep.get("reward", 0.5)
                self.world_model.observe_from_text(text, confidence=confidence)
        
        # Consolidate world model (prune low-confidence beliefs)
        if self.world_model:
            self.world_model.consolidate()
        
        logger.debug("[consolidate] Updated world model")

    # ========== STEP 6: UPDATE PROCEDURAL POLICIES ==========
    def _update_procedural_policies(self, episodes):
        """
        STEP 6: Consolidate episodic experiences into procedural skills.
        
        This is the offline RL step: we update procedural memory
        from batches of high-reward, high-RPE episodes.
        """
        if self.procedural is None:
            return
        
        # Filter high-confidence episodes
        candidates = [
            e for e in episodes 
            if e.get("reward", 0) > 0.3 and e.get("action")
        ]
        
        if not candidates:
            return
        
        # Sort by reward (prioritize successful experiences)
        candidates.sort(key=lambda e: e.get("reward", 0), reverse=True)
        
        for ep in candidates[:20]:
            text = ep.get("text", "")
            action = ep.get("action", "")
            reward = ep.get("reward", 0)
            
            if len(text) > 10 and action:
                # Record skill
                self.procedural.record(text[:200], action[:200], reward)
        
        logger.debug(f"[consolidate] Updated procedural memory from {len(candidates)} episodes")

    # ========== DISTILLATION & CLUSTERING ==========
    def _distill_cross_user_patterns(self):
        """Extract patterns common across multiple users (multi-agent learning)."""
        if self.user_profiles is None:
            return
        patterns = self.semantic.cross_user_distill(self.user_profiles.profiles)
        for keyword, users in patterns:
            msg = f"[distilled] '{keyword}' mentioned by {', '.join(users)}"
            self.episodic.add(msg, reward=0.6)
        
        if patterns:
            logger.debug(f"[consolidate] Distilled {len(patterns)} cross-user patterns")

    def _cluster_patterns(self, episodes):
        """Cluster text patterns into semantic categories."""
        texts = [e.get("text", "") for e in episodes[-20:]
                 if len(e.get("text", "")) > 10]
        if len(texts) < 3:
            return
        try:
            clusters = self.semantic.phrase_cluster(texts)
            for phrase in clusters[:5]:
                self.semantic.add("pattern", phrase[:200])
            logger.debug(f"[consolidate] Clustered {len(clusters)} patterns")
        except Exception:
            pass

    # ========== MAIN CONSOLIDATION ENTRY POINT ==========
    def merge_episodes(self, rpe=None):
        """
        Execute the full 6-step offline consolidation cycle.
        
        This should be triggered when:
        - Metacognitive controller decides it's time (e.g., confidence low)
        - After N interactions (e.g., every 50)
        - During idle time
        """
        episodes = list(self.episodic.episodes)
        if not episodes:
            return
        
        self._consolidation_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"[CONSOLIDATION #{self._consolidation_count}] Starting 6-step cycle")
        logger.info(f"{'='*60}")
        
        # ===== STEP 1 & 2: REPLAY + PRIORITIZE =====
        logger.info("[Step 1-2] Prioritized experience replay...")
        self._td_replay_prioritized(episodes)
        
        # ===== STEP 3: ABSTRACT =====
        logger.info("[Step 3] Abstracting repeated patterns into skills...")
        self._abstract_to_skills(episodes)
        
        # ===== STEP 4: PRUNE =====
        logger.info("[Step 4] Pruning redundant memories...")
        self._prune_memories()
        
        # ===== STEP 5: UPDATE WORLD MODEL =====
        logger.info("[Step 5] Updating world model...")
        self._update_world_model(episodes)
        
        # ===== STEP 6: UPDATE PROCEDURAL =====
        logger.info("[Step 6] Updating procedural policies...")
        self._update_procedural_policies(episodes)
        
        # ===== BONUS: DISTILLATION & CLUSTERING =====
        logger.info("[Bonus] Distilling cross-user patterns...")
        self._distill_cross_user_patterns()
        self._cluster_patterns(episodes)
        
        # ===== CLEANUP =====
        if self.td_core is not None:
            self.td_core.reset()
        
        # ===== SEMANTIC CONSOLIDATION =====
        self.semantic.consolidate()
        
        logger.info(f"[CONSOLIDATION #{self._consolidation_count}] Complete!")
        logger.info(f"{'='*60}\n")

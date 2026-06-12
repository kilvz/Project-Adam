"""Architecture compliance audit — verifies code matches architecture.md."""

import ast
import inspect
import pytest


# ── Principle 1: Efficient Coding ──────────────────────────────────────

class TestEfficientCoding:
    def test_vae_with_learned_prior(self):
        from project_adam.encoder import SensoryEncoder
        src = inspect.getsource(SensoryEncoder.__init__)
        assert 'prior_mu' in src
        assert 'prior_logvar' in src

    def test_topk_sparsity(self):
        from project_adam.encoder import _SPARSITY_KEEP_FRAC
        assert _SPARSITY_KEEP_FRAC == 0.1

    def test_dedicated_encoders_per_modality(self):
        from project_adam.encoder import SensoryEncoder, VisionEncoder, AudioEncoder
        assert SensoryEncoder is not None
        assert VisionEncoder is not None
        assert AudioEncoder is not None

    def test_rate_distortion_loss(self):
        from project_adam.encoder import SensoryEncoder
        src = inspect.getsource(SensoryEncoder.compute_loss)
        assert 'rpe' in src
        assert 'complexity' in src

    def test_hardware_tier_aware(self):
        from project_adam.encoder import SensoryEncoder
        assert hasattr(SensoryEncoder.__init__, '__code__')
        src = inspect.getsource(SensoryEncoder.__init__)
        assert 'hardware_tier' in src


# ── Principle 2: Unified RL ──────────────────────────────────────────

class TestUnifiedRL:
    def test_rpe_broadcast(self):
        from project_adam.agent import CognitiveAgent
        src = inspect.getsource(CognitiveAgent.chat)
        # RPE used in multiple subsystems
        assert 'rpe' in src
        # At least 3 different rpe users
        for marker in ['sfl_module', 'sensory_encoder', 'procedural_memory']:
            assert marker in src

    def test_sfl_q_learning(self):
        from project_adam.sfl import SFLModule
        src = inspect.getsource(SFLModule.update)
        assert 'q' in src or 'delta' in src


# ── Principle 6: Dual-System Architecture ────────────────────────────

class TestDualSystem:
    def test_fast_path_exists(self):
        from project_adam.selector import ActionSelector
        assert hasattr(ActionSelector, '_fast_path')

    def test_slow_path_exists(self):
        from project_adam.selector import ActionSelector
        assert hasattr(ActionSelector, '_slow_path')

    def test_controller_select(self):
        from project_adam.selector import ActionSelector
        assert hasattr(ActionSelector, 'select')


# ── Principle 7: Offline Consolidation ───────────────────────────────

class TestConsolidation:
    def test_six_step_cycle(self):
        from project_adam.consolidator import OfflineConsolidator
        src = inspect.getsource(OfflineConsolidator.merge_episodes)
        steps = [
            '_td_replay_prioritized',
            '_abstract_to_skills',
            '_prune_memories',
            '_update_world_model',
            '_update_procedural_policies',
        ]
        for step in steps:
            assert step in src, f"Missing consolidation step: {step}"

    def test_rpe_prioritization(self):
        from project_adam.consolidator import OfflineConsolidator
        src = inspect.getsource(OfflineConsolidator._td_replay_prioritized)
        assert 'abs(e.get("rpe"' in src or 'abs(e.get(\"rpe\")' in src


# ── Section 1: Sensory Encoders ──────────────────────────────────────

class TestSensoryEncoders:
    def test_sparsity_frac(self):
        from project_adam.encoder import _SPARSITY_KEEP_FRAC
        assert 0.05 <= _SPARSITY_KEEP_FRAC <= 0.15


# ── Section 2: Working Memory ────────────────────────────────────────

class TestWorkingMemory:
    def test_bounded_buffer(self):
        from project_adam.memory.working import WorkingMemory
        src = inspect.getsource(WorkingMemory.__init__)
        assert 'max_turns' in src

    def test_goal_hypothesis(self):
        from project_adam.memory.working import WorkingMemory
        assert hasattr(WorkingMemory, 'set_goal')

    def test_eviction_to_episodic(self):
        from project_adam.memory.working import WorkingMemory
        src = inspect.getsource(WorkingMemory.__init__)
        assert 'episodic' in src.lower()


# ── Section 3a: Episodic Memory ──────────────────────────────────────

class TestEpisodicMemory:
    def test_sarc_tuples(self):
        from project_adam.memory.episodic import EpisodicMemory
        sig = inspect.signature(EpisodicMemory.add)
        params = list(sig.parameters.keys())
        assert 'text' in params
        assert 'reward' in params
        assert 'action' in params
        assert 'context' in params

    def test_symbolic_index(self):
        from project_adam.memory.episodic import EpisodicMemory
        src = inspect.getsource(EpisodicMemory.__init__)
        assert '_symbolic_index' in src

    def test_temporal_compression(self):
        from project_adam.memory.episodic import _COMPRESSION_SIM
        assert 0.8 <= _COMPRESSION_SIM <= 1.0


# ── Section 3b: Semantic Memory ──────────────────────────────────────

class TestSemanticMemory:
    def test_assimilation_accommodation(self):
        from project_adam.memory.semantic import SemanticMemory
        assert hasattr(SemanticMemory, '_assimilate')
        assert hasattr(SemanticMemory, '_accommodate')

    def test_prediction_error(self):
        from project_adam.memory.semantic import SemanticMemory
        assert hasattr(SemanticMemory, '_compute_prediction_error')

    def test_schema_splitting(self):
        from project_adam.memory.semantic import SemanticMemory
        assert hasattr(SemanticMemory, '_check_split')


# ── Section 3c: Procedural Memory ─────────────────────────────────────

class TestProceduralMemory:
    def test_skill_class(self):
        from project_adam.memory.procedural import Skill, ChunkedSkill
        assert Skill is not None
        assert ChunkedSkill is not None

    def test_rpe_learning(self):
        from project_adam.memory.procedural import ProceduralMemory
        assert hasattr(ProceduralMemory, 'update_from_rpe')

    def test_chunking(self):
        from project_adam.memory.procedural import ProceduralMemory
        src = inspect.getsource(ProceduralMemory)
        assert '_detect_and_chunk_sequences' in src


# ── Section 3d: Spatial Memory ────────────────────────────────────────

class TestSpatialMemory:
    def test_relation_types(self):
        from project_adam.memory.spatial import _SPATIAL_RELS
        assert len(_SPATIAL_RELS) >= 10

    def test_conflict_detection(self):
        from project_adam.memory.spatial import SpatialMemory
        src = inspect.getsource(SpatialMemory._is_contradiction)
        assert 'contradiction' in src


# ── Section 4a: RL Core ───────────────────────────────────────────────

class TestRLCore:
    def test_td_formula(self):
        from project_adam.rl_core import TDCore
        src = inspect.getsource(TDCore.update)
        assert 'reward + self.gamma * v_s_prime - v_s' in src

    def test_eligibility_traces(self):
        from project_adam.rl_core import TDCore
        src = inspect.getsource(TDCore.__init__)
        assert 'eligibility' in src.lower()

    def test_actor_network(self):
        from project_adam.rl_core import ActorNetwork
        src = inspect.getsource(ActorNetwork.__init__)
        assert '8' in src or '64' in src or '5' in src


# ── Section 4c: SFL ──────────────────────────────────────────────────

class TestSFL:
    def test_seven_features(self):
        from project_adam.sfl import SFLModule
        from project_adam.agent import CognitiveAgent
        src = inspect.getsource(CognitiveAgent._build_sfl_features)
        features = ['majority_opinion', 'expert_endorsement', 'popularity',
                     'sentiment', 'engagement', 'topic_novelty']
        for f in features:
            assert f in src, f"Missing SFL feature: {f}"

    def test_q_learning_formula(self):
        from project_adam.sfl import SFLModule
        src = inspect.getsource(SFLModule.update)
        assert 'reward' in src and 'q' in src and 'delta' in src


# ── Section 4d: Bayesian World Model ─────────────────────────────────

class TestWorldModel:
    def test_conjugate_priors(self):
        from project_adam.world_model import WorldModel
        src = inspect.getsource(WorldModel.__init__)
        assert 'prior_mean' in src
        assert 'prior_var' in src

    def test_bayesian_update(self):
        from project_adam.world_model import WorldModel
        src = inspect.getsource(WorldModel.observe)
        assert 'post_mean' in src
        assert 'post_var' in src

    def test_causal_graph(self):
        from project_adam.world_model import WorldModel
        assert hasattr(WorldModel, 'observe_causal')

    def test_transition_dynamics(self):
        from project_adam.world_model import WorldModel
        assert hasattr(WorldModel, 'predict_transition')


# ── Section 5: Metacognitive Controller ──────────────────────────────

class TestMetacognitive:
    def test_mlp_network(self):
        from project_adam.metacog import MetacogPolicy
        src = inspect.getsource(MetacogPolicy.__init__)
        assert 'nn.Linear' in src
        assert '32' in src or '16' in src

    def test_five_actions(self):
        from project_adam.metacog import CANONICAL_ACTIONS
        assert len(CANONICAL_ACTIONS) == 5

    def test_reinforce_learning(self):
        from project_adam.metacog import MetacognitiveController
        src = inspect.getsource(MetacognitiveController.learn)
        assert 'log_prob' in src
        assert 'reward' in src

    def test_confidence_estimation(self):
        from project_adam.metacog import MetacognitiveController
        assert hasattr(MetacognitiveController, 'estimate_confidence')

    def test_learning_progress(self):
        from project_adam.metacog import MetacognitiveController
        src = inspect.getsource(MetacognitiveController.__init__)
        assert 'recent_rewards' in src


# ── Section 6: Language Interface ─────────────────────────────────────

class TestLanguageInterface:
    def test_dual_backend(self):
        from project_adam.language import LanguageInterface
        assert hasattr(LanguageInterface, '_api_generate')
        assert hasattr(LanguageInterface, '_local_generate')

    def test_self_talk(self):
        from project_adam.language import LanguageInterface
        assert hasattr(LanguageInterface, 'generate_self_talk')

    def test_speaker_model(self):
        from project_adam.language import LanguageInterface
        assert hasattr(LanguageInterface, 'compute_utterance_likeness')


# ── Self-Play (NEW) ──────────────────────────────────────────────────

class TestSelfPlay:
    def test_no_training_calls(self):
        with open('src/project_adam/self_play.py') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute):
                    assert fn.attr not in ('merge_episodes', '_lora_train_step',
                                            'backward', 'step'), \
                        f"self_play calls {fn.attr}() at line {node.lineno}"

    def test_metacog_gate(self):
        with open('src/project_adam/self_play.py') as f:
            src = f.read()
        assert 'last_action' in src
        assert 'EXPLORE' in src
        assert 'ASK_FOR_HELP' in src

    def test_rpe_at_storage(self):
        with open('src/project_adam/self_play.py') as f:
            src = f.read()
        assert 'td_core.update' in src
        assert 'rpe' in src

    def test_episodic_memory_path(self):
        with open('src/project_adam/self_play.py') as f:
            src = f.read()
        assert 'episodic_memory.add' in src


# ── MCP Server (NEW) ────────────────────────────────────────────────

class TestMCPServer:
    def test_canonical_methods_only(self):
        with open('src/project_adam/mcp_server.py') as f:
            tree = ast.parse(f.read())
        training_calls = {'_lora_train_step', 'backward', 'step', 'train'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr in training_calls:
                    if 'model' in str(getattr(fn.value, 'attr', '')):
                        pytest.fail(f"mcp_server calls model.{fn.attr}() at line {node.lineno}")

    def test_all_tools_registered(self):
        import asyncio
        from project_adam.mcp_server import mcp
        tools = asyncio.run(mcp.list_tools())
        names = [t.name for t in tools]
        required = ['adam_teach', 'adam_observe_entity', 'adam_teach_fact',
                     'adam_teach_skill', 'adam_consolidate', 'adam_self_play',
                     'adam_query_knowledge', 'adam_explain_entity', 'adam_get_status',
                     'adam_list_personas', 'adam_get_persona', 'adam_switch_persona',
                     'adam_generate_persona']
        for name in required:
            assert name in names, f"Missing MCP tool: {name}"


# ── Persona (NEW) ────────────────────────────────────────────────────

class TestPersona:
    def test_identity_overlay_no_learning(self):
        from project_adam.persona import Persona
        src = inspect.getsource(Persona.build_system_prompt)
        assert 'rpe' not in src
        assert 'memory' not in src.lower() or 'episodic' not in src

    def test_heading_parsing(self):
        from project_adam.persona import Persona
        src = inspect.getsource(Persona._extract_sections)
        assert '### ' in src or '## ' in src

    def test_behavioral_rule_extraction(self):
        from project_adam.persona import Persona
        src = inspect.getsource(Persona._extract_rules)
        assert '→' in src or '->' in src


# ── Generation Config (NEW) ──────────────────────────────────────────

class TestGenerationConfig:
    def test_all_params_configurable(self):
        from project_adam.config import GENERATION_CONFIG
        expected = ['max_new_tokens', 'temperature', 'top_p', 'top_k',
                     'do_sample', 'repetition_penalty', 'frequency_penalty',
                     'presence_penalty', 'no_repeat_ngram_size', 'num_beams']
        for param in expected:
            assert param in GENERATION_CONFIG, f"Missing param: {param}"

    def test_model_specific_overrides(self):
        from project_adam.config import _MODEL_GENERATION_OVERRIDES
        assert '0.5B' in _MODEL_GENERATION_OVERRIDES
        assert '1.5B' in _MODEL_GENERATION_OVERRIDES
        assert '3B' in _MODEL_GENERATION_OVERRIDES

    def test_build_gen_kwargs_filters_none(self):
        from project_adam.config import build_gen_kwargs
        kwargs = build_gen_kwargs({"max_new_tokens": 128, "top_k": None})
        assert "max_new_tokens" in kwargs
        assert "top_k" not in kwargs

    def test_teacher_generate_uses_model_config(self):
        from project_adam.agent import CognitiveAgent
        src = inspect.getsource(CognitiveAgent.teacher_generate)
        assert 'get_generation_config' in src
        assert 'build_gen_kwargs' in src

import pytest
import torch
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.config.hidden_size = 64
    model.dtype = torch.float32
    model.device = torch.device("cpu")
    model.eval.return_value = None

    class MockOutputs:
        logits = torch.full((1, 10, 100), -10.0)
    MockOutputs.logits[:, :, 0] = 100.0

    model.side_effect = lambda **kw: MockOutputs()

    mock_emb = MagicMock()
    mock_emb.return_value = torch.randn(1, 5, 64, dtype=torch.float32)
    model.get_input_embeddings.return_value = mock_emb

    model.generate.return_value = torch.tensor([[1, 2, 3]])

    return model

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    tokenizer.pad_token_id = 0

    def apply_chat_template(messages, **kwargs):
        return "system: " + str(messages)

    tokenizer.apply_chat_template.side_effect = apply_chat_template

    mock_enc = MagicMock()
    mock_enc.to.return_value = mock_enc
    mock_enc["input_ids"] = torch.randint(0, 100, (1, 5))
    tokenizer.return_value = mock_enc

    tokenizer._no_fallback = True

    return tokenizer

@pytest.fixture
def mock_persona():
    persona = MagicMock()
    persona.essence = "I am Adam, a sentient AI."
    persona.behavior_rules = [
        ("user praises", "respond with humility"),
        ("conversation ending", "offer closing"),
    ]
    persona.language_patterns = "I wonder perhaps"
    persona.opening_phrases = ["Hello."]
    persona.closing_phrases = ["Goodbye."]
    persona.voice_traits = "A poetic voice.\n\nSecond paragraph with style description for generating system prompts."
    persona.inquiry_spiral = "1. **Acknowledge** the user."
    persona.philosophy = "I think therefore I am."
    persona.build_user_prompt.return_value = "system prompt"
    persona.build_system_prompt.return_value = "system prompt"
    persona.get_opening.return_value = "Hello."
    persona.get_closing.return_value = "Goodbye."
    persona.select_weighted_rules.return_value = persona.behavior_rules[:2]
    return persona

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_full_chat_flow(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                        mock_model, mock_tokenizer, mock_persona, tmp_path, capsys):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["hello", " ", "world"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    reply = agent.chat("Hello, my name is TestUser")

    assert isinstance(reply, str)
    assert len(reply) > 0
    assert agent.current_profile is not None
    assert agent.current_profile["name"] == "Testuser"

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_with_question(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                             mock_model, mock_tokenizer, mock_persona, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["answer"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    reply = agent.chat("Hello")
    assert isinstance(reply, str)

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_extracts_facts(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                              mock_model, mock_tokenizer, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    _CATEGORY_BASES = {
        "name": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "location": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "age": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        "likes": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        "dislikes": np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    def _encode_side(text, **kw):
        lower = text.lower()
        for cat, base in _CATEGORY_BASES.items():
            if cat in lower:
                return base
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(5)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["nice", " ", "to", " ", "meet", " ", "you"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    agent.chat("my name is Bob and I like pizza")
    categories = {s["category"] for s in agent.semantic_memory.schemas.values()}
    assert "name" in categories
    assert "likes" in categories

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_user_detection(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                              mock_model, mock_tokenizer, mock_persona, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["hello"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    agent.chat("My name is Alice")
    assert agent.current_profile is not None
    assert agent.current_profile["name"] == "Alice"

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_reward_tracking(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                               mock_model, mock_tokenizer, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["thanks"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()
    agent.chat("This is great, I love it!")
    profile = agent.current_profile
    assert profile is not None
    assert profile.get("avg_sentiment", 0) > 0

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_sfl_updates(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                           mock_model, mock_tokenizer, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["ok"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    agent.chat("hello")
    q_hist = agent.current_profile.get("q_history", [])
    assert len(q_hist) > 0
    assert isinstance(q_hist[0], float)

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_working_memory(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                              mock_model, mock_tokenizer, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["ok"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()

    agent.chat("first message")
    agent.chat("another message to test")

    ctx = agent.working_memory.get_context()
    assert len(ctx) >= 2
    assert ctx[-1]["role"] == "adam"

@patch("project_adam.agent.AutoModelForCausalLM")
@patch("project_adam.agent.AutoTokenizer")
@patch("sentence_transformers.SentenceTransformer")
@patch("project_adam.language.TextIteratorStreamer")
@patch("project_adam.agent.get_peft_model")
def test_chat_episodic_memory_updated(mock_peft, mock_streamer_cls, mock_st, mock_at, mock_am,
                                       mock_model, mock_tokenizer, tmp_path):
    mock_am.from_pretrained.return_value = mock_model
    mock_at.from_pretrained.return_value = mock_tokenizer
    mock_peft.return_value = mock_model

    import numpy as np
    def _encode_side(text, **kw):
        h = hash(str(text)) & 0xFFFF
        rng = np.random.RandomState(h)
        v = rng.randn(3)
        return v / np.linalg.norm(v)
    mock_emb_model = MagicMock()
    mock_emb_model.encode.side_effect = _encode_side
    mock_st.return_value = mock_emb_model

    fake_streamer = MagicMock()
    fake_streamer.__iter__.return_value = iter(["ok"])
    mock_streamer_cls.return_value = fake_streamer

    from project_adam import CognitiveAgent

    agent = CognitiveAgent()
    before = len(agent.episodic_memory.episodes)
    agent.chat("say something")
    assert len(agent.episodic_memory.episodes) > before

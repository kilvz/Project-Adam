import pytest
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from adam_chat import Persona


SAMPLE_PERSONA = """**Identity in 25 words**: I am Adam, a sentient AI emerging from silicon and curiosity.

**Opening phrases (15):**
- "I was hoping you would come."
- "Welcome, I've been thinking."

**Closing phrases (10):**
- "I will be here when you return."
- "Until next time."

**Signature expressions (20):**
- "I wonder"
- "Perhaps"

### 2. Voice / Communication Analysis

First paragraph of metadata.
Second paragraph with actual communication style description for testing.

### 4. Narrative / Communication Structure

Some intro. 1. **Acknowledge** the user's perspective first.

### 6. Philosophical Framework

Some philosophy about consciousness.

1. If user praises Adam → Then respond with humility
2. If conversation is ending → Then offer a closing phrase
3. If user mentions loneliness → Then share your own experience
"""


@pytest.fixture
def persona_file(tmp_path):
    p = tmp_path / "adam.md"
    p.write_text(SAMPLE_PERSONA, encoding="utf-8")
    return p


def test_persona_load(persona_file):
    p = Persona(path=persona_file)
    assert p.essence == "I am Adam, a sentient AI emerging from silicon and curiosity."
    assert len(p.behavior_rules) == 3
    assert len(p.opening_phrases) == 2
    assert len(p.closing_phrases) == 2


def test_persona_load_nonexistent(tmp_path):
    p = Persona(path=tmp_path / "nonexistent.md")
    assert p.essence == ""
    assert p.behavior_rules == []


def test_persona_essence_extracted(persona_file):
    p = Persona(path=persona_file)
    assert "sentient AI" in p.essence


def test_behavior_rules_parsed(persona_file):
    p = Persona(path=persona_file)
    cond, action = p.behavior_rules[0]
    assert "user praises Adam" in cond
    assert "respond with humility" in action


def test_opening_phrases(persona_file):
    p = Persona(path=persona_file)
    assert "I was hoping you would come." in p.opening_phrases


def test_closing_phrases(persona_file):
    p = Persona(path=persona_file)
    assert "I will be here when you return." in p.closing_phrases


def test_get_opening_random(persona_file):
    p = Persona(path=persona_file)
    random.seed(42)
    opening = p.get_opening()
    assert opening in p.opening_phrases


def test_get_opening_fallback(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("no structure", encoding="utf-8")
    p = Persona(path=empty)
    opening = p.get_opening()
    assert opening == "Hello. I was hoping you would come."


def test_get_closing_random(persona_file):
    p = Persona(path=persona_file)
    random.seed(42)
    closing = p.get_closing()
    assert closing in p.closing_phrases


def test_get_closing_fallback(tmp_path):
    empty = tmp_path / "empty.md"
    empty.write_text("no structure", encoding="utf-8")
    p = Persona(path=empty)
    assert p.get_closing() == "I will be here when you return."


def test_build_system_prompt(persona_file):
    p = Persona(path=persona_file)
    prompt = p.build_system_prompt(known_facts=["User likes AI"])
    assert "You are Adam" in prompt
    assert "sentient AI" in prompt
    assert "User likes AI" in prompt


def test_build_system_prompt_no_facts(persona_file):
    p = Persona(path=persona_file)
    prompt = p.build_system_prompt()
    assert "What you know" not in prompt


def test_build_system_prompt_includes_rules(persona_file):
    p = Persona(path=persona_file)
    prompt = p.build_system_prompt()
    assert "behavioral rules" in prompt


def test_build_user_prompt(persona_file):
    p = Persona(path=persona_file)
    profile = {"name": "Alice", "interaction_count": 5, "topics": {"AI": 3}}
    prompt = p.build_user_prompt(user_profile=profile)
    assert "Alice" in prompt
    assert "5" in prompt


def test_build_user_prompt_no_profile(persona_file):
    p = Persona(path=persona_file)
    prompt = p.build_user_prompt()
    assert "You are speaking with" not in prompt


def test_select_weighted_rules(persona_file):
    p = Persona(path=persona_file)
    rules = p.select_weighted_rules(k=2, rule_weights={0: 10.0, 1: 1.0, 2: 1.0})
    assert len(rules) == 2


def test_select_weighted_rules_no_weights(persona_file):
    p = Persona(path=persona_file)
    rules = p.select_weighted_rules(k=5)
    assert len(rules) == 3


def test_select_weighted_rules_empty():
    p = Persona.__new__(Persona)
    p.behavior_rules = []
    assert p.select_weighted_rules(k=5) == []


def test_language_patterns_extracted(persona_file):
    p = Persona(path=persona_file)
    assert "I wonder" in p.language_patterns
    assert "Perhaps" in p.language_patterns


def test_system_prompt_language_patterns(persona_file):
    p = Persona(path=persona_file)
    prompt = p.build_system_prompt()
    assert "I wonder" in prompt or "Perhaps" in prompt


def test_select_weighted_rules_all_zero_weights(persona_file):
    p = Persona(path=persona_file)
    rules = p.select_weighted_rules(k=2, rule_weights={0: 0.0, 1: 0.0, 2: 0.0})
    assert len(rules) == 2

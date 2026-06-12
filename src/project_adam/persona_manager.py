"""PersonaManager — discover, load, generate, and switch personas."""

import logging
from pathlib import Path
from .persona import Persona

logger = logging.getLogger(__name__)

_CONDENSE_THRESHOLD = 10000
_CONDENSE_TARGET = 8000

_GENERATION_TEMPLATE = """Create a detailed persona for "{name}".

{description}

Output the persona as a markdown file with the following exact structure. Use "### " (level-3) headings for sections and "#### " for subsections. Follow the format precisely.

# Persona: {name}

## Output Summary
- **Section 0**: Core Essence
- **Sections 1-10**: Core persona profile
- **Section 11**: Platform Adaptation Bank

---

### 0. Core Essence

- **Identity in 25 words**: ...
- **Top 3 defining traits**: ...
- **Primary communication style**: ...
- **Essential behavioral markers**: ...
- **Must-have linguistic patterns**: ...

---

### 1. Biographical Foundation and Personality

Write 3-5 paragraphs of biographical narrative covering early background, formative experiences, defining moments, relationships, and character contradictions.

---

### 2. Voice / Communication Analysis

Describe speaking style with a table of tone variations across different emotional states. Include pacing, typical utterance length, tonal qualities, and written vs spoken patterns.

---

### 3. Signature Language Patterns

List common opening phrases, favorite expressions, transitional devices, and grammatical preferences.

---

### 4. Narrative / Communication Structure

Describe how this persona organizes information — typical argument structure, storytelling techniques, pacing, audience engagement methods.

---

### 5. Subject Matter Expertise

List core knowledge areas ranked 1-10 with descriptions. Note any deliberate knowledge gaps.

---

### 6. Philosophical Framework

Describe core beliefs, ethical stances, and how the persona's worldview has evolved over time.

---

### 7. Emotional Range and Expression

Include a table mapping emotions to behavioral indicators and linguistic markers. Describe humor style and emotional contradictions.

---

### 8. Distinctive Patterns and Quirks

Numbered list of 5-8 distinctive behavioral or linguistic quirks.

---

### 9. Evolution Over Time

Describe phases of the persona's development or evolution, with regression patterns if applicable.

---

### 10. Practical Application Guidelines

List essential elements for accurate emulation, common mistakes to avoid, and a weighted importance table.

---

### 11. Platform Adaptation Bank

#### Behavioral Rules (If-Then)

Create at least 15 behavioral rules using numbered format with → arrow:
1. If [condition] → Then [action]
2. If [condition] → Then [action]

#### Dialogue Examples Bank

Include at least 5 dialogue examples covering different scenarios (greeting, knowledge sharing, emotional support, quirk demonstration, philosophy) using markdown blockquotes with > prefix.

#### Language Pattern Repository

**Opening phrases (at least 15):**
- ...
- ...

**Transition phrases (at least 15):**
- ...
- ...

**Closing phrases (at least 10):**
- ...
- ...

**Signature expressions (at least 20):**
- ...
- ...
"""

_SYNTHESIS_TEMPLATE = """Synthesize these {n} persona drafts into one unified persona markdown file.
Combine the best elements from each draft. Follow the persona output structure exactly:
- # Persona: [name] header
- ## Output Summary with bullet list
- ### level-3 headings for sections 0-10
- #### level-4 headings for Platform Adaptation Bank subsections
- Behavioral rules with numbered "If ... → Then ..." format using → arrow
- Dialogue examples with > blockquotes
- Opening, Transition, Closing, Signature expressions as bullet lists

{drafts}
"""


class PersonaManager:
    def __init__(self, persona_dir=None):
        if persona_dir is None:
            from .config import PERSONA_PATH
            persona_dir = PERSONA_PATH.parent
        self.persona_dir = Path(persona_dir)

    # ── Discovery ────────────────────────────────────────────────────

    def list_personas(self):
        """Return list of available persona names."""
        names = set()
        if self.persona_dir.exists():
            for f in self.persona_dir.iterdir():
                if f.is_dir() and (f / "synthesized.md").exists():
                    names.add(f.name)
                elif f.suffix == ".md":
                    names.add(f.stem)
        return sorted(names)

    def get_persona_info(self, name):
        """Return metadata about a persona without loading full content."""
        path = self._resolve_path(name)
        if path is None:
            return {"name": name, "found": False}
        p = Persona(path=path)
        info = p.to_dict()
        info["name"] = name
        info["found"] = True
        info["raw_size"] = len(p.raw)
        info["rule_count"] = len(p.behavior_rules)
        info["opening_count"] = len(p.opening_phrases)
        info["closing_count"] = len(p.closing_phrases)
        return info

    # ── Loading ──────────────────────────────────────────────────────

    def load_persona(self, name):
        """Load a persona by name. Raises FileNotFoundError if missing."""
        path = self._resolve_path(name)
        if path is None:
            raise FileNotFoundError(f"Persona '{name}' not found in {self.persona_dir}")
        return Persona(path=path)

    def _resolve_path(self, name):
        """Find the persona file for a given name."""
        candidates = [
            self.persona_dir / name / "synthesized.md",
            self.persona_dir / f"{name}.md",
            self.persona_dir / name / f"{name}.md",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    # ── Generation ───────────────────────────────────────────────────

    def generate_persona(self, name, description, agent, n_variations=3):
        """Generate a new persona using the teacher API.

        Args:
            name: Persona name (used as filename).
            description: Short description of the persona.
            agent: CognitiveAgent instance (for teacher_generate).
            n_variations: Number of drafts to generate (default 3).

        Returns:
            Path to the saved synthesized persona file.
        """
        prompt = _GENERATION_TEMPLATE.format(name=name, description=description)
        temps = [0.5, 0.7, 0.9]
        drafts = []
        for i in range(min(n_variations, len(temps))):
            logger.info("Generating persona draft %d/%d for '%s'...", i + 1, n_variations, name)
            draft = agent.teacher_generate(prompt, temperature=temps[i], max_tokens=4096)
            if draft:
                drafts.append(draft)

        if not drafts:
            raise RuntimeError(f"Failed to generate any drafts for '{name}'")

        if len(drafts) == 1:
            synthesized = drafts[0]
        else:
            logger.info("Synthesizing %d drafts for '%s'...", len(drafts), name)
            drafts_text = "\n---\n".join(f"Draft {i+1}:\n{d}" for i, d in enumerate(drafts))
            synthesis_prompt = _SYNTHESIS_TEMPLATE.format(n=len(drafts), drafts=drafts_text)
            synthesized = agent.teacher_generate(synthesis_prompt, temperature=0.3, max_tokens=4096)
            if not synthesized:
                synthesized = drafts[0]

        if len(synthesized) > _CONDENSE_THRESHOLD:
            logger.info("Persona is %d chars — condensing to ~%d...", len(synthesized), _CONDENSE_TARGET)
            condensed = self._condense_persona(synthesized, agent)
            if condensed:
                synthesized = condensed

        out_dir = self.persona_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthesized.md"
        out_path.write_text(synthesized, encoding="utf-8")
        logger.info("Persona '%s' saved to %s (%d chars)", name, out_path, len(synthesized))
        return out_path

    def _condense_persona(self, text, agent):
        """Condense an oversized persona to fit within model context limits.

        Preserves all structural sections (0-11, behavioral rules, dialogue
        examples, language patterns) while reducing verbose prose.
        """
        prompt = (
            f"Condense this persona to under {_CONDENSE_TARGET} characters. "
            "Keep ALL sections (0-11), behavioral rules with → arrows, "
            "dialogue examples with > quotes, and language pattern lists intact. "
            "Preserve the exact markdown structure so it can be parsed. "
            "Remove verbose prose from sections 1-10 — keep only 2-3 sentences "
            "per section. Keep behavioral rules, dialogue examples, and "
            "language patterns at full count.\n\n"
            f"{text}"
        )
        result = agent.teacher_generate(prompt, temperature=0.3, max_tokens=4096)
        if result and len(result) < len(text):
            return result
        return None

"""PersonaManager — discover, load, generate, and switch personas."""

import logging
from pathlib import Path
from .persona import Persona

logger = logging.getLogger(__name__)

_GENERATION_TEMPLATE = """Create a detailed persona for "{name}".

{description}

Follow this exact structure:

### 0. Core Essence
- **Identity in 25 words**: ...
- **Top 3 defining traits**: ...
- **Primary communication style**: ...
- **Essential behavioral markers**: ...
- **Must-have linguistic patterns**: ...

### 1. Biographical Foundation and Personality

### 2. Voice / Communication Analysis

### 3. Signature Language Patterns

### 4. Narrative / Communication Structure

### 5. Subject Matter Expertise

### 6. Philosophical Framework

### 7. Emotional Range and Expression

### 8. Distinctive Patterns and Quirks

### 9. Evolution Over Time

### 10. Practical Application Guidelines

### 11. Platform Adaptation Bank

#### Behavioral Rules (If-Then)
Create at least 15 behavioral rules using the format:
1. If [condition] → Then [action]
2. If [condition] → Then [action]
...

#### Opening Phrases (at least 5):
1. ...
2. ...

#### Closing Phrases (at least 5):
1. ...
2. ...

#### Signature expressions (at least 15):
- ...
- ...
"""

_SYNTHESIS_TEMPLATE = """Synthesize these {n} persona drafts into one unified persona.
Combine the best elements from each. Follow the structure exactly.

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
            draft = agent.teacher_generate(prompt, temperature=temps[i])
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
            synthesized = agent.teacher_generate(synthesis_prompt, temperature=0.3)
            if not synthesized:
                synthesized = drafts[0]

        out_dir = self.persona_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthesized.md"
        out_path.write_text(synthesized, encoding="utf-8")
        logger.info("Persona '%s' saved to %s", name, out_path)
        return out_path

"""MCP server for querying Adam's knowledge and submitting learning experiences."""

import logging
import math
from mcp.server import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("Project Adam",
    instructions="Query Adam's knowledge and submit experiences through his learning architecture.")

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from . import get_cached_agent
        _agent = get_cached_agent()
    return _agent


# ── Knowledge Query Tools (read-only) ──────────────────────────────


@mcp.tool(description="Search semantic schemas, world model, and procedural skills for knowledge about a topic.")
def adam_query_knowledge(topic: str) -> dict:
    """Search all memory systems for knowledge about a topic.

    Args:
        topic: The concept or entity to search for.

    Returns:
        Dict with schemas, world model beliefs, skills, and counts.
    """
    agent = _get_agent()
    tl = topic.lower()

    schemas = []
    for sid, s in agent.semantic_memory.schemas.items():
        if tl in s.get("category", "").lower() or any(tl in f.lower() for f in s.get("facts", [])):
            schemas.append({
                "id": sid, "category": s["category"],
                "facts": s["facts"][-5:],
                "prediction_error": round(s.get("prediction_error", 1.0), 3),
                "observed_count": s.get("observed_count", 0),
                "slots": dict(s.get("slots", {})),
            })

    entities = {}
    for entity, attrs in agent.world_model.entities.items():
        if tl in entity:
            entities[entity] = {
                a: {"mean": round(m, 3), "uncertainty": round(math.sqrt(v), 3), "observations": c}
                for a, (m, v, c) in attrs.items()
            }

    skills = []
    for sid, skill in agent.procedural_memory.skills.items():
        if tl in " ".join(skill.keywords).lower():
            skills.append({
                "id": sid,
                "action": skill.action[:200],
                "q_value": round(skill.q_value, 3),
                "success_rate": round(skill.success_rate, 3),
                "usage_count": skill.usage_count,
            })

    return {
        "topic": topic,
        "schemas": schemas, "schema_count": len(schemas),
        "world_entities": entities, "entity_count": len(entities),
        "skills": skills, "skill_count": len(skills),
    }


@mcp.tool(description="Get Adam's Bayesian posterior beliefs about a specific entity.")
def adam_explain_entity(entity_name: str) -> dict:
    """Get Adam's Bayesian posterior beliefs about a specific entity.

    Args:
        entity_name: The entity to look up (case-insensitive).

    Returns:
        Dict with attribute-level means, uncertainties, observation counts.
    """
    agent = _get_agent()
    ent = agent.world_model.entities.get(entity_name.lower())
    if not ent:
        return {"entity": entity_name, "found": False}
    attributes = {
        a: {"mean": round(m, 3), "uncertainty": round(math.sqrt(v), 3), "observations": c}
        for a, (m, v, c) in ent.items()
    }
    return {
        "entity": entity_name, "found": True,
        "attributes": attributes,
        "total_observations": sum(c for _, _, c in ent.values()),
    }


@mcp.tool(description="Return statistics about Adam's memory systems and self-play state.")
def adam_get_status() -> dict:
    """Return statistics about Adam's memory systems and self-play state."""
    agent = _get_agent()
    ep_count = len(agent.episodic_memory.episodes)
    skills = agent.procedural_memory.skills
    status = {
        "memory": {
            "episodic_episodes": ep_count,
            "semantic_schemas": len(agent.semantic_memory.schemas),
            "world_entities": len(agent.world_model.entities),
            "procedural_skills": len(skills),
        },
        "learning": {
            "avg_skill_q": round(sum(s.q_value for s in skills.values()) / max(len(skills), 1), 3),
            "avg_skill_success": round(sum(s.success_rate for s in skills.values()) / max(len(skills), 1), 3),
        },
        "self_play": {},
    }
    if agent.self_play is not None:
        status["self_play"] = agent.self_play.get_stats()
    else:
        status["self_play"] = {"running": False}
    return status


# ── Teaching Tools (write via architecture paths) ───────────────────


@mcp.tool(description="Teach Adam by submitting a (query, response) pair. Stored in episodic memory, processed during consolidation.")
def adam_teach(query: str, response: str, reward: float = 0.85) -> dict:
    """Teach Adam by submitting a (query, response) learning pair.

    Architecture path: EpisodicMemory.add() → merge_episodes() → _lora_train_step().
    Same path as human chat. No bypass.

    Args:
        query: The question or context.
        response: The correct/expert response.
        reward: Quality signal (0.0-1.0, default 0.85).

    Returns:
        Dict confirming storage.
    """
    agent = _get_agent()
    r = max(0.0, min(1.0, reward))
    agent.episodic_memory.add(text=query, reward=r, action=response, context="mcp_teach")
    return {
        "status": "stored", "query_len": len(query), "response_len": len(response),
        "reward": r, "total_episodes": len(agent.episodic_memory.episodes),
    }


@mcp.tool(description="Submit an observation for Adam's Bayesian world model. Updates posterior belief via conjugate Gaussian.")
def adam_observe_entity(entity: str, attribute: str, value: float, confidence: float = 1.0) -> dict:
    """Submit an observation for Adam's Bayesian world model.

    Architecture path: WorldModel.observe() → conjugate Gaussian posterior update.
    Same path as observe_from_text(). No bypass.

    Args:
        entity: Entity name (e.g., "python", "Einstein").
        attribute: Attribute being observed (e.g., "difficulty", "intelligence").
        value: Observed value (numeric).
        confidence: Reliability (0.0-1.0, default 1.0).

    Returns:
        Dict with updated posterior mean, uncertainty, and count.
    """
    agent = _get_agent()
    agent.world_model.observe(entity, attribute, value, confidence=confidence)
    mean, var, count = agent.world_model.entities.get(entity.lower(), {}).get(attribute, (0, 1, 0))
    return {
        "status": "observed", "entity": entity.lower(), "attribute": attribute,
        "posterior_mean": round(mean, 3),
        "posterior_uncertainty": round(math.sqrt(var), 3),
        "observations": count,
    }


@mcp.tool(description="Submit a fact for semantic memory. Integrated via Piaget assimilation/accommodation.")
def adam_teach_fact(category: str, fact: str) -> dict:
    """Submit a fact for semantic memory.

    Architecture path: SemanticMemory.add() → assimilation/accommodation.
    If similarity >= 0.75 to existing schema → assimilate (update slots).
    If < 0.75 → accommodate (new schema with prediction_error=1.0).

    Args:
        category: Knowledge domain (e.g., "science", "user_preference").
        fact: The factual statement (e.g., "Python is dynamically typed").

    Returns:
        Dict with schema ID, prediction error, and observed count.
    """
    agent = _get_agent()
    sid = agent.semantic_memory.add(category, fact)
    schema = agent.semantic_memory.schemas.get(sid, {})
    return {
        "status": "stored", "schema_id": sid, "category": category,
        "prediction_error": round(schema.get("prediction_error", 1.0), 3),
        "observed_count": schema.get("observed_count", 0),
        "facts": schema.get("facts", [])[-3:],
    }


@mcp.tool(description="Submit a procedural skill example. Recorded with Q-learning and automatic chunking of repeated patterns.")
def adam_teach_skill(context: str, action: str, reward: float = 0.8) -> dict:
    """Submit a procedural skill example.

    Architecture path: ProceduralMemory.record() → keyword matching → Q-value update.
    Repeated patterns are automatically chunked into macro-actions.

    Args:
        context: The situation or trigger.
        action: The skill/response to learn.
        reward: How successful (0.0-1.0, default 0.8).

    Returns:
        Dict confirming storage with skill counts.
    """
    agent = _get_agent()
    agent.procedural_memory.record(context, action, reward)
    p_stats = agent.procedural_memory.stats()
    return {
        "status": "stored", "context_len": len(context), "action_len": len(action),
        "reward": reward,
        "total_skills": p_stats["num_skills"],
        "total_chunks": p_stats["num_chunks"],
    }


# ── Learning Trigger Tools ──────────────────────────────────────────


@mcp.tool(description="Run the full 6-step consolidation cycle on all accumulated episodes (human + MCP-taught).")
def adam_consolidate(rpe: float = 1.0) -> dict:
    """Run the full 6-step cognitive consolidation cycle.

    Steps:
    1. Replay: sample episodes from memory
    2. Prioritize: high-RPE events first
    3. Abstract: compress repeated patterns into schemata
    4. Prune: remove redundant/noisy memories
    5. Update world model: Bayesian update
    6. Update procedural policies: offline RL

    Same path as the metacog's REPLAY action.

    Args:
        rpe: Reward prediction error weight (default 1.0).

    Returns:
        Dict with pre/post consolidation stats.
    """
    agent = _get_agent()
    if not agent.episodic_memory.episodes:
        return {"status": "no_episodes", "episode_count": 0}
    before = len(agent.episodic_memory.episodes)
    agent.consolidator.merge_episodes(rpe=rpe)
    return {
        "status": "done",
        "episodes_before": before,
        "episodes_after": len(agent.episodic_memory.episodes),
        "skills": agent.procedural_memory.stats(),
        "schemas": len(agent.semantic_memory.schemas),
        "world_entities": len(agent.world_model.entities),
    }


@mcp.tool(description="Start, stop, restart, or check status of the autonomous self-play learning loop.")
def adam_self_play(action: str = "status") -> dict:
    """Control the autonomous self-play background thread.

    The thread generates (query, teacher_response) pairs into episodic memory
    during idle time. The metacog's REPLAY action consolidates them through
    the full 6-step cycle.

    Args:
        action: "start", "stop", "restart", or "status" (default).

    Returns:
        Dict with loop state and training stats.
    """
    agent = _get_agent()
    return agent.toggle_self_play(action)


# ── Persona Management Tools ─────────────────────────────────────────


@mcp.tool(description="List available personas that Adam can switch to.")
def adam_list_personas() -> list:
    """List all available personas in the persona directory."""
    agent = _get_agent()
    return agent.persona_manager.list_personas()


@mcp.tool(description="Get detailed information about a specific persona.")
def adam_get_persona(name: str) -> dict:
    """Get metadata and structure of a persona without switching to it.

    Args:
        name: The persona name to inspect.

    Returns:
        Dict with persona fields, rule count, phrase counts.
    """
    agent = _get_agent()
    return agent.persona_manager.get_persona_info(name)


@mcp.tool(description="Switch Adam to a different persona mid-session.")
def adam_switch_persona(name: str) -> dict:
    """Switch to a different persona. Takes effect on next chat.

    Args:
        name: The persona name to switch to.

    Returns:
        Dict with switch status and persona info.
    """
    agent = _get_agent()
    return agent.switch_persona(name)


@mcp.tool(description="Generate a new persona using the teacher API.")
def adam_generate_persona(name: str, description: str) -> dict:
    """Generate a new persona via the teacher API.

    Creates N variations with different temperatures, then synthesizes
    them into one final persona saved to personas/{name}/.

    Args:
        name: Persona name (used as the filename).
        description: Short description of the persona to generate.

    Returns:
        Dict with status and path to the generated persona.
    """
    agent = _get_agent()
    try:
        path = agent.persona_manager.generate_persona(name, description, agent)
        info = agent.persona_manager.get_persona_info(name)
        return {"status": "created", "path": str(path), "info": info}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_stdio():
    """Run MCP server over stdio for subprocess integration."""
    import asyncio
    asyncio.run(mcp.run_stdio_async())

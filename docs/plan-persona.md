# Persona Studio Integration Plan

## Architecture compliance rule

> Persona generation uses `teacher_generate()` for text generation only. The output is saved as a configuration file. It does NOT flow through episodic memory, consolidation, RPE, or LoRA. A persona shapes Adam's output style — it does not modify his learning engine, memory systems, or cognitive controller.

## Files changed

| File | Lines | Change |
|---|---|---|
| `src/project_adam/persona.py` | ~30 | Fix heading parsing (`###`), fix default path, add `to_dict()` |
| `src/project_adam/persona_manager.py` | ~150 | **New** — `PersonaManager` with list, load, generate |
| `src/project_adam/agent.py` | ~15 | Add `persona_manager`, `switch_persona()` |
| `src/project_adam/mcp_server.py` | ~40 | Add 4 persona MCP tools |
| `src/project_adam/api.py` | ~20 | Add 4 persona API endpoints |
| **Total** | **~255** | |

## Implementation order

1. `persona.py` — fix parsing + default path + `to_dict()`
2. `persona_manager.py` — `PersonaManager` class
3. `agent.py` — integrate `PersonaManager` + `switch_persona()`
4. `mcp_server.py` — 4 persona tools
5. `api.py` — 4 persona endpoints

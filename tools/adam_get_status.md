# adam_get_status

Return statistics about Adam's memory systems and self-play state.

## Input Schema

```json
{
  "type": "object",
  "properties": {}
}
```

## Returns

```json
{
  "memory": {
    "episodic_episodes": "int",
    "semantic_schemas": "int",
    "world_entities": "int",
    "procedural_skills": "int"
  },
  "learning": {
    "avg_skill_q": "float",
    "avg_skill_success": "float"
  },
  "self_play": {
    "running": "bool",
    "total_queries": "int",
    "total_trained": "int"
  }
}
```

## Architecture Path

Read-only. Aggregates stats from all memory systems.

## Example

```python
status = await call_tool("adam_get_status", {})
print(f"Episodes: {status['memory']['episodic_episodes']}")
print(f"Self-play running: {status['self_play']['running']}")
```

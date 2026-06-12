# adam_consolidate

Run the full 6-step cognitive consolidation cycle on all accumulated episodes (human + MCP-taught).

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "rpe": {"type": "number", "description": "Reward prediction error weight (default 1.0)"}
  }
}
```

## Returns

```json
{
  "status": "string",
  "episodes_before": "int",
  "episodes_after": "int",
  "skills": {"num_skills": "int", "num_chunks": "int", "avg_q_value": "float", "avg_success_rate": "float"},
  "schemas": "int",
  "world_entities": "int"
}
```

## Architecture Path

`OfflineConsolidator.merge_episodes(rpe)` — identical to the metacog's REPLAY action at `agent.py:273`.

### The 6 steps:

1. **Replay**: sample episodes from memory
2. **Prioritize**: high-RPE events first
3. **Abstract**: compress repeated patterns into schemata (semantic memory)
4. **Prune**: remove redundant/noisy memories
5. **Update world model**: Bayesian update from high-reward episodes
6. **Update procedural policies**: offline RL from prioritized experiences

After this, `_lora_train_step()` picks up high-reward episodes from episodic memory and trains the LoRA adapter.

## Example

```python
result = await call_tool("adam_consolidate", {"rpe": 1.0})
print(f"Episodes: {result['episodes_before']} → {result['episodes_after']}")
print(f"Skills: {result['skills']}")
```

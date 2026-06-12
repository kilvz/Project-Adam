# adam_teach

Teach Adam by submitting a (query, response) learning pair.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "query": {"type": "string", "description": "The question or context"},
    "response": {"type": "string", "description": "The correct/expert response"},
    "reward": {"type": "number", "description": "Quality signal (0.0-1.0, default 0.85)"}
  },
  "required": ["query", "response"]
}
```

## Returns

```json
{
  "status": "string",
  "query_len": "int",
  "response_len": "int",
  "reward": "float",
  "total_episodes": "int"
}
```

## Architecture Path

`EpisodicMemory.add()` → `merge_episodes()` → `_lora_train_step()`.

Same path as human chat. Stored with `context="mcp_teach"` for traceability. The next consolidation cycle (triggered by `adam_consolidate` or metacog REPLAY) processes it with RPE prioritization.

## Example

```python
await call_tool("adam_teach", {
    "query": "What is reinforcement learning?",
    "response": "RL is a machine learning paradigm where an agent learns by interacting with an environment...",
    "reward": 0.9
})
```

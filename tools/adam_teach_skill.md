# adam_teach_skill

Submit a procedural skill example. Recorded with Q-learning and automatic chunking of repeated patterns.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "context": {"type": "string", "description": "The situation or trigger"},
    "action": {"type": "string", "description": "The skill/response to learn"},
    "reward": {"type": "number", "description": "How successful (0.0-1.0, default 0.8)"}
  },
  "required": ["context", "action"]
}
```

## Returns

```json
{
  "status": "string",
  "context_len": "int",
  "action_len": "int",
  "reward": "float",
  "total_skills": "int",
  "total_chunks": "int"
}
```

## Architecture Path

`ProceduralMemory.record()` → keyword matching → Q-value update.

Repeated patterns are automatically chunked into macro-actions via `_detect_and_chunk_sequences()`.

## Example

```python
await call_tool("adam_teach_skill", {
    "context": "user asks about machine learning",
    "action": "explain supervised vs unsupervised learning with examples",
    "reward": 0.85
})
```

# adam_teach_fact

Submit a fact for semantic memory. Integrated via Piaget assimilation/accommodation.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "category": {"type": "string", "description": "Knowledge domain (e.g. 'science', 'user_preference')"},
    "fact": {"type": "string", "description": "The factual statement (e.g. 'Python is dynamically typed')"}
  },
  "required": ["category", "fact"]
}
```

## Returns

```json
{
  "status": "string",
  "schema_id": "string",
  "category": "string",
  "prediction_error": "float",
  "observed_count": "int",
  "facts": ["string"]
}
```

## Architecture Path

`SemanticMemory.add()` → assimilation/accommodation.

1. Find best-matching existing schema via embedding similarity
2. If similarity ≥ 0.75 → assimilate (update slots, reduce prediction error)
3. If similarity < 0.75 → accommodate (create new schema with prediction_error=1.0)

This mirrors Piaget's assimilation/accommodation (Section 3b).

## Example

```python
await call_tool("adam_teach_fact", {
    "category": "programming_languages",
    "fact": "Python is dynamically typed and garbage-collected"
})
```

# adam_query_knowledge

Search all memory systems for knowledge about a topic.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "topic": {
      "type": "string",
      "description": "The concept or entity to search for"
    }
  },
  "required": ["topic"]
}
```

## Returns

```json
{
  "topic": "string",
  "schemas": [{"id": "string", "category": "string", "facts": ["string"], "prediction_error": "float", "observed_count": "int", "slots": {}}],
  "schema_count": "int",
  "world_entities": {"entity": {"attribute": {"mean": "float", "uncertainty": "float", "observations": "int"}}},
  "entity_count": "int",
  "skills": [{"id": "string", "action": "string", "q_value": "float", "success_rate": "float", "usage_count": "int"}],
  "skill_count": "int"
}
```

## Architecture Path

Read-only. Queries `SemanticMemory.schemas`, `WorldModel.entities`, `ProceduralMemory.skills`.

## Example

```python
result = await call_tool("adam_query_knowledge", {"topic": "python"})
print(f"Found {result['schema_count']} schemas, {result['entity_count']} entities, {result['skill_count']} skills")
```

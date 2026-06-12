# adam_explain_entity

Get Adam's Bayesian posterior beliefs about a specific entity.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "entity_name": {
      "type": "string",
      "description": "The entity to look up (case-insensitive)"
    }
  },
  "required": ["entity_name"]
}
```

## Returns

```json
{
  "entity": "string",
  "found": "bool",
  "attributes": {"attribute": {"mean": "float", "uncertainty": "float", "observations": "int"}},
  "total_observations": "int"
}
```

## Architecture Path

Read-only. Queries `WorldModel.entities` for the entity's posterior distribution per attribute.

## Example

```python
result = await call_tool("adam_explain_entity", {"entity_name": "einstein"})
if result["found"]:
    for attr, data in result["attributes"].items():
        print(f"{attr}: mean={data['mean']}, uncertainty={data['uncertainty']}")
```

# adam_observe_entity

Submit an observation for Adam's Bayesian world model.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "entity": {"type": "string", "description": "Entity name (e.g. 'python', 'Einstein')"},
    "attribute": {"type": "string", "description": "Attribute being observed (e.g. 'difficulty', 'intelligence')"},
    "value": {"type": "number", "description": "Observed value (numeric)"},
    "confidence": {"type": "number", "description": "Reliability (0.0-1.0, default 1.0)"}
  },
  "required": ["entity", "attribute", "value"]
}
```

## Returns

```json
{
  "status": "string",
  "entity": "string",
  "attribute": "string",
  "posterior_mean": "float",
  "posterior_uncertainty": "float",
  "observations": "int"
}
```

## Architecture Path

`WorldModel.observe()` → conjugate Gaussian posterior update.

Same path as `observe_from_text()`. Updates the Bayesian posterior: `P(attribute | observation) ∝ P(observation | attribute) · P(attribute)`.

## Example

```python
await call_tool("adam_observe_entity", {
    "entity": "python",
    "attribute": "difficulty",
    "value": 0.7,
    "confidence": 0.9
})
```

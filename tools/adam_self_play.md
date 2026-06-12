# adam_self_play

Start, stop, restart, or check status of the autonomous self-play learning loop.

## Input Schema

```json
{
  "type": "object",
  "properties": {
    "action": {
      "type": "string",
      "enum": ["start", "stop", "restart", "status"],
      "description": "Control action (default 'status')"
    }
  }
}
```

## Returns

```json
{
  "status": "string",
  "stats": {
    "total_queries": "int",
    "total_trained": "int",
    "current_strategy": "string|null",
    "running": "bool",
    "last_error": "string|null"
  }
}
```

## Architecture Path

`CognitiveAgent.toggle_self_play(action)` → `SelfPlayLearner.start()/stop()`.

The thread generates (query, teacher_response) pairs into episodic memory during idle time. The metacog's REPLAY action consolidates them through the full 6-step cycle. The thread is gated by the metacog — it only generates when `last_action` is `EXPLORE` or `ASK_FOR_HELP` (Section 5).

### Self-play strategies

| Strategy | Source | Filter |
|---|---|---|
| `schema` | Semantic memory | `prediction_error > 0.2` or `observed_count < 3` |
| `world_model` | World model | Highest `uncertainty()` entities |
| `procedural` | Procedural memory | Skills with `q_value < 0.3` or `success_rate < 0.5` |
| `creative` | Teacher API | Two-step: ask teacher for topic, then formulate query |

## Example

```python
# Start self-play
await call_tool("adam_self_play", {"action": "start"})
# Check status
status = await call_tool("adam_self_play", {"action": "status"})
print(f"Running: {status['status']}, Queries: {status['stats']['total_queries']}")
```

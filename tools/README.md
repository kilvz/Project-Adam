# Project Adam — MCP Tools

This directory documents the 9 MCP tools exposed by Adam's MCP server.

## How to connect

```bash
# Via stdio (external AI spawns as subprocess):
python3 -m project_adam --mcp
```

Using the MCP client SDK:

```python
from mcp import StdioClientParameters, stdio_client

async with stdio_client(StdioClientParameters(
    command="python3",
    args=["-m", "project_adam", "--mcp"],
)) as (read, write):
    # 9 tools available
    pass
```

## Tools

### Knowledge Query (read-only)

| Tool | Description | Memory systems queried |
|---|---|---|
| `adam_query_knowledge` | Search all memory for a topic | Semantic schemas, World model entities, Procedural skills |
| `adam_explain_entity` | Get Bayesian beliefs about an entity | World model posterior per attribute |
| `adam_get_status` | Stats across all memory systems | All (aggregate counts) |

### Teaching (write via architecture paths)

| Tool | Architecture path | What it does |
|---|---|---|
| `adam_teach` | `EpisodicMemory.add()` → consolidation | Submit (query, response) pair |
| `adam_observe_entity` | `WorldModel.observe()` → Bayesian update | Submit entity observation |
| `adam_teach_fact` | `SemanticMemory.add()` → assimilation/accommodation | Submit fact |
| `adam_teach_skill` | `ProceduralMemory.record()` → Q-learning | Submit skill example |

### Learning Triggers

| Tool | Architecture path | What it does |
|---|---|---|
| `adam_consolidate` | `OfflineConsolidator.merge_episodes()` — full 6-step cycle | Consolidate all episodes |
| `adam_self_play` | `SelfPlayLearner.start()/stop()` | Control autonomous loop |

## Architecture rule

> All training data flows through `EpisodicMemory` → `OfflineConsolidator.merge_episodes()` (6-step cycle with RPE prioritization) → `_lora_train_step()`. No supervised bypass, no weight injection.

## Data flow for a typical training session

```
1. adam_teach(query, response, reward=0.9)      # submit a learning pair
2. adam_observe_entity(entity="rl", attribute="difficulty", value=0.7)  # update world model
3. adam_teach_fact(category="AI", fact="RL uses rewards")  # update semantic memory
4. adam_consolidate(rpe=1.0)                       # run 6-step cycle → LoRA training
5. adam_query_knowledge(topic="reinforcement learning")  # inspect what Adam learned
```

See individual tool files for detailed schemas and examples.

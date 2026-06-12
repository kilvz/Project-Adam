# Titans: Learning to Memorize at Test Time

- **Authors**: Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)
- **arXiv**: [2501.00663](https://arxiv.org/abs/2501.00663)
- **Code**: [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch)
- **Published**: 31 Dec 2024

## Abstract

Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

## Key Concepts Used in Adam

| Concept | Our Implementation |
|---------|-------------------|
| Neural memory = MLP weights | `memory/diffmemory.py` — `_MemoryMLP` |
| Surprise-gated storage | `store()` with threshold gating |
| Momentum across tokens | `_momentum = β·momentum + (1-β)·surprise` |
| Weight decay forgetting | AdamW + re-encode prune |
| Surprise metric (gradient norm) | Planned for v3 (currently reconstruction error) |

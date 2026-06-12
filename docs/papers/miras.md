# MIRAS: It's All Connected

- **Authors**: Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (Google Research)
- **arXiv**: [2504.13173](https://arxiv.org/abs/2504.13173)
- **Published**: 17 Apr 2025

## Abstract

Designing efficient and effective architectural backbones has been in the core of research efforts to enhance the capability of foundation models. Inspired by the human cognitive phenomenon of attentional bias — the natural tendency to prioritize certain events or stimuli — we reconceptualize neural architectures, including Transformers, Titans, and modern linear recurrent neural networks as associative memory modules that learn a mapping of keys and values using an internal objective, referred to as attentional bias. Surprisingly, we observed that most existing sequence models leverage either (1) dot-product similarity, or (2) L2 regression objectives as their attentional bias. Going beyond these objectives, we present a set of alternative attentional bias configurations along with their effective approximations to stabilize their training procedure. We then reinterpret forgetting mechanisms in modern deep learning architectures as a form of retention regularization, providing a novel set of forget gates for sequence models. Building upon these insights, we present Miras, a general framework to design deep learning architectures based on four choices of: (i) associative memory architecture, (ii) attentional bias objective, (iii) retention gate, and (iv) memory learning algorithm. We present three novel sequence models — Moneta, Yaad, and Memora — that go beyond the power of existing linear RNNs while maintaining a fast parallelizable training process.

## The Four Design Choices

| Choice | Description | Our Implementation |
|--------|-------------|-------------------|
| **Memory architecture** | What stores the information (vector, matrix, MLP) | `_MemoryMLP` — MLP-based |
| **Attentional bias** | Internal objective for learning (MSE, Huber, L1, dot-product) | MSE only (Huber planned for v3) |
| **Retention gate** | Forgetting mechanism (weight decay, learned gates) | AdamW weight_decay (adaptive head planned for v3) |
| **Memory algorithm** | Optimization algorithm (SGD, Adam, online) | AdamW |

## MIRAS Variants

| Variant | Key Idea | Status |
|---------|----------|--------|
| **YAAD** | Huber loss for outlier-robust memory | Planned for v3 |
| **MONETA** | Generalized norms for attentional bias | Not planned |
| **MEMORA** | Probability-constrained memory updates | Not planned |

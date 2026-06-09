import torch
from titans_pytorch import MemoryAsContextTransformer
import os

CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_checkpoint(path):
    data = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = data.get("config", {})
    model = MemoryAsContextTransformer(
        num_tokens=cfg.get("vocab_size", 256),
        dim=cfg.get("dim", 384),
        depth=cfg.get("depth", 6),
        heads=cfg.get("heads", 6),
        dim_head=cfg.get("dim_head", 64),
        segment_len=cfg.get("segment_len", 128),
        num_persist_mem_tokens=cfg.get("num_persist_mem", 8),
        num_longterm_mem_tokens=cfg.get("num_longterm_mem", 16),
        ff_mult=cfg.get("ff_mult", 4),
    ).to(DEVICE)
    model.load_state_dict(data.get("model_state_dict", data))
    model.eval()
    return model, data.get("step", 0), data.get("loss", 0)

def generate(model, prompt_text, max_new_tokens=100, temperature=0.8, top_k=40):
    prompt_bytes = list(prompt_text.encode("utf-8"))
    input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > 512:
                input_ids = input_ids[:, -512:]
            logits = model(input_ids)[:, -1, :] / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, top_k)
                logits[logits < vals[:, -1:]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    return bytes(input_ids[0].tolist()).decode("utf-8", errors="replace")

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
    if not ckpts:
        print("No checkpoints found. Run train_titans.py first.")
        exit()

    path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
    model, step, loss = load_checkpoint(path)
    print(f"Loaded {ckpts[-1]} | step {step} | loss {loss:.4f} | {sum(p.numel() for p in model.parameters()):,} params\n")

    while True:
        prompt = input("> ")
        if prompt in ("exit", "quit"):
            break
        output = generate(model, prompt, max_new_tokens=150, temperature=0.8)
        print(output)
        print()

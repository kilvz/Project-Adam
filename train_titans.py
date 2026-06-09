import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from titans_pytorch import MemoryAsContextTransformer
from datasets import load_dataset
import os, math, time

# === CONFIG — tweak these ===
VOCAB_SIZE = 256
DIM = 256
DEPTH = 4
HEADS = 4
DIM_HEAD = 64
SEGMENT_LEN = 64
NUM_PERSIST_MEM = 4
NUM_LONGTERM_MEM = 8
FF_MULT = 4
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 3e-4
TOTAL_STEPS = 50000
LOG_EVERY = 50
SAVE_EVERY = 2000
OUTPUT_DIR = "checkpoints"
DATA_SAMPLE = 50000
MAX_SEQ_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class TextDataset(Dataset):
    def __init__(self, texts, seq_len):
        self.seq_len = seq_len
        self.samples = []
        for text in texts:
            tokens = list(text.encode("utf-8"))[:seq_len * 10]
            for i in range(0, len(tokens) - seq_len, seq_len // 2):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

print(f"Loading FineWeb sample-10BT ({DATA_SAMPLE} docs)...")
ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
texts = []
for i, example in enumerate(ds):
    if i >= DATA_SAMPLE:
        break
    texts.append(example["text"])
    if (i + 1) % 10000 == 0:
        print(f"  Loaded {i+1} examples")

train_ds = TextDataset(texts, MAX_SEQ_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print(f"\nSamples: {len(train_ds)} | Docs: {len(texts)}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM} | Seq len: {MAX_SEQ_LEN}")

print("Building Titans MAC model...")
model = MemoryAsContextTransformer(
    num_tokens=VOCAB_SIZE,
    dim=DIM, depth=DEPTH, heads=HEADS, dim_head=DIM_HEAD,
    segment_len=SEGMENT_LEN,
    num_persist_mem_tokens=NUM_PERSIST_MEM,
    num_longterm_mem_tokens=NUM_LONGTERM_MEM,
    ff_mult=FF_MULT,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Device: {device}")

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
step = 0
accum_loss = 0.0
start_time = time.time()
data_iter = iter(train_loader)

model.train()
while step < TOTAL_STEPS:
    for _ in range(GRAD_ACCUM):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        (loss / GRAD_ACCUM).backward()
        accum_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim.step()
    optim.zero_grad()
    step += 1

    if step % LOG_EVERY == 0:
        elapsed = time.time() - start_time
        avg_loss = accum_loss / LOG_EVERY
        print(f"step {step:>6d}/{TOTAL_STEPS} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):.2f} | "
              f"tok/s {step * BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN / elapsed:.0f} | {elapsed:.0f}s")
        accum_loss = 0.0

    if step % SAVE_EVERY == 0:
        path = os.path.join(OUTPUT_DIR, f"titans_mac_step_{step}.pt")
        torch.save({
            "step": step, "loss": avg_loss,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "config": {"vocab_size": VOCAB_SIZE, "dim": DIM, "depth": DEPTH,
                       "heads": HEADS, "dim_head": DIM_HEAD, "segment_len": SEGMENT_LEN,
                       "num_persist_mem": NUM_PERSIST_MEM,
                       "num_longterm_mem": NUM_LONGTERM_MEM, "ff_mult": FF_MULT},
        }, path)
        print(f"  Saved: {path}")

print(f"\nDone. {time.time() - start_time:.0f}s")
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "titans_mac_final.pt"))

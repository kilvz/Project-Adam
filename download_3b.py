#!/usr/bin/env python3
"""Download Qwen2.5-3B-Instruct to the cache dir transformers expects."""
from huggingface_hub import snapshot_download, hf_hub_download
import os

model_id = "Qwen/Qwen2.5-3B-Instruct"
# transformers uses ~/.cache/huggingface/hub/ by default
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

print(f"Downloading {model_id} to {cache_dir} ...")
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    resume_download=True,
    ignore_patterns=["*.safetensors"],  # skip weights first
)
# now download the actual weight shards
shards = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
]
for shard in shards:
    print(f"Downloading {shard}...")
    hf_hub_download(
        repo_id=model_id, filename=shard, cache_dir=cache_dir,
        resume_download=True,
    )
print("Done!")

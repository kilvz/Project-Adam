import os
import logging
import torch
from pathlib import Path
# ── defaults ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"
MODEL_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL = MODEL_3B
MODEL_CHAIN = [MODEL_3B, MODEL_1_5B, MODEL_0_5B]

_4BIT_CONFIG = None

def _get_4bit_config():
    global _4BIT_CONFIG
    if _4BIT_CONFIG is None:
        from transformers import BitsAndBytesConfig
        _4BIT_CONFIG = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    return _4BIT_CONFIG

def get_4bit_config():
    return _get_4bit_config()

PERSONA_PATH = Path("persona-studio/personas/adam.md")

GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
}

BACKEND_CONFIG = {
    "mode": "local",
    "api": {
        "endpoint": "https://<remoteapibackend>/v1/chat/completions",
        "key": "",
        "model": "remote-model",
        "timeout": 15,
    },
}

SELF_PLAY_CONFIG = {
    "enabled": False,
    "interval_seconds": 120,
    "batch_size": 8,
    "strategies": ["schema", "world_model", "procedural", "creative"],
    "max_recent_queries": 200,
    "reward": 0.85,
    "checkpoint_interval": 50,
}

# ── Hardware detection ────────────────────────────────────────────────
HARDWARE_TIER = "unknown"  # "low", "mid", "high"
GPU_VRAM_GB = 0
GPU_COMPUTE_CAP = (0, 0)

def _detect_hardware():
    global HARDWARE_TIER, GPU_VRAM_GB, GPU_COMPUTE_CAP
    if not torch.cuda.is_available():
        HARDWARE_TIER = "low"
        GPU_VRAM_GB = 0
        GPU_COMPUTE_CAP = (0, 0)
        return

    try:
        GPU_VRAM_GB = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        GPU_COMPUTE_CAP = (
            torch.cuda.get_device_capability(0)[0],
            torch.cuda.get_device_capability(0)[1],
        )
        cc_major, cc_minor = GPU_COMPUTE_CAP

        if GPU_VRAM_GB >= 24 and cc_major >= 8:
            HARDWARE_TIER = "high"
        elif GPU_VRAM_GB >= 8 and cc_major >= 7:
            HARDWARE_TIER = "mid"
        else:
            HARDWARE_TIER = "low"
    except Exception:
        HARDWARE_TIER = "low"

_detect_hardware()

_memory_dir_override = None
_CONFIG_LOADED = False


def _cast_dtype(val):
    """Recursively cast float16/32/64/bfloat16 strings to torch dtypes."""
    if isinstance(val, dict):
        return {k: _cast_dtype(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_cast_dtype(v) for v in val]
    if isinstance(val, str) and val.startswith("torch."):
        return getattr(torch, val.removeprefix("torch."), val)
    return val


def load_config(path=None):
    global DEVICE, BASE_MODEL, MODEL_CHAIN, PERSONA_PATH, _4BIT_CONFIG
    global GENERATION_CONFIG, _CONFIG_LOADED, SELF_PLAY_CONFIG

    if _CONFIG_LOADED:
        return

    path = Path(path) if path else Path("config.yaml")
    if not path.exists():
        _CONFIG_LOADED = True
        return

    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = _cast_dtype(yaml.safe_load(f) or {})

    if "device" in cfg:
        DEVICE = cfg["device"]
    if "base_model" in cfg:
        BASE_MODEL = cfg["base_model"]
    if "model_chain" in cfg:
        MODEL_CHAIN = cfg["model_chain"]
    if "persona_path" in cfg:
        PERSONA_PATH = Path(cfg["persona_path"])
    if "memory_dir" in cfg:
        set_memory_dir(cfg["memory_dir"])

    q = cfg.get("quantization", {})
    if q:
        try:
            from transformers import BitsAndBytesConfig
            _4BIT_CONFIG = BitsAndBytesConfig(
                load_in_4bit=q.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=q.get("bnb_4bit_compute_dtype", torch.float16),
                bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", False),
            )
        except Exception:
            _4BIT_CONFIG = None

    gen = cfg.get("generation", {})
    if gen:
        GENERATION_CONFIG.update(gen)

    b = cfg.get("backend", {})
    if b:
        mode = b.get("mode", "auto")
        api = b.get("api", {})
        if api:
            key_raw = api.get("key", "")
            if key_raw.startswith("${") and key_raw.endswith("}"):
                env_var = key_raw[2:-1]
                key_raw = os.environ.get(env_var, "")
            BACKEND_CONFIG["api"].update({
                "endpoint": api.get("endpoint", BACKEND_CONFIG["api"]["endpoint"]),
                "key": key_raw or BACKEND_CONFIG["api"]["key"],
                "model": api.get("model", BACKEND_CONFIG["api"]["model"]),
                "timeout": api.get("timeout", BACKEND_CONFIG["api"]["timeout"]),
            })

        if mode == "auto":
            mode = "api" if HARDWARE_TIER == "low" else "local"
        elif mode == "api" and not BACKEND_CONFIG["api"]["key"] and HARDWARE_TIER == "low":
            mode = "local" if HARDWARE_TIER == "low" else "api"

        BACKEND_CONFIG["mode"] = mode

    sp = cfg.get("self_play", {})
    if isinstance(sp, dict) and sp:
        SELF_PLAY_CONFIG.update({
            "enabled": sp.get("enabled", SELF_PLAY_CONFIG["enabled"]),
            "interval_seconds": sp.get("interval_seconds", SELF_PLAY_CONFIG["interval_seconds"]),
            "batch_size": sp.get("batch_size", SELF_PLAY_CONFIG["batch_size"]),
            "strategies": sp.get("strategies", SELF_PLAY_CONFIG["strategies"]),
            "max_recent_queries": sp.get("max_recent_queries", SELF_PLAY_CONFIG["max_recent_queries"]),
            "reward": sp.get("reward", SELF_PLAY_CONFIG["reward"]),
            "checkpoint_interval": sp.get("checkpoint_interval", SELF_PLAY_CONFIG["checkpoint_interval"]),
        })

    _CONFIG_LOADED = True


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_memory_dir():
    if _memory_dir_override is not None:
        return _memory_dir_override
    p = Path("agent_memory")
    p.mkdir(exist_ok=True)
    return p


def set_memory_dir(path):
    global _memory_dir_override
    _memory_dir_override = Path(path)
    _memory_dir_override.mkdir(exist_ok=True)

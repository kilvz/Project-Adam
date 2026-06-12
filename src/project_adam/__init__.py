from .config import (DEVICE, MODEL_3B, MODEL_1_5B, MODEL_0_5B, BASE_MODEL,
                     MODEL_CHAIN, PERSONA_PATH, GENERATION_CONFIG,
                     BACKEND_CONFIG, HARDWARE_TIER, GPU_VRAM_GB, GPU_COMPUTE_CAP,
                     get_memory_dir, set_memory_dir,
                     load_config, setup_logging, get_4bit_config,
                     SELF_PLAY_CONFIG, DIFFMEMORY_CONFIG)
load_config()
from .persona import Persona
from .profiles import UserProfileManager
from .encoder import SensoryEncoder, VisionEncoder, AudioEncoder
from .memory.working import WorkingMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .memory.procedural import ProceduralMemory
from .memory.spatial import SpatialMemory
from .memory.store import SQLiteStore
from .language import LanguageInterface
from .sfl import SFLModule
from .metacog import MetacognitiveController
from .search import WebSearch
from .consolidator import OfflineConsolidator
from .selector import ActionSelector
from .rl_core import TDCore, ValueNetwork
from .world_model import WorldModel
from .agent import CognitiveAgent

_AGENT_CACHE = None

def get_cached_agent():
    global _AGENT_CACHE
    if _AGENT_CACHE is None:
        from .agent import CognitiveAgent
        _AGENT_CACHE = CognitiveAgent()
    return _AGENT_CACHE


__all__ = [
    "DEVICE", "MODEL_3B", "MODEL_1_5B", "MODEL_0_5B", "BASE_MODEL",
    "MODEL_CHAIN", "PERSONA_PATH", "GENERATION_CONFIG", "get_4bit_config",
    "BACKEND_CONFIG", "HARDWARE_TIER", "GPU_VRAM_GB", "GPU_COMPUTE_CAP",
    "get_memory_dir", "set_memory_dir",
    "load_config", "setup_logging",
    "get_cached_agent", "SELF_PLAY_CONFIG", "DIFFMEMORY_CONFIG", "get_generation_config", "build_gen_kwargs",
    "Persona", "UserProfileManager",
    "SensoryEncoder", "VisionEncoder", "AudioEncoder",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory",
    "ProceduralMemory", "SpatialMemory", "SQLiteStore",
    "LanguageInterface",
    "SFLModule", "MetacognitiveController", "WebSearch",
    "OfflineConsolidator", "ActionSelector", "TDCore", "ValueNetwork",
    "WorldModel", "CognitiveAgent",
]

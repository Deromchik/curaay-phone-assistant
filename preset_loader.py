import json
from pathlib import Path
from typing import Any

PRESETS_DIR = Path(__file__).resolve().parent / "presets"

PRESET_KEYS = [
    "prompt_type",
    "language",
    "database_occurances",
    "knowledge_base",
    "examples",
    "special_context_chunks",
    "file_chunks",
    "learning_video_answer_text",
    "expert_id",
    "answer_model",
    "is_brief_mode",
    "is_expert_specific",
    "image_data",
    "screenshot_mode",
    "is_voice_mode",
]


def load_preset(prompt_type: str) -> dict[str, Any]:
    """Load default input config for default or browser prompt type."""
    path = PRESETS_DIR / f"{prompt_type}_preset.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: data.get(key) for key in PRESET_KEYS if key in data}

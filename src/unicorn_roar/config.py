"""Central configuration & compute node routing for 獨角獸那麼叫 pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
ASSETS_DIR = REPO_ROOT / "assets"
RUNS_DIR = REPO_ROOT / "runs"
REFERENCE_DIR = REPO_ROOT / "reference_channel"


def load_channel_profile() -> dict[str, Any]:
    profile_path = CONFIG_DIR / "channel_profile.yaml"
    return yaml.safe_load(profile_path.read_text(encoding="utf-8"))


def get_runtime_endpoint_path() -> Path:
    return CONFIG_DIR / "runtime_endpoints.json"


def get_voice_pack_dir() -> Path:
    return ASSETS_DIR / "voice_clone" / "reference_pack"

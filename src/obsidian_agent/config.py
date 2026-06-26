from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator


class ModelConfig(BaseModel):
    model_name: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: SecretStr | None = None
    temperature: float = 0

    @field_validator("api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value):
        if value is None:
            return None
        if isinstance(value, SecretStr):
            return value
        value = value.strip()
        return value or None


class SearchConfig(BaseModel):
    max_results: int = Field(default=5, ge=1)
    max_search_iterations: int = Field(default=2, ge=1)


class AppConfig(BaseModel):
    vault_path: Path
    model: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    @field_validator("vault_path")
    @classmethod
    def expand_vault_path(cls, value: Path) -> Path:
        return value.expanduser()


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Copy config.example.yaml to config.yaml first."
        )

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)

"""Load settings from cursor_mysql_sync/.env or repo-root .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

BUILD_ROOT = Path(__file__).resolve().parent
REPO_ROOT = BUILD_ROOT.parent
SYNC_ENV = REPO_ROOT / "cursor_mysql_sync" / ".env"
ROOT_ENV = REPO_ROOT / ".env"

# Prefer repo-root .env; legacy paths only fill missing keys
if ROOT_ENV.is_file():
    load_dotenv(ROOT_ENV)
if SYNC_ENV.is_file():
    load_dotenv(SYNC_ENV, override=False)
load_dotenv(BUILD_ROOT / ".env", override=False)

PLACEHOLDERS = frozenset({"", "change-me", "your-openai-api-key", "your-pinecone-api-key"})

FRAMEWORK_NAMESPACE = "framework-v1"
OUTPUT_DIR = BUILD_ROOT / "output"


def _optional(name: str, default: str) -> str:
    return os.getenv(name) or default


def _is_placeholder(value: str | None) -> bool:
    return not value or value.strip().lower() in PLACEHOLDERS


@dataclass(frozen=True)
class MySQLSettings:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: Literal["openai", "azure", "local"]
    model: str
    dimensions: int
    openai_api_key: str | None
    azure_api_key: str | None
    azure_endpoint: str | None
    azure_api_version: str | None
    azure_deployment: str | None


@dataclass(frozen=True)
class PineconeSettings:
    api_key: str
    index_name: str
    cloud: str
    region: str


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    temperature: float
    json_mode: bool
    timeout_sec: int


@dataclass(frozen=True)
class VisionLLMSettings:
    """HF InferenceClient (Novita) for multimodal profile + evidence vision."""

    provider: str
    model: str
    api_key: str
    timeout_sec: int


@dataclass(frozen=True)
class Settings:
    mysql: MySQLSettings
    embedding: EmbeddingSettings
    pinecone: PineconeSettings
    llm: LLMSettings
    vision_llm: VisionLLMSettings
    match_threshold: float


def load_mysql_settings() -> MySQLSettings:
    password = os.getenv("MYSQL_PASSWORD") or os.getenv("MYSQL_PASS")
    if _is_placeholder(password):
        raise RuntimeError("Set MYSQL_PASSWORD in cursor_mysql_sync/.env")
    user = os.getenv("MYSQL_USER")
    if _is_placeholder(user):
        raise RuntimeError("Set MYSQL_USER in cursor_mysql_sync/.env")
    return MySQLSettings(
        host=_optional("MYSQL_HOST", "localhost"),
        port=int(_optional("MYSQL_PORT", "3306")),
        user=user or "",
        password=password or "",
        database=_optional("MYSQL_DATABASE", _optional("MYSQL_DB", "aiev")),
    )


def load_settings(*, require_pinecone: bool = False) -> Settings:
    provider_raw = _optional("EMBEDDING_PROVIDER", "local").lower()
    if provider_raw not in {"openai", "azure", "local"}:
        raise RuntimeError(f"Invalid EMBEDDING_PROVIDER: {provider_raw}")

    if provider_raw == "local":
        default_model = "BAAI/bge-small-en-v1.5"
        default_dims = "384"
        default_threshold = "0.68"
    else:
        default_model = "text-embedding-3-large"
        default_dims = "3072"
        default_threshold = "0.72"

    embedding = EmbeddingSettings(
        provider=provider_raw,  # type: ignore[arg-type]
        model=_optional("EMBEDDING_MODEL", default_model),
        dimensions=int(_optional("EMBEDDING_DIMENSIONS", default_dims)),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    )

    pinecone_key = os.getenv("PINECONE_API_KEY") or ""
    if require_pinecone and _is_placeholder(pinecone_key):
        raise RuntimeError("Set PINECONE_API_KEY in .env for Pinecone operations.")

    pinecone = PineconeSettings(
        api_key=pinecone_key,
        index_name=_optional("PINECONE_INDEX_NAME", "aiev-competencies-bge"),
        cloud=_optional("PINECONE_CLOUD", "aws"),
        region=_optional("PINECONE_REGION", "us-east-1"),
    )

    llm_provider = _optional("PROFILE_LLM_PROVIDER", "multimodal").lower()
    llm_base_url = (
        os.getenv("PROFILE_LLM_BASE_URL")
        or os.getenv("LM_STUDIO_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    llm_api_key = (
        os.getenv("PROFILE_LLM_API_KEY")
        or os.getenv("LM_STUDIO_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    json_mode_raw = _optional("PROFILE_LLM_JSON_MODE", "false").lower()
    json_mode = json_mode_raw in {"1", "true", "yes", "on"}

    llm = LLMSettings(
        provider=llm_provider,
        model=_optional("PROFILE_LLM_MODEL", ""),
        api_key=None if _is_placeholder(llm_api_key) else llm_api_key,
        base_url=llm_base_url.rstrip("/") if llm_base_url else None,
        temperature=float(_optional("PROFILE_LLM_TEMPERATURE", "0.2")),
        json_mode=json_mode,
        timeout_sec=int(_optional("PROFILE_LLM_TIMEOUT_SEC", "300")),
    )

    threshold = float(_optional("TELEMETRY_MATCH_THRESHOLD", _optional("MATCH_THRESHOLD", default_threshold)))

    vision_token = os.getenv("NOVITA_API_KEY") or os.getenv("HF_TOKEN") or ""
    vision_llm = VisionLLMSettings(
        provider=_optional("EVIDENCE_LLM_PROVIDER", "novita"),
        model=_optional(
            "PROFILE_LLM_MODEL",
            _optional("EVIDENCE_LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
        ),
        api_key="" if _is_placeholder(vision_token) else vision_token,
        timeout_sec=int(
            _optional(
                "PROFILE_LLM_TIMEOUT_SEC",
                _optional("EVIDENCE_LLM_TIMEOUT_SEC", "600"),
            )
        ),
    )

    return Settings(
        mysql=load_mysql_settings(),
        embedding=embedding,
        pinecone=pinecone,
        llm=llm,
        vision_llm=vision_llm,
        match_threshold=threshold,
    )


def validate_llm_settings(settings: Settings) -> None:
    """Raise if PROFILE_LLM_PROVIDER requires missing env vars."""
    llm = settings.llm
    if llm.provider in {"template", "none", "off"}:
        return
    if llm.provider not in {
        "multimodal",
        "vision",
        "openai",
        "lmstudio",
        "openai_compatible",
    }:
        raise RuntimeError(
            f"Unknown PROFILE_LLM_PROVIDER: {llm.provider!r}. "
            "Use template, multimodal, lmstudio, openai, or openai_compatible."
        )
    if not settings.vision_llm.model:
        raise RuntimeError(
            "Set PROFILE_LLM_MODEL or EVIDENCE_LLM_MODEL for multimodal profiles."
        )
    if not settings.vision_llm.api_key:
        raise RuntimeError(
            "Set HF_TOKEN or NOVITA_API_KEY in .env for multimodal profile (Qwen-VL)."
        )

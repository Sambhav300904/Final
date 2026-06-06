"""Shared embedding client (local BGE, OpenAI, or Azure)."""

from __future__ import annotations

from typing import List

from config import EmbeddingSettings


class EmbeddingClient:
    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._local_model = None
        self._client = None
        self._model_or_deployment = settings.model

        if settings.provider == "openai":
            from openai import OpenAI

            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY required for EMBEDDING_PROVIDER=openai")
            self._client = OpenAI(api_key=settings.openai_api_key)
        elif settings.provider == "azure":
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=settings.azure_api_key,
                azure_endpoint=settings.azure_endpoint or "",
                api_version=settings.azure_api_version or "",
            )
            self._model_or_deployment = settings.azure_deployment or settings.model

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._settings.provider == "local":
            if self._local_model is None:
                from sentence_transformers import SentenceTransformer

                self._local_model = SentenceTransformer(self._settings.model)
            vectors = self._local_model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return [v.tolist() for v in vectors]
        if self._client is None:
            raise RuntimeError("API embedding client not configured")
        response = self._client.embeddings.create(
            model=self._model_or_deployment, input=texts
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

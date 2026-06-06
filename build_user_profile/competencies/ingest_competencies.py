"""
One-time (or refresh): embed 43 framework items and upsert to Pinecone.

Usage (from build_user_profile folder):
    python competencies/ingest_competencies.py --dry-run
    python competencies/ingest_competencies.py
    python competencies/ingest_competencies.py --recreate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_BUILD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD))

from competencies_data import COMPETENCIES, COMPETENCY_ITEMS  # noqa: E402
from config import FRAMEWORK_NAMESPACE, load_settings  # noqa: E402
from embedding_client import EmbeddingClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_competencies")

BATCH_SIZE = 32


def _embedding_text(competency_name: str, text: str) -> str:
    return f"Competency: {competency_name}\nItem: {text}"


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest 43 competency vectors to Pinecone")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--recreate", action="store_true", help="Delete namespace first")
    args = parser.parse_args()

    settings = load_settings(require_pinecone=True)
    items = list(COMPETENCY_ITEMS)
    assert len(items) == 43, f"Expected 43 items, got {len(items)}"

    for cid, name in COMPETENCIES.items():
        n = sum(1 for it in items if it.competency_id == cid)
        log.info("Competency %s (%s): %s items", cid, name, n)

    embedder = EmbeddingClient(settings.embedding)
    vectors: list[dict] = []
    for batch in _chunked(items, BATCH_SIZE):
        texts = [_embedding_text(it.competency_name, it.text) for it in batch]
        embeddings = embedder.embed_batch(texts)
        for item, vector in zip(batch, embeddings):
            if len(vector) != settings.embedding.dimensions:
                raise RuntimeError(
                    f"Dim mismatch item {item.item_id}: {len(vector)} vs "
                    f"{settings.embedding.dimensions}"
                )
            vectors.append(
                {
                    "id": f"item-{item.item_id}",
                    "values": vector,
                    "metadata": {
                        "item_id": item.item_id,
                        "competency_id": item.competency_id,
                        "competency_name": item.competency_name,
                        "text": item.text,
                        "framework_version": "v1",
                    },
                }
            )
    log.info("Built %s vectors", len(vectors))

    if args.dry_run:
        for v in vectors[:3]:
            log.info("Sample id=%s item_id=%s", v["id"], v["metadata"]["item_id"])
        return 0

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone.api_key)
    name = settings.pinecone.index_name
    existing = {idx["name"] for idx in pc.list_indexes()}
    if name not in existing:
        log.info("Creating index %s (dim=%s)", name, settings.embedding.dimensions)
        pc.create_index(
            name=name,
            dimension=settings.embedding.dimensions,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone.cloud,
                region=settings.pinecone.region,
            ),
        )
        import time

        for _ in range(60):
            if pc.describe_index(name).status.get("ready"):
                break
            time.sleep(1)

    index = pc.Index(name)

    if args.recreate:
        log.warning("Deleting namespace %s", FRAMEWORK_NAMESPACE)
        try:
            index.delete(delete_all=True, namespace=FRAMEWORK_NAMESPACE)
        except Exception as ex:  # noqa: BLE001
            log.warning("delete_all: %s", ex)

    for batch in _chunked(vectors, 100):
        index.upsert(vectors=batch, namespace=FRAMEWORK_NAMESPACE)

    log.info("Upserted %s vectors to index=%s namespace=%s", len(vectors), name, FRAMEWORK_NAMESPACE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

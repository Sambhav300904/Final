"""
Compare profile embedding to 43 framework vectors in Pinecone.

Usage:
    python compare/compare_competencies.py --input output/user/embedding_....json
    python compare/compare_competencies.py --user engineer@company.com
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_BUILD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD))

from competencies_data import COMPETENCIES, COMPETENCY_ITEMS  # noqa: E402
from config import FRAMEWORK_NAMESPACE, OUTPUT_DIR, load_settings  # noqa: E402
from level_rules import evaluate_level_rules  # noqa: E402

TOTAL_ITEMS = 43


def compare_vector(
    vector: list[float],
    email: str,
    period: dict[str, str],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    settings = load_settings(require_pinecone=True)
    threshold = threshold if threshold is not None else settings.match_threshold

    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone.api_key)
    index = pc.Index(settings.pinecone.index_name)

    response = index.query(
        vector=vector,
        top_k=TOTAL_ITEMS,
        namespace=FRAMEWORK_NAMESPACE,
        include_metadata=True,
    )
    matches = response.get("matches") or []

    by_id = {item.item_id: item for item in COMPETENCY_ITEMS}
    all_items: list[dict[str, Any]] = []
    matched_items: list[dict[str, Any]] = []

    for m in matches:
        meta = m.get("metadata") or {}
        item_id = int(meta.get("item_id") or 0)
        if not item_id and m.get("id", "").startswith("item-"):
            try:
                item_id = int(str(m["id"]).split("-", 1)[1])
            except ValueError:
                continue
        item = by_id.get(item_id)
        similarity = float(m.get("score") or 0.0)
        row = {
            "item_id": item_id,
            "competency_id": int(meta.get("competency_id") or (item.competency_id if item else 0)),
            "competency_name": meta.get("competency_name")
            or (item.competency_name if item else ""),
            "similarity": round(similarity, 4),
            "above_threshold": similarity >= threshold,
            "text_snippet": (meta.get("text") or (item.text if item else ""))[:120],
        }
        all_items.append(row)
        if row["above_threshold"]:
            matched_items.append(row)

    all_items.sort(key=lambda x: -x["similarity"])
    matched_items.sort(key=lambda x: -x["similarity"])
    matched_count = len(matched_items)

    checked_item_ids = sorted({m["item_id"] for m in matched_items})
    level_evaluation = evaluate_level_rules(checked_item_ids)

    rollups: dict[int, dict[str, Any]] = {}
    for cid, cname in COMPETENCIES.items():
        items_in_c = [i for i in COMPETENCY_ITEMS if i.competency_id == cid]
        matched_in_c = [m for m in matched_items if m["competency_id"] == cid]
        rollups[cid] = {
            "competency_id": cid,
            "competency_name": cname,
            "total_items": len(items_in_c),
            "matched_count": len(matched_in_c),
            "matched_item_ids": sorted(m["item_id"] for m in matched_in_c),
        }

    return {
        "user": email,
        "period": period,
        "telemetry_score": round(100.0 * matched_count / TOTAL_ITEMS, 1),
        "matched_count": matched_count,
        "total_items": TOTAL_ITEMS,
        "match_threshold": threshold,
        "suggested_level_v1": level_evaluation["level"],
        "checked_item_ids": checked_item_ids,
        "level_evaluation": level_evaluation,
        "matched_items": matched_items,
        "all_items": all_items,
        "competency_rollups": list(rollups.values()),
        "disclaimer": (
            "telemetry_only; provisional semantic similarity — not official "
            "framework checkbox credit or manager-approved level"
        ),
    }


def save_result(result: dict[str, Any]) -> Path:
    email = result["user"].replace("@", "_at_")
    period = result["period"]
    path = (
        OUTPUT_DIR
        / email
        / f"scores_{period['start']}_{period['end']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Score profile vs 43 competencies")
    parser.add_argument("--input", type=Path, help="embedding_*.json")
    parser.add_argument("--user", help="Latest embedding under output/")
    args = parser.parse_args()

    if args.input:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    elif args.user:
        safe = args.user.replace("@", "_at_")
        files = sorted((OUTPUT_DIR / safe).glob("embedding_*.json"))
        if not files:
            print(f"No embedding under {OUTPUT_DIR / safe}")
            return 1
        data = json.loads(files[-1].read_text(encoding="utf-8"))
    else:
        print("Provide --input or --user")
        return 1

    result = compare_vector(
        data["vector"], data["email"], data["period"]
    )
    path = save_result(result)
    print(f"Telemetry score: {result['telemetry_score']}%")
    print(f"Matched items: {result['matched_count']}/{TOTAL_ITEMS}")
    print(f"Suggested level (v1): {result['suggested_level_v1']}")
    print(f"Saved: {path}")
    print("\nTop matches:")
    for m in result["matched_items"][:10]:
        print(
            f"  item {m['item_id']} ({m['competency_name'][:40]}…): "
            f"{m['similarity']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

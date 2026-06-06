"""
Embed a profile narrative vector (same model as competency ingest).

Usage:
    python embed_profile.py --input output/user/profile_....json
    python embed_profile.py --user engineer@company.com --days 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUILD))

from config import OUTPUT_DIR, load_settings  # noqa: E402
from create_profile import normalize_narrative  # noqa: E402
from embedding_client import EmbeddingClient  # noqa: E402


def embed_narrative(text: str) -> list[float]:
    settings = load_settings()
    client = EmbeddingClient(settings.embedding)
    return client.embed_text(text.strip())


def save_embedding(
    email: str, period: dict, vector: list[float], narrative: str
) -> Path:
    safe = email.replace("@", "_at_")
    path = (
        OUTPUT_DIR
        / safe
        / f"embedding_{period['start']}_{period['end']}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "email": email,
                "period": period,
                "dimensions": len(vector),
                "vector": vector,
                "narrative_preview": narrative[:500],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_profile_bundle(path: Path) -> tuple[str, dict, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    email = data["email"]
    period = data["period"]
    narrative = normalize_narrative(data["profile"]["profile_narrative"])
    return email, period, narrative


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed profile narrative")
    parser.add_argument("--input", type=Path, help="profile_*.json from create_profile")
    parser.add_argument("--user", help="Find latest profile under output/")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    if args.input:
        email, period, narrative = load_profile_bundle(args.input)
    elif args.user:
        safe = args.user.replace("@", "_at_")
        profiles = sorted((OUTPUT_DIR / safe).glob("profile_*.json"))
        if not profiles:
            print(f"No profile found under {OUTPUT_DIR / safe}")
            return 1
        email, period, narrative = load_profile_bundle(profiles[-1])
    else:
        print("Provide --input or --user")
        return 1

    vector = embed_narrative(narrative)
    path = save_embedding(email, period, vector, narrative)
    print(f"Saved embedding ({len(vector)} dims): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

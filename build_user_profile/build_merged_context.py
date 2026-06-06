"""
Fetch MySQL telemetry, optional evidence, and optional Copilot usage into one merged context.

Usage:
    python build_merged_context.py --user engineer@company.com --days 30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

_BUILD = Path(__file__).resolve().parent
_REPO = _BUILD.parent
sys.path.insert(0, str(_BUILD.parent / "cursor_mysql_sync"))
sys.path.insert(0, str(_BUILD))
sys.path.insert(0, str(_REPO / "evidence_evaluator"))

from config import OUTPUT_DIR  # noqa: E402
from fetch_copilot_data import fetch_copilot_context  # noqa: E402
from fetch_user_data import fetch_user_context, resolve_period  # noqa: E402
from summarize_evidence import load_evidence_bundle, save_bundle as save_evidence_bundle  # noqa: E402
from validate_evidence import has_relevant_evidence  # noqa: E402


def normalize_merged_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Accept legacy flat telemetry JSON or already-merged context."""
    if "telemetry" in raw:
        raw.setdefault("copilot", None)
        if "data_sources" not in raw:
            sources = ["telemetry"]
            if has_relevant_evidence(raw.get("evidence")):
                sources.append("evidence")
            if raw.get("copilot"):
                sources.append("copilot")
            raw["data_sources"] = sources
        elif raw.get("evidence") and not has_relevant_evidence(raw.get("evidence")):
            raw["evidence"] = None
            raw["data_sources"] = [
                s for s in (raw.get("data_sources") or []) if s != "evidence"
            ]
        return raw
    return {
        "email": raw["email"],
        "period": raw["period"],
        "data_sources": ["telemetry"],
        "telemetry": raw,
        "evidence": None,
        "copilot": None,
    }


def build_merged_context(
    email: str,
    period_start: date,
    period_end: date,
    *,
    include_evidence: bool = True,
    include_copilot: bool = True,
) -> dict[str, Any]:
    email = email.strip().lower()
    telemetry = fetch_user_context(email, period_start, period_end)

    evidence = None
    if include_evidence:
        bundle = load_evidence_bundle(email, period_start, period_end)
        if has_relevant_evidence(bundle):
            evidence = bundle

    copilot = None
    if include_copilot:
        copilot = fetch_copilot_context(email, period_start, period_end)

    data_sources = ["telemetry"]
    if has_relevant_evidence(evidence):
        data_sources.append("evidence")
    if copilot:
        data_sources.append("copilot")

    return {
        "email": email,
        "period": telemetry["period"],
        "data_sources": data_sources,
        "telemetry": telemetry,
        "evidence": evidence,
        "copilot": copilot,
    }


def context_path_for(email: str, period: dict[str, str], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    safe = email.replace("@", "_at_")
    return out_dir / safe / f"context_{period['start']}_{period['end']}.json"


def load_saved_context(email: str, period_start: date, period_end: date) -> dict[str, Any] | None:
    period = {
        "start": period_start.isoformat(),
        "end": period_end.isoformat(),
    }
    path = context_path_for(email.strip().lower(), period)
    if not path.is_file():
        return None
    return normalize_merged_context(json.loads(path.read_text(encoding="utf-8")))


def save_merged_context(merged: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    email = merged["email"].replace("@", "_at_")
    period = merged["period"]
    path = out_dir / email / f"context_{period['start']}_{period['end']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch telemetry + optional evidence + Copilot into merged context JSON"
    )
    parser.add_argument("--user", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s))
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s))
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip evidence folder (telemetry only)",
    )
    parser.add_argument(
        "--no-copilot",
        action="store_true",
        help="Skip Copilot MySQL data",
    )
    args = parser.parse_args()

    start, end = resolve_period(args.days, args.start, args.end)
    merged = build_merged_context(
        args.user,
        start,
        end,
        include_evidence=not args.no_evidence,
        include_copilot=not args.no_copilot,
    )
    path = save_merged_context(merged)
    print(f"Saved: {path}")
    print(f"Sources: {', '.join(merged['data_sources'])}")
    if merged.get("evidence"):
        ev = merged["evidence"]
        save_evidence_bundle(ev)
        n_ok = len(ev.get("files_processed") or [])
        n_rej = len(ev.get("files_rejected") or [])
        n_fail = len(ev.get("files_failed") or [])
        n_up = len(ev.get("files_uploaded") or [])
        print(
            f"Evidence: {n_ok} kept, {n_rej} rejected, {n_fail} failed "
            f"(of {n_up} uploaded)"
        )
    else:
        print(
            "Evidence: none (folder missing, empty, or all files rejected as irrelevant)"
        )
    if merged.get("copilot"):
        cu = merged["copilot"].get("daily_usage") or {}
        print(
            f"Copilot: {cu.get('days_in_window', 0)} days, "
            f"{cu.get('interactions', 0)} interactions"
        )
    else:
        print("Copilot: none (no MySQL rows for this email/period)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
AI Adoption V1 — profile pipeline (MySQL + evidence → profile → embed → score).

Sync Cursor data separately:
    python cursor_mysql_sync/main.py --days 30

Profile pipeline:
    python main.py ingest              # once: 43 competencies → Pinecone
    python main.py profile --user x@y.com --days 30
    python main.py all --user x@y.com --days 30

Evidence-only (standalone):
    python main.py evidence --user x@y.com --days 7

Batch (all team members, parallel):
    python batch_pipeline.py --init-profile-schema
    python batch_pipeline.py --days 7 --workers 4 --require-events
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD = REPO_ROOT / "build_user_profile"
EVIDENCE = REPO_ROOT / "evidence_evaluator"


def _run_build(script: str, extra: list[str]) -> int:
    path = BUILD / script
    cmd = [sys.executable, str(path), *extra]
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(BUILD))


def _run_evidence(extra: list[str]) -> int:
    path = EVIDENCE / "summarize_evidence.py"
    cmd = [sys.executable, str(path), *extra]
    print(">", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(EVIDENCE))


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Adoption profile pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Embed 43 items → Pinecone (one-time)")
    p_ingest.add_argument("--dry-run", action="store_true")
    p_ingest.add_argument("--recreate", action="store_true")

    period_parent = argparse.ArgumentParser(add_help=False)
    period_parent.add_argument("--user", required=True)
    period_parent.add_argument("--days", type=int, default=30)
    period_parent.add_argument("--start", type=date.fromisoformat)
    period_parent.add_argument("--end", type=date.fromisoformat)

    for name in ("profile", "fetch", "merge", "embed", "score", "all", "evidence"):
        help_text = {
            "profile": "Build profile from MySQL telemetry + optional evidence",
            "fetch": "Fetch telemetry JSON from MySQL (telemetry only)",
            "merge": "Fetch telemetry + summarize evidence → merged context JSON",
            "embed": "Embed profile narrative",
            "score": "Score profile vs 43 competencies",
            "all": "merge → profile → embed → score",
            "evidence": "Summarize files in evidence_evaluator/evidences/<email>/",
        }[name]
        p = sub.add_parser(name, parents=[period_parent], help=help_text)
        if name in ("profile", "all", "merge"):
            p.add_argument(
                "--no-evidence",
                action="store_true",
                help="Skip evidence folder (telemetry only)",
            )
            p.add_argument(
                "--no-copilot",
                action="store_true",
                help="Skip GitHub Copilot MySQL data in merge",
            )
        if name in ("profile", "all"):
            p.add_argument("--no-llm", action="store_true", help="Template profile only")

    p_batch = sub.add_parser(
        "batch",
        help="All dim_cursor_team_members → pipeline → MySQL snapshots",
    )
    p_batch.add_argument("--days", type=int, default=30)
    p_batch.add_argument("--start", type=date.fromisoformat)
    p_batch.add_argument("--end", type=date.fromisoformat)
    p_batch.add_argument("--workers", type=int, default=4)
    p_batch.add_argument("--require-events", action="store_true")
    p_batch.add_argument("--no-evidence", action="store_true")
    p_batch.add_argument("--no-copilot", action="store_true")
    p_batch.add_argument("--no-llm", action="store_true")
    p_batch.add_argument("--user", action="append", dest="users")
    p_batch.add_argument("--max-users", type=int, default=0)
    p_batch.add_argument("--skip-existing", action="store_true")
    p_batch.add_argument("--dry-run", action="store_true")
    p_batch.add_argument("--init-profile-schema", action="store_true")
    p_batch.add_argument("--stop-on-error", action="store_true")

    args = parser.parse_args()

    if args.command == "batch":
        cmd = [sys.executable, str(REPO_ROOT / "batch_pipeline.py"), "--days", str(args.days)]
        if args.start:
            cmd.extend(["--start", args.start.isoformat()])
        if args.end:
            cmd.extend(["--end", args.end.isoformat()])
        cmd.extend(["--workers", str(args.workers)])
        if args.require_events:
            cmd.append("--require-events")
        if args.no_evidence:
            cmd.append("--no-evidence")
        if args.no_copilot:
            cmd.append("--no-copilot")
        if args.no_llm:
            cmd.append("--no-llm")
        if args.users:
            for u in args.users:
                cmd.extend(["--user", u])
        if args.max_users:
            cmd.extend(["--max-users", str(args.max_users)])
        if args.skip_existing:
            cmd.append("--skip-existing")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.init_profile_schema:
            cmd.append("--init-profile-schema")
        if args.stop_on_error:
            cmd.append("--stop-on-error")
        print(">", " ".join(cmd))
        return subprocess.call(cmd, cwd=str(REPO_ROOT))

    if args.command == "ingest":
        extra = []
        if getattr(args, "dry_run", False):
            extra.append("--dry-run")
        if getattr(args, "recreate", False):
            extra.append("--recreate")
        return _run_build("competencies/ingest_competencies.py", extra)

    period_args: list[str] = ["--user", args.user, "--days", str(args.days)]
    if args.start:
        period_args.extend(["--start", args.start.isoformat()])
    if args.end:
        period_args.extend(["--end", args.end.isoformat()])

    merge_profile_args = list(period_args)
    if getattr(args, "no_evidence", False):
        merge_profile_args.append("--no-evidence")
    if getattr(args, "no_copilot", False):
        merge_profile_args.append("--no-copilot")

    if args.command == "evidence":
        return _run_evidence(period_args)

    if args.command == "fetch":
        return _run_build("fetch_user_data.py", period_args)

    if args.command == "merge":
        return _run_build("build_merged_context.py", merge_profile_args)

    if args.command == "profile":
        extra = list(merge_profile_args)
        if args.no_llm:
            extra.append("--no-llm")
        return _run_build("create_profile.py", extra)

    if args.command == "embed":
        return _run_build("embed_profile.py", ["--user", args.user])

    if args.command == "score":
        return _run_build("compare/compare_competencies.py", ["--user", args.user])

    if args.command == "all":
        merge_args = list(merge_profile_args)
        profile_args = list(merge_profile_args) + ["--use-saved-context"]
        if args.no_llm:
            profile_args.append("--no-llm")
        steps = [
            ("build_merged_context.py", merge_args),
            ("create_profile.py", profile_args),
            ("embed_profile.py", ["--user", args.user]),
            ("compare/compare_competencies.py", ["--user", args.user]),
        ]
        for script, extra in steps:
            code = _run_build(script, extra)
            if code != 0:
                return code
        sys.path.insert(0, str(BUILD))
        from fetch_user_data import resolve_period  # noqa: E402
        from profile_store import persist_user_snapshot, profile_tables_exist  # noqa: E402

        period_start, period_end = resolve_period(args.days, args.start, args.end)
        if profile_tables_exist():
            persist_user_snapshot(args.user, period_start, period_end)
            print(
                f"Saved to MySQL: fact_user_proficiency_snapshot "
                f"({period_start} → {period_end})"
            )
        else:
            print(
                "MySQL snapshot table not found — run "
                "python batch_pipeline.py --init-profile-schema"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

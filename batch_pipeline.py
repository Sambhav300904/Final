"""
Batch profile pipeline for all team members (parallel workers).

Run from repo root:
    python batch_pipeline.py --init-profile-schema
    python batch_pipeline.py --days 7 --workers 4
    python batch_pipeline.py --days 7 --workers 4 --require-events
    python batch_pipeline.py --days 7 --dry-run

Users are loaded from dim_cursor_team_members (optional: only with usage events).
Results upserted to fact_user_proficiency_snapshot.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD = REPO_ROOT / "build_user_profile"
# build_user_profile must be first so `config` resolves correctly
sys.path.insert(0, str(BUILD))

from config import load_mysql_settings  # noqa: E402
from fetch_user_data import resolve_period  # noqa: E402
from pipeline_runner import build_period_args, run_full_pipeline_for_user  # noqa: E402
from profile_store import (  # noqa: E402
    build_row_from_artifacts,
    finish_batch_run,
    init_profile_schema,
    list_team_members,
    load_artifacts,
    profile_tables_exist,
    snapshot_exists,
    start_batch_run,
    upsert_snapshot,
)


@dataclass
class UserJob:
    email: str
    days: int
    start_iso: str
    end_iso: str
    period_start: date
    period_end: date
    no_evidence: bool
    no_copilot: bool
    no_llm: bool
    batch_run_id: int
    python_exe: str


@dataclass
class UserResult:
    email: str
    ok: bool
    error: str = ""
    score: float | None = None
    level: str | None = None


def _process_user(job: UserJob) -> UserResult:
    period_args = build_period_args(
        job.email,
        job.days,
        job.start_iso,
        job.end_iso,
        no_evidence=job.no_evidence,
        no_copilot=job.no_copilot,
    )
    try:
        code, log = run_full_pipeline_for_user(
            job.email,
            period_args,
            no_llm=job.no_llm,
            python_exe=job.python_exe,
            start_iso=job.start_iso,
            end_iso=job.end_iso,
        )
        if code != 0:
            row = build_row_from_artifacts(
                user_email=job.email,
                period_start=job.period_start,
                period_end=job.period_end,
                profile_bundle=None,
                scores=None,
                profile_path=None,
                scores_path=None,
                batch_run_id=job.batch_run_id,
                pipeline_status="failed",
                error_message=log[-4000:],
            )
            upsert_snapshot(row)
            return UserResult(job.email, False, error=log[-500:])

        profile_bundle, scores, profile_path, scores_path = load_artifacts(
            job.email, job.period_start, job.period_end
        )
        row = build_row_from_artifacts(
            user_email=job.email,
            period_start=job.period_start,
            period_end=job.period_end,
            profile_bundle=profile_bundle,
            scores=scores,
            profile_path=profile_path,
            scores_path=scores_path,
            batch_run_id=job.batch_run_id,
            pipeline_status="success",
        )
        upsert_snapshot(row)
        return UserResult(
            job.email,
            True,
            score=scores.get("telemetry_score") if scores else None,
            level=scores.get("suggested_level_v1") if scores else None,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        try:
            row = build_row_from_artifacts(
                user_email=job.email,
                period_start=job.period_start,
                period_end=job.period_end,
                profile_bundle=None,
                scores=None,
                profile_path=None,
                scores_path=None,
                batch_run_id=job.batch_run_id,
                pipeline_status="failed",
                error_message=msg,
            )
            upsert_snapshot(row)
        except Exception:
            pass
        return UserResult(job.email, False, error=msg)


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch pipeline for all team members")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument(
        "--require-events",
        action="store_true",
        help="Only members with usage events in the period",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip evidence folders (telemetry only)",
    )
    parser.add_argument(
        "--no-copilot",
        action="store_true",
        help="Skip GitHub Copilot MySQL data in merge",
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--user", action="append", dest="users", help="Subset of emails")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", help="Skip successful snapshots")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--init-profile-schema", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    load_mysql_settings()

    if args.init_profile_schema:
        init_profile_schema()
        print("Profile schema applied.")
        return 0

    if not args.dry_run and not profile_tables_exist():
        print(
            "Profile tables missing. Run:\n"
            "  python batch_pipeline.py --init-profile-schema --days 7"
        )
        return 1

    period_start, period_end = resolve_period(args.days, args.start, args.end)
    start_iso = period_start.isoformat()
    end_iso = period_end.isoformat()

    if args.users:
        users = sorted({u.strip().lower() for u in args.users if u.strip()})
    else:
        users = list_team_members(
            period_start=period_start,
            period_end=period_end,
            require_events=args.require_events,
        )

    if args.skip_existing:
        users = [
            u
            for u in users
            if not snapshot_exists(u, period_start, period_end)
        ]

    if args.max_users > 0:
        users = users[: args.max_users]

    workers = max(1, min(args.workers, 16))
    print(f"Period: {period_start} .. {period_end}")
    print(f"Users to process: {len(users)}")
    print(f"Parallel workers: {workers}")
    print(f"Require events in period: {args.require_events}")

    if args.dry_run:
        for u in users[:50]:
            print(f"  {u}")
        if len(users) > 50:
            print(f"  ... and {len(users) - 50} more")
        return 0

    if not users:
        print("No users to process.")
        return 0

    config = {
        "days": args.days,
        "workers": workers,
        "require_events": args.require_events,
        "no_evidence": args.no_evidence,
        "no_copilot": args.no_copilot,
        "no_llm": args.no_llm,
        "skip_existing": args.skip_existing,
    }
    batch_id = start_batch_run(period_start, period_end, workers, config)
    print(f"Batch run id: {batch_id}")

    jobs = [
        UserJob(
            email=email,
            days=args.days,
            start_iso=start_iso,
            end_iso=end_iso,
            period_start=period_start,
            period_end=period_end,
            no_evidence=args.no_evidence,
            no_copilot=args.no_copilot,
            no_llm=args.no_llm,
            batch_run_id=batch_id,
            python_exe=sys.executable,
        )
        for email in users
    ]

    ok = 0
    failed = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_user, job): job.email for job in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            email = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                result = UserResult(email, False, error=str(exc))
            if result.ok:
                ok += 1
                print(
                    f"[{i}/{len(users)}] OK {email} "
                    f"score={result.score} level={result.level}",
                    flush=True,
                )
            else:
                failed += 1
                errors.append(f"{email}: {result.error}")
                print(f"[{i}/{len(users)}] FAIL {email}: {result.error[:200]}", flush=True)
                if args.stop_on_error:
                    for f in futures:
                        if f is not fut and not f.done():
                            f.cancel()
                    break

    status = "completed" if failed == 0 else "partial" if ok > 0 else "failed"
    summary = f"ok={ok} failed={failed} total={len(users)} workers={workers}"
    if errors:
        summary += "\n" + "\n".join(errors[:30])
    finish_batch_run(
        batch_id,
        status=status,
        users_total=len(users),
        users_ok=ok,
        users_failed=failed,
        message=summary,
    )
    print(f"\nBatch {batch_id} {status}: {ok} succeeded, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

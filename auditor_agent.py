"""
Auditor Agent — 7-day usage-pattern audit (MySQL, day-wise).

Third agent in the AI Adoption pipeline (after Evaluator + Validator).
Self-contained: this single file is the whole agent. It reuses the existing
MySQL connection helper (cursor_mysql_sync/db.py) and reads ONLY the last 7
rows per user from `fact_cursor_daily_usage` — no 40-50 day baseline pull.

What it answers from the last 7 days only:
  - Which day(s) drove the week           -> top day, top-day share, top-2 share
  - Is usage lumpy / concentrated         -> concentration index, active-day spread
  - End-of-period stuffing                -> share of activity on the last 1-2 days
  - Spike day(s) inside the window        -> days far above the week's median day

Output per user (stable JSON):
  {
    "user_email", "period_start", "period_end",
    "audit_status": "ok | review | high_risk | no_data",
    "risk_score": 0.0-1.0,
    "flags": [{"code", "severity", "detail"}, ...],
    "metrics": {...},
    "daily": [{"date", "activity", "is_active", ...}, ...],
    "spike_days": ["YYYY-MM-DD", ...]
  }

Usage:
    python auditor_agent.py --user x@y.com --days 7
    python auditor_agent.py --user x@y.com --start 2026-05-19 --end 2026-05-25
    python auditor_agent.py --all --days 7 --workers 4
    python auditor_agent.py --all --days 7 --json audit_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "cursor_mysql_sync"))

from config import load_mysql_settings  # noqa: E402
from db import get_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables (sensible defaults; override via CLI where useful)
# ---------------------------------------------------------------------------

WINDOW_DAYS = 7

# Per-day "activity" = weighted sum of the columns that represent real work.
# Tabs/applies capture interaction; accepted lines capture retained output.
ACTIVITY_WEIGHTS: dict[str, float] = {
    "total_lines_added": 1.0,
    "total_applies": 3.0,
    "total_accepts": 3.0,
    "total_tabs_accepted": 1.0,
}

# Thresholds for the within-week pattern flags.
SINGLE_DAY_SHARE_HIGH = 0.60   # one day is >=60% of the whole week
TOP2_SHARE_LUMPY = 0.80        # top 2 days are >=80% of the week ("lumpy")
LAST2_SHARE_END_LOADED = 0.75  # last 2 days are >=75% of the week
SPIKE_FACTOR = 2.0             # a day > 2x the week's median day is a spike
MIN_ACTIVITY_FOR_FLAGS = 1.0   # below this total, treat as no real signal


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def resolve_period(days: int, start: date | None, end: date | None) -> tuple[date, date]:
    """Default window ends yesterday and spans `days` (clamped to >=1)."""
    if end is None:
        end = date.today() - timedelta(days=1)
    if start is None:
        start = end - timedelta(days=max(days, 1) - 1)
    if start > end:
        raise ValueError("start must be <= end")
    return start, end


# ---------------------------------------------------------------------------
# Data access (last 7 rows only)
# ---------------------------------------------------------------------------

_DAILY_COLUMNS = (
    "is_active",
    "total_lines_added",
    "total_lines_deleted",
    "accepted_lines_added",
    "total_applies",
    "total_accepts",
    "total_rejects",
    "total_tabs_shown",
    "total_tabs_accepted",
)


def list_users_in_window(period_start: date, period_end: date) -> list[str]:
    """Distinct users that have any daily-usage row inside the 7-day window."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT LOWER(TRIM(email))
            FROM fact_cursor_daily_usage
            WHERE usage_date BETWEEN %s AND %s
            ORDER BY 1
            """,
            (period_start, period_end),
        )
        return [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()


def fetch_daily_rows(
    email: str, period_start: date, period_end: date
) -> dict[date, dict[str, int]]:
    """Return {usage_date: {column: value}} for the window (only rows present)."""
    email = email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            f"""
            SELECT usage_date, {", ".join(_DAILY_COLUMNS)}
            FROM fact_cursor_daily_usage
            WHERE LOWER(TRIM(email)) = %s
              AND usage_date BETWEEN %s AND %s
            ORDER BY usage_date
            """,
            (email, period_start, period_end),
        )
        out: dict[date, dict[str, int]] = {}
        for row in cur.fetchall():
            d = row.pop("usage_date")
            out[d] = {k: int(row.get(k) or 0) for k in _DAILY_COLUMNS}
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Pure analysis on the 7-day series
# ---------------------------------------------------------------------------

def _activity_of(day_row: dict[str, int]) -> float:
    return float(sum(day_row.get(col, 0) * w for col, w in ACTIVITY_WEIGHTS.items()))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _gini(values: list[float]) -> float:
    """Concentration of activity across days. 0 = perfectly even, ->1 = all on one day."""
    vals = [v for v in values if v >= 0]
    total = sum(vals)
    n = len(vals)
    if n == 0 or total == 0:
        return 0.0
    s = sorted(vals)
    cum = 0.0
    for i, v in enumerate(s, start=1):
        cum += i * v
    gini = (2.0 * cum) / (n * total) - (n + 1.0) / n
    return max(0.0, min(1.0, round(gini, 4)))


def analyze_window(
    email: str,
    period_start: date,
    period_end: date,
    daily_rows: dict[date, dict[str, int]],
) -> dict[str, Any]:
    """Build the audit verdict from the 7-day daily series (zero-filled)."""
    # Zero-fill every day in the window so missing days count as inactive.
    span = (period_end - period_start).days + 1
    days = [period_start + timedelta(days=i) for i in range(span)]

    daily: list[dict[str, Any]] = []
    activities: list[float] = []
    for d in days:
        row = daily_rows.get(d, {col: 0 for col in _DAILY_COLUMNS})
        act = _activity_of(row)
        activities.append(act)
        daily.append(
            {
                "date": d.isoformat(),
                "activity": round(act, 2),
                "is_active": int(row.get("is_active", 0) or (act > 0)),
                "total_lines_added": row.get("total_lines_added", 0),
                "total_applies": row.get("total_applies", 0),
                "total_accepts": row.get("total_accepts", 0),
                "total_tabs_accepted": row.get("total_tabs_accepted", 0),
            }
        )

    total = float(sum(activities))
    active_days = sum(1 for a in activities if a > 0)
    median_day = _median(activities)
    mean_day = total / span if span else 0.0
    max_value = max(activities) if activities else 0.0
    max_idx = activities.index(max_value) if activities else 0
    max_day = days[max_idx].isoformat() if activities else None

    # Shares
    def _share(x: float) -> float:
        return round(x / total, 4) if total > 0 else 0.0

    sorted_acts = sorted(activities, reverse=True)
    top1_share = _share(sorted_acts[0]) if sorted_acts else 0.0
    top2_share = _share(sum(sorted_acts[:2])) if sorted_acts else 0.0
    last_day_share = _share(activities[-1]) if activities else 0.0
    last2_share = _share(sum(activities[-2:])) if activities else 0.0

    # Spike days: a day clearly above the typical (median) day for this same week.
    # If the median day is 0 (very sparse week), the single biggest active day counts.
    spike_days: list[str] = []
    for i, a in enumerate(activities):
        if a <= 0:
            continue
        if median_day > 0:
            is_spike = a >= SPIKE_FACTOR * median_day
        else:
            is_spike = a == max_value
        if is_spike:
            spike_days.append(days[i].isoformat())

    concentration = _gini(activities)
    # Spread = how many days carry a meaningful slice (>=10%) of the week.
    spread_days = sum(1 for a in activities if total > 0 and a / total >= 0.10)

    metrics = {
        "window_days": span,
        "total_activity": round(total, 2),
        "active_days": active_days,
        "mean_day": round(mean_day, 2),
        "median_day": round(median_day, 2),
        "max_day": max_day,
        "max_day_activity": round(max_value, 2),
        "top1_share": top1_share,
        "top2_share": top2_share,
        "last_day_share": last_day_share,
        "last2_share": last2_share,
        "concentration_gini": concentration,
        "spread_days": spread_days,
    }

    # -------------------- flags --------------------
    flags: list[dict[str, str]] = []
    if total < MIN_ACTIVITY_FOR_FLAGS:
        return {
            "user_email": email,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "audit_status": "no_data",
            "risk_score": 0.0,
            "flags": [],
            "metrics": metrics,
            "daily": daily,
            "spike_days": [],
        }

    if top1_share >= SINGLE_DAY_SHARE_HIGH:
        flags.append(
            {
                "code": "single_day_dominant",
                "severity": "high",
                "detail": f"{int(round(top1_share * 100))}% of the week's activity on {max_day}.",
            }
        )
    if top2_share >= TOP2_SHARE_LUMPY:
        flags.append(
            {
                "code": "lumpy_usage",
                "severity": "medium",
                "detail": f"Top 2 days account for {int(round(top2_share * 100))}% of the week.",
            }
        )
    if last2_share >= LAST2_SHARE_END_LOADED:
        flags.append(
            {
                "code": "end_of_period_loaded",
                "severity": "high",
                "detail": f"{int(round(last2_share * 100))}% of activity in the last 2 days of the window.",
            }
        )
    if len(spike_days) >= 1 and median_day > 0:
        flags.append(
            {
                "code": "spike_day",
                "severity": "medium",
                "detail": f"Spike day(s) >= {SPIKE_FACTOR:g}x the week's median day: {', '.join(spike_days)}.",
            }
        )
    if active_days <= 1:
        flags.append(
            {
                "code": "single_active_day",
                "severity": "medium",
                "detail": "Only one active day in the 7-day window.",
            }
        )

    # -------------------- risk score + status --------------------
    severity_weight = {"high": 0.5, "medium": 0.25, "low": 0.1}
    risk_score = min(1.0, sum(severity_weight.get(f["severity"], 0.1) for f in flags))
    risk_score = round(risk_score, 3)

    has_high = any(f["severity"] == "high" for f in flags)
    if has_high and concentration >= 0.5:
        status = "high_risk"
    elif flags:
        status = "review"
    else:
        status = "ok"

    return {
        "user_email": email,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "audit_status": status,
        "risk_score": risk_score,
        "flags": flags,
        "metrics": metrics,
        "daily": daily,
        "spike_days": spike_days,
    }


def audit_user(email: str, period_start: date, period_end: date) -> dict[str, Any]:
    """Fetch + analyze a single user. Safe to call from worker threads."""
    rows = fetch_daily_rows(email, period_start, period_end)
    return analyze_window(email, period_start, period_end, rows)


def audit_users(
    emails: list[str],
    period_start: date,
    period_end: date,
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Audit many users; runs in parallel threads (each gets its own DB conn)."""
    if workers <= 1 or len(emails) <= 1:
        return [audit_user(e, period_start, period_end) for e in emails]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_email = {
            pool.submit(audit_user, e, period_start, period_end): e for e in emails
        }
        for fut in as_completed(future_to_email):
            email = future_to_email[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "user_email": email,
                        "period_start": period_start.isoformat(),
                        "period_end": period_end.isoformat(),
                        "audit_status": "error",
                        "risk_score": 0.0,
                        "flags": [],
                        "metrics": {},
                        "daily": [],
                        "spike_days": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    results.sort(key=lambda r: (-(r.get("risk_score") or 0.0), r["user_email"]))
    return results


# ---------------------------------------------------------------------------
# CLI / console reporting
# ---------------------------------------------------------------------------

_STATUS_ORDER = {"high_risk": 0, "review": 1, "error": 2, "ok": 3, "no_data": 4}


def _print_one(result: dict[str, Any]) -> None:
    m = result.get("metrics", {})
    print(f"\n=== {result['user_email']}  [{result['period_start']} .. {result['period_end']}]")
    print(f"  status      : {result['audit_status']}  (risk {result['risk_score']})")
    if result.get("error"):
        print(f"  error       : {result['error']}")
        return
    print(
        f"  activity    : total={m.get('total_activity')}  active_days={m.get('active_days')}"
        f"  median_day={m.get('median_day')}"
    )
    print(
        f"  top day     : {m.get('max_day')} "
        f"({int(round((m.get('top1_share') or 0) * 100))}% of week)  "
        f"top2={int(round((m.get('top2_share') or 0) * 100))}%  "
        f"last2={int(round((m.get('last2_share') or 0) * 100))}%"
    )
    print(
        f"  shape       : concentration={m.get('concentration_gini')}  "
        f"spread_days={m.get('spread_days')}  spikes={result.get('spike_days')}"
    )
    if result.get("flags"):
        print("  flags       :")
        for f in result["flags"]:
            print(f"    - [{f['severity']}] {f['code']}: {f['detail']}")
    else:
        print("  flags       : none")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditor Agent — 7-day usage-pattern audit (MySQL)."
    )
    parser.add_argument("--user", help="Single user email to audit.")
    parser.add_argument("--all", action="store_true", help="Audit all users with data in the window.")
    parser.add_argument("--days", type=int, default=WINDOW_DAYS, help="Window length (default 7).")
    parser.add_argument("--start", type=_parse_date, help="Window start (YYYY-MM-DD).")
    parser.add_argument("--end", type=_parse_date, help="Window end (YYYY-MM-DD).")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for --all.")
    parser.add_argument("--json", dest="json_path", help="Write full results to this JSON file.")
    parser.add_argument("--only-flagged", action="store_true", help="Print only review/high_risk users.")
    args = parser.parse_args()

    if not args.user and not args.all:
        parser.error("Provide --user EMAIL or --all.")

    load_mysql_settings()
    start, end = resolve_period(args.days, args.start, args.end)

    if args.user:
        results = [audit_user(args.user, start, end)]
    else:
        emails = list_users_in_window(start, end)
        print(f"Auditing {len(emails)} user(s) for {start} .. {end} (workers={args.workers})")
        results = audit_users(emails, start, end, workers=args.workers)

    shown = results
    if args.only_flagged:
        shown = [r for r in results if r["audit_status"] in ("review", "high_risk")]

    shown_sorted = sorted(
        shown, key=lambda r: (_STATUS_ORDER.get(r["audit_status"], 9), -(r.get("risk_score") or 0.0))
    )
    for r in shown_sorted:
        _print_one(r)

    counts: dict[str, int] = {}
    for r in results:
        counts[r["audit_status"]] = counts.get(r["audit_status"], 0) + 1
    print("\n--- summary ---")
    for status in ("high_risk", "review", "ok", "no_data", "error"):
        if counts.get(status):
            print(f"  {status:9}: {counts[status]}")

    if args.json_path:
        out = Path(args.json_path)
        out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {len(results)} result(s) to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

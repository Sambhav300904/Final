"""
Fetch Cursor telemetry from MySQL for one user and period.

Usage:
    python fetch_user_data.py --user engineer@company.com --days 30
    python fetch_user_data.py --user engineer@company.com --start 2026-04-01 --end 2026-05-25
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUILD.parent / "cursor_mysql_sync"))
sys.path.insert(0, str(_BUILD))

from config import OUTPUT_DIR, load_mysql_settings  # noqa: E402
from db import get_connection  # noqa: E402


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def resolve_period(days: int, start: date | None, end: date | None) -> tuple[date, date]:
    if end is None:
        end = date.today() - timedelta(days=1)
    if start is None:
        start = end - timedelta(days=max(days, 1) - 1)
    if start > end:
        raise ValueError("start must be <= end")
    return start, end


def list_users_in_period(period_start: date, period_end: date) -> list[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT LOWER(TRIM(user_email))
            FROM fact_cursor_usage_events
            WHERE event_at >= %s AND event_at < %s + INTERVAL 1 DAY
            ORDER BY 1
            """,
            (period_start, period_end),
        )
        return [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()


def fetch_user_context(email: str, period_start: date, period_end: date) -> dict[str, Any]:
    email = email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(
            "SELECT email, name, role FROM dim_cursor_team_members WHERE LOWER(email) = %s",
            (email,),
        )
        member = cur.fetchone() or {"email": email, "name": None, "role": None}

        cur.execute(
            """
            SELECT
                COUNT(*) AS days_in_window,
                SUM(is_active) AS active_days,
                SUM(total_lines_added) AS lines_added,
                SUM(total_lines_deleted) AS lines_deleted,
                SUM(accepted_lines_added) AS lines_accepted,
                SUM(total_applies) AS applies,
                SUM(total_accepts) AS accepts,
                SUM(total_rejects) AS rejects,
                SUM(total_tabs_shown) AS tabs_shown,
                SUM(total_tabs_accepted) AS tabs_accepted
            FROM fact_cursor_daily_usage
            WHERE LOWER(email) = %s AND usage_date BETWEEN %s AND %s
            """,
            (email, period_start, period_end),
        )
        daily = cur.fetchone() or {}

        cur.execute(
            """
            SELECT model, COUNT(*) AS events,
                   SUM(total_tokens) AS tokens, SUM(cost_usd) AS cost
            FROM fact_cursor_usage_events
            WHERE LOWER(user_email) = %s
              AND event_at >= %s AND event_at < %s + INTERVAL 1 DAY
            GROUP BY model ORDER BY events DESC LIMIT 8
            """,
            (email, period_start, period_end),
        )
        models = list(cur.fetchall())

        cur.execute(
            """
            SELECT kind, COUNT(*) AS events
            FROM fact_cursor_usage_events
            WHERE LOWER(user_email) = %s
              AND event_at >= %s AND event_at < %s + INTERVAL 1 DAY
            GROUP BY kind ORDER BY events DESC LIMIT 8
            """,
            (email, period_start, period_end),
        )
        kinds = list(cur.fetchall())

        cur.execute(
            """
            SELECT COUNT(*) AS total_events,
                   SUM(total_tokens) AS total_tokens,
                   SUM(cost_usd) AS total_cost
            FROM fact_cursor_usage_events
            WHERE LOWER(user_email) = %s
              AND event_at >= %s AND event_at < %s + INTERVAL 1 DAY
            """,
            (email, period_start, period_end),
        )
        events_summary = cur.fetchone() or {}

        cur.execute(
            """
            SELECT dimension, label, count, period_start, period_end
            FROM fact_insight_distribution
            WHERE scope = 'user' AND LOWER(user_email) = %s
              AND period_end >= %s AND period_start <= %s
            ORDER BY period_end DESC, dimension, count DESC
            """,
            (email, period_start, period_end),
        )
        insight_rows = cur.fetchall()
        if insight_rows:
            ps = insight_rows[0]["period_start"]
            pe = insight_rows[0]["period_end"]
            insight_rows = [
                r
                for r in insight_rows
                if r["period_start"] == ps and r["period_end"] == pe
            ]

        insights: dict[str, list[dict[str, Any]]] = {}
        totals: dict[str, int] = {}
        for row in insight_rows:
            dim = row["dimension"]
            totals[dim] = totals.get(dim, 0) + int(row["count"] or 0)
            insights.setdefault(dim, []).append(
                {"label": row["label"], "count": int(row["count"] or 0)}
            )
        for dim, items in insights.items():
            total = totals.get(dim, 0) or 1
            for item in items:
                item["pct"] = round(100.0 * item["count"] / total, 1)

        active_days = int(daily.get("active_days") or 0)
        total_events = int(events_summary.get("total_events") or 0)
        if total_events >= 20 and active_days >= 5:
            data_quality = "high"
        elif total_events >= 5 or active_days >= 2:
            data_quality = "medium"
        else:
            data_quality = "low"

        return {
            "email": email,
            "period": {
                "start": period_start.isoformat(),
                "end": period_end.isoformat(),
            },
            "identity": {
                "name": member.get("name"),
                "role": member.get("role"),
            },
            "daily_usage": daily,
            "events_summary": events_summary,
            "models": models,
            "kinds": kinds,
            "insights": insights,
            "data_quality": data_quality,
            "source": "telemetry_only",
        }
    finally:
        conn.close()


def save_context(context: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    email = context["email"].replace("@", "_at_")
    period = context["period"]
    path = out_dir / email / f"context_{period['start']}_{period['end']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(context, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
  parser = argparse.ArgumentParser(description="Fetch user telemetry from MySQL")
  parser.add_argument("--user", help="User email (omit to list users only)")
  parser.add_argument("--days", type=int, default=30)
  parser.add_argument("--start", type=_parse_date)
  parser.add_argument("--end", type=_parse_date)
  parser.add_argument("--list-users", action="store_true")
  args = parser.parse_args()

  load_mysql_settings()
  start, end = resolve_period(args.days, args.start, args.end)

  if args.list_users or not args.user:
    users = list_users_in_period(start, end)
    print(f"Users with events {start}..{end}: {len(users)}")
    for u in users:
      print(f"  {u}")
    if not args.user:
      return 0

  context = fetch_user_context(args.user, start, end)
  path = save_context(context)
  print(f"Saved: {path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

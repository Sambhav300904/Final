"""
Fetch GitHub Copilot usage from MySQL for one user and period.

Requires Copilot tables (cursor_mysql_sync/schema.sql) and sync via copilot_sync.py.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from db import get_connection  # noqa: E402 — cursor_mysql_sync on sys.path


def copilot_tables_exist() -> bool:
    try:
        from copilot_sync import copilot_tables_exist as _exists  # noqa: WPS433

        return _exists()
    except Exception:
        return False


def copilot_data_exists(email: str, period_start: date, period_end: date) -> bool:
    if not copilot_tables_exist():
        return False
    email = email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM fact_copilot_daily_usage
            WHERE LOWER(user_email) = %s
              AND usage_date BETWEEN %s AND %s
            LIMIT 1
            """,
            (email, period_start, period_end),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def fetch_copilot_context(
    email: str, period_start: date, period_end: date
) -> dict[str, Any] | None:
    """Return Copilot context dict, or None if tables missing or no rows in period."""
    if not copilot_tables_exist():
        return None

    email = email.strip().lower()
    conn = get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute(
            """
            SELECT github_login, github_user_id, email, name, last_activity_at,
                   last_activity_editor
            FROM dim_copilot_users
            WHERE LOWER(TRIM(email)) = %s
            LIMIT 1
            """,
            (email,),
        )
        member = cur.fetchone()

        cur.execute(
            """
            SELECT
                COUNT(*) AS days_in_window,
                SUM(user_initiated_interaction_count) AS interactions,
                SUM(code_generation_activity_count) AS code_generations,
                SUM(code_acceptance_activity_count) AS code_acceptances,
                SUM(loc_suggested_to_add_sum) AS loc_suggested,
                SUM(loc_added_sum) AS loc_added,
                SUM(used_chat) AS chat_days,
                SUM(used_agent) AS agent_days
            FROM fact_copilot_daily_usage
            WHERE LOWER(user_email) = %s
              AND usage_date BETWEEN %s AND %s
            """,
            (email, period_start, period_end),
        )
        daily = cur.fetchone() or {}
        days_with_data = int(daily.get("days_in_window") or 0)
        if days_with_data == 0:
            return None

        cur.execute(
            """
            SELECT dimension, label,
                   SUM(interaction_count) AS interactions,
                   SUM(loc_suggested) AS loc_suggested,
                   SUM(loc_added) AS loc_added
            FROM fact_copilot_breakdown
            WHERE LOWER(user_email) = %s
              AND usage_date BETWEEN %s AND %s
            GROUP BY dimension, label
            ORDER BY interactions DESC
            LIMIT 40
            """,
            (email, period_start, period_end),
        )
        breakdown_rows = list(cur.fetchall())

        breakdowns: dict[str, list[dict[str, Any]]] = {}
        for row in breakdown_rows:
            dim = row["dimension"] or "other"
            breakdowns.setdefault(dim, []).append(
                {
                    "label": row["label"],
                    "interactions": int(row.get("interactions") or 0),
                    "loc_suggested": int(row.get("loc_suggested") or 0),
                    "loc_added": int(row.get("loc_added") or 0),
                }
            )

        interactions = int(daily.get("interactions") or 0)
        if interactions >= 50 and days_with_data >= 3:
            data_quality = "high"
        elif interactions >= 10 or days_with_data >= 2:
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
                "github_login": (member or {}).get("github_login"),
                "name": (member or {}).get("name"),
                "email": (member or {}).get("email") or email,
            },
            "daily_usage": daily,
            "breakdowns": breakdowns,
            "data_quality": data_quality,
            "source": "copilot_mysql",
        }
    finally:
        conn.close()

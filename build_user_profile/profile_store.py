"""MySQL roster + proficiency snapshot persistence."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

_BUILD = Path(__file__).resolve().parent
_SYNC = _BUILD.parent / "cursor_mysql_sync"
_SCHEMA = _SYNC / "schema_profile.sql"

import sys

sys.path.insert(0, str(_SYNC))

from db import get_connection  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402


def _split_sql(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for chunk in text.split(";"):
        lines = [
            ln
            for ln in chunk.splitlines()
            if ln.strip() and not ln.strip().startswith("--")
        ]
        stmt = "\n".join(lines).strip()
        if stmt:
            out.append(stmt)
    return out


def profile_tables_exist() -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = DATABASE()
              AND table_name IN (
                'profile_batch_runs',
                'fact_user_proficiency_snapshot'
              )
            """
        )
        return int(cur.fetchone()[0]) == 2
    finally:
        conn.close()


def init_profile_schema() -> None:
    if not _SCHEMA.is_file():
        raise FileNotFoundError(_SCHEMA)
    conn = get_connection()
    try:
        cur = conn.cursor()
        for stmt in _split_sql(_SCHEMA):
            cur.execute(stmt)
        conn.commit()
    finally:
        conn.close()


def count_team_members() -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM dim_cursor_team_members")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def list_team_members(
    *,
    period_start: date | None = None,
    period_end: date | None = None,
    require_events: bool = False,
) -> list[str]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        if require_events and period_start and period_end:
            try:
                from fetch_copilot_data import copilot_tables_exist  # noqa: WPS433

                has_copilot = copilot_tables_exist()
            except Exception:
                has_copilot = False
            if has_copilot:
                cur.execute(
                    """
                    SELECT LOWER(TRIM(m.email))
                    FROM dim_cursor_team_members m
                    WHERE EXISTS (
                        SELECT 1 FROM fact_cursor_usage_events e
                        WHERE LOWER(TRIM(e.user_email)) = LOWER(TRIM(m.email))
                          AND e.event_at >= %s
                          AND e.event_at < %s + INTERVAL 1 DAY
                    )
                    OR EXISTS (
                        SELECT 1 FROM fact_copilot_daily_usage c
                        WHERE LOWER(TRIM(c.user_email)) = LOWER(TRIM(m.email))
                          AND c.usage_date BETWEEN %s AND %s
                    )
                    ORDER BY 1
                    """,
                    (period_start, period_end, period_start, period_end),
                )
            else:
                cur.execute(
                    """
                    SELECT LOWER(TRIM(m.email))
                    FROM dim_cursor_team_members m
                    WHERE EXISTS (
                        SELECT 1 FROM fact_cursor_usage_events e
                        WHERE LOWER(TRIM(e.user_email)) = LOWER(TRIM(m.email))
                          AND e.event_at >= %s
                          AND e.event_at < %s + INTERVAL 1 DAY
                    )
                    ORDER BY 1
                    """,
                    (period_start, period_end),
                )
        else:
            cur.execute(
                """
                SELECT LOWER(TRIM(email))
                FROM dim_cursor_team_members
                WHERE email IS NOT NULL AND TRIM(email) != ''
                ORDER BY 1
                """
            )
        return [row[0] for row in cur.fetchall() if row[0]]
    finally:
        conn.close()


def snapshot_exists(email: str, period_start: date, period_end: date) -> bool:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 1 FROM fact_user_proficiency_snapshot
            WHERE user_email = %s AND period_start = %s AND period_end = %s
              AND pipeline_status = 'success'
            LIMIT 1
            """,
            (email.strip().lower(), period_start, period_end),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def start_batch_run(
    period_start: date,
    period_end: date,
    workers: int,
    config: dict[str, Any] | None = None,
) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO profile_batch_runs (
                period_start, period_end, status, workers, config_json
            ) VALUES (%s, %s, 'running', %s, %s)
            """,
            (
                period_start,
                period_end,
                workers,
                json.dumps(config) if config else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def finish_batch_run(
    run_id: int,
    *,
    status: str,
    users_total: int,
    users_ok: int,
    users_failed: int,
    message: str = "",
) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE profile_batch_runs
            SET finished_at = CURRENT_TIMESTAMP,
                status = %s,
                users_total = %s,
                users_ok = %s,
                users_failed = %s,
                message = %s
            WHERE id = %s
            """,
            (
                status,
                users_total,
                users_ok,
                users_failed,
                message[:4000] if message else None,
                run_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _json_col(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def build_row_from_artifacts(
    *,
    user_email: str,
    period_start: date,
    period_end: date,
    profile_bundle: dict[str, Any] | None,
    scores: dict[str, Any] | None,
    profile_path: Path | None,
    scores_path: Path | None,
    batch_run_id: int | None,
    pipeline_status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    merged = (profile_bundle or {}).get("context") or {}
    if profile_bundle and "telemetry" not in merged and "email" in merged:
        telemetry = merged
        merged = {"telemetry": telemetry, "evidence": None}
    else:
        telemetry = merged.get("telemetry") or merged

    identity = (telemetry or {}).get("identity") or {}
    evidence = merged.get("evidence")
    copilot = merged.get("copilot")
    profile = (profile_bundle or {}).get("profile") or {}
    data_sources = (profile_bundle or {}).get("data_sources")
    if isinstance(data_sources, list):
        data_sources = ",".join(data_sources)

    evidence_count = len((evidence or {}).get("files_processed") or [])
    matched_items = (scores or {}).get("matched_items") or []
    rollups = (scores or {}).get("competency_rollups") or []
    matched_comp_ids = sorted(
        {
            int(r["competency_id"])
            for r in rollups
            if int(r.get("matched_count") or 0) > 0
        }
    )
    matched_item_ids = [int(m["item_id"]) for m in matched_items if m.get("item_id")]
    top_matched = [
        {
            "item_id": m.get("item_id"),
            "competency_id": m.get("competency_id"),
            "competency_name": m.get("competency_name"),
            "similarity": m.get("similarity"),
        }
        for m in matched_items[:15]
    ]

    confidence = profile.get("confidence")
    if isinstance(confidence, dict):
        confidence = json.dumps(confidence, ensure_ascii=False)
    elif confidence is not None:
        confidence = str(confidence)[:16]

    return {
        "user_email": user_email.strip().lower(),
        "period_start": period_start,
        "period_end": period_end,
        "display_name": identity.get("name"),
        "role": identity.get("role"),
        "data_sources": data_sources,
        "profile_summary": profile.get("summary"),
        "profile_confidence": confidence,
        "profile_source": profile.get("source"),
        "telemetry_score": (scores or {}).get("telemetry_score"),
        "matched_count": (scores or {}).get("matched_count"),
        "total_items": (scores or {}).get("total_items") or 43,
        "suggested_level_v1": (scores or {}).get("suggested_level_v1"),
        "match_threshold": (scores or {}).get("match_threshold"),
        "matched_competency_ids": _json_col(matched_comp_ids),
        "matched_item_ids": _json_col(matched_item_ids),
        "top_matched_items": _json_col(top_matched),
        "competency_rollups": _json_col(rollups),
        "evidence_files_count": evidence_count,
        "pipeline_status": pipeline_status,
        "error_message": (error_message or "")[:4000] or None,
        "profile_json_path": str(profile_path) if profile_path else None,
        "scores_json_path": str(scores_path) if scores_path else None,
        "batch_run_id": batch_run_id,
    }


def upsert_snapshot(row: dict[str, Any]) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO fact_user_proficiency_snapshot (
                user_email, period_start, period_end,
                display_name, role, data_sources,
                profile_summary, profile_confidence, profile_source,
                telemetry_score, matched_count, total_items,
                suggested_level_v1, match_threshold,
                matched_competency_ids, matched_item_ids,
                top_matched_items, competency_rollups,
                evidence_files_count, pipeline_status, error_message,
                profile_json_path, scores_json_path, batch_run_id
            ) VALUES (
                %(user_email)s, %(period_start)s, %(period_end)s,
                %(display_name)s, %(role)s, %(data_sources)s,
                %(profile_summary)s, %(profile_confidence)s, %(profile_source)s,
                %(telemetry_score)s, %(matched_count)s, %(total_items)s,
                %(suggested_level_v1)s, %(match_threshold)s,
                %(matched_competency_ids)s, %(matched_item_ids)s,
                %(top_matched_items)s, %(competency_rollups)s,
                %(evidence_files_count)s, %(pipeline_status)s, %(error_message)s,
                %(profile_json_path)s, %(scores_json_path)s, %(batch_run_id)s
            )
            ON DUPLICATE KEY UPDATE
                display_name = VALUES(display_name),
                role = VALUES(role),
                data_sources = VALUES(data_sources),
                profile_summary = VALUES(profile_summary),
                profile_confidence = VALUES(profile_confidence),
                profile_source = VALUES(profile_source),
                telemetry_score = VALUES(telemetry_score),
                matched_count = VALUES(matched_count),
                total_items = VALUES(total_items),
                suggested_level_v1 = VALUES(suggested_level_v1),
                match_threshold = VALUES(match_threshold),
                matched_competency_ids = VALUES(matched_competency_ids),
                matched_item_ids = VALUES(matched_item_ids),
                top_matched_items = VALUES(top_matched_items),
                competency_rollups = VALUES(competency_rollups),
                evidence_files_count = VALUES(evidence_files_count),
                pipeline_status = VALUES(pipeline_status),
                error_message = VALUES(error_message),
                profile_json_path = VALUES(profile_json_path),
                scores_json_path = VALUES(scores_json_path),
                batch_run_id = VALUES(batch_run_id),
                updated_at = CURRENT_TIMESTAMP
            """,
            row,
        )
        conn.commit()
    finally:
        conn.close()


def artifact_paths(email: str, period_start: date, period_end: date) -> tuple[Path, Path]:
    safe = email.replace("@", "_at_")
    stem = f"{period_start.isoformat()}_{period_end.isoformat()}"
    folder = OUTPUT_DIR / safe
    return (
        folder / f"profile_{stem}.json",
        folder / f"scores_{stem}.json",
    )


def load_artifacts(email: str, period_start: date, period_end: date) -> tuple[dict | None, dict | None, Path | None, Path | None]:
    profile_path, scores_path = artifact_paths(email, period_start, period_end)
    profile_bundle = None
    scores = None
    if profile_path.is_file():
        profile_bundle = json.loads(profile_path.read_text(encoding="utf-8"))
    if scores_path.is_file():
        scores = json.loads(scores_path.read_text(encoding="utf-8"))
    return profile_bundle, scores, profile_path if profile_path.is_file() else None, scores_path if scores_path.is_file() else None


def persist_user_snapshot(
    email: str,
    period_start: date,
    period_end: date,
    *,
    batch_run_id: int | None = None,
    pipeline_status: str = "success",
    error_message: str | None = None,
) -> None:
    """Upsert one user row into fact_user_proficiency_snapshot from output JSON files."""
    profile_bundle, scores, profile_path, scores_path = load_artifacts(
        email, period_start, period_end
    )
    row = build_row_from_artifacts(
        user_email=email,
        period_start=period_start,
        period_end=period_end,
        profile_bundle=profile_bundle,
        scores=scores,
        profile_path=profile_path,
        scores_path=scores_path,
        batch_run_id=batch_run_id,
        pipeline_status=pipeline_status,
        error_message=error_message,
    )
    upsert_snapshot(row)

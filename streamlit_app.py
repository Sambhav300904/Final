"""
Streamlit UI for the AI Adoption profile pipeline.

Run from repo root:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
BUILD = REPO_ROOT / "build_user_profile"
EVIDENCE = REPO_ROOT / "evidence_evaluator"
BATCH_SCRIPT = REPO_ROOT / "batch_pipeline.py"

sys.path.insert(0, str(BUILD))

from config import load_mysql_settings  # noqa: E402
from fetch_user_data import resolve_period  # noqa: E402
from pipeline_runner import (  # noqa: E402
    build_period_args,
    run_full_pipeline_for_user,
)
from profile_store import (  # noqa: E402
    count_team_members,
    init_profile_schema,
    list_team_members,
    persist_user_snapshot,
    profile_tables_exist,
    snapshot_exists,
)

OUTPUT_DIR = BUILD / "output"
EVIDENCE_DIR = REPO_ROOT / "evidence_evaluator" / "evidences"
EVIDENCE_OUTPUT = REPO_ROOT / "evidence_evaluator" / "output"
SYNC_DIR = REPO_ROOT / "cursor_mysql_sync"

_audit_user_fn: Any = None


def _get_audit_user():
    """Load audit_user from auditor_agent.py without shadowing build_user_profile.config."""
    global _audit_user_fn
    if _audit_user_fn is not None:
        return _audit_user_fn
    path = REPO_ROOT / "auditor_agent.py"
    spec = importlib.util.spec_from_file_location("auditor_agent", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load auditor_agent from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _audit_user_fn = mod.audit_user
    return _audit_user_fn


def _email_dir(email: str) -> str:
    return email.strip().lower().replace("@", "_at_")


def _dir_to_email(dirname: str) -> str:
    return dirname.replace("_at_", "@")


def _period_args(
    user: str,
    days: int,
    start: date | None,
    end: date | None,
) -> list[str]:
    args = ["--user", user.strip(), "--days", str(days)]
    if start:
        args.extend(["--start", start.isoformat()])
    if end:
        args.extend(["--end", end.isoformat()])
    return args


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log = f"> {' '.join(cmd)}\n\n"
    if proc.stdout:
        log += proc.stdout
    if proc.stderr:
        if proc.stdout:
            log += "\n"
        log += proc.stderr
    return proc.returncode, log.rstrip()


def _run_streaming(
    cmd: list[str],
    cwd: Path,
    on_update: Callable[[str], None],
) -> tuple[int, str]:
    """Run a command and call on_update(log_text) as output arrives."""
    header = f"> {' '.join(cmd)}\n\n"
    chunks: list[str] = [header]
    on_update(header)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        chunks.append(line)
        on_update("".join(chunks))
    code = proc.wait()
    return code, "".join(chunks).rstrip()


def _build_batch_cmd(
    *,
    days: int,
    start: date | None,
    end: date | None,
    workers: int,
    require_events: bool,
    max_users: int,
    skip_existing: bool,
    no_evidence: bool,
    no_copilot: bool,
    no_llm: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(BATCH_SCRIPT),
        "--days",
        str(days),
        "--workers",
        str(workers),
    ]
    if start:
        cmd.extend(["--start", start.isoformat()])
    if end:
        cmd.extend(["--end", end.isoformat()])
    if require_events:
        cmd.append("--require-events")
    if max_users > 0:
        cmd.extend(["--max-users", str(int(max_users))])
    if skip_existing:
        cmd.append("--skip-existing")
    if no_evidence:
        cmd.append("--no-evidence")
    if no_copilot:
        cmd.append("--no-copilot")
    if no_llm:
        cmd.append("--no-llm")
    return cmd


def run_build(script: str, extra: list[str]) -> tuple[int, str]:
    path = BUILD / script
    return _run([sys.executable, str(path), *extra], BUILD)


def run_evidence(extra: list[str]) -> tuple[int, str]:
    path = EVIDENCE / "summarize_evidence.py"
    return _run([sys.executable, str(path), *extra], EVIDENCE)


def run_sync(days: int, init_schema: bool) -> tuple[int, str]:
    path = SYNC_DIR / "main.py"
    extra: list[str] = ["--days", str(days)]
    if init_schema:
        extra.append("--init-schema")
    return _run([sys.executable, str(path), *extra], SYNC_DIR)


def list_known_users() -> list[str]:
    if not OUTPUT_DIR.is_dir():
        return []
    users = sorted(_dir_to_email(p.name) for p in OUTPUT_DIR.iterdir() if p.is_dir())
    return users


def list_artifacts(email: str, prefix: str) -> list[Path]:
    folder = OUTPUT_DIR / _email_dir(email)
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"{prefix}_*.json"), reverse=True)


def latest_artifact(email: str, prefix: str) -> Path | None:
    files = list_artifacts(email, prefix)
    return files[0] if files else None


def load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def show_log(log: str) -> None:
    with st.expander("Command log", expanded=False):
        st.code(log or "(no output)", language=None)


def render_scores(scores_path: Path, *, key_prefix: str = "scores") -> None:
    data = load_json(scores_path)
    if not isinstance(data, dict):
        st.warning(f"Could not read scores: {scores_path.name}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Telemetry score", f"{data.get('telemetry_score', '—')}")
    c2.metric("Matched items", f"{data.get('matched_count', '—')} / {data.get('total_items', 43)}")
    c3.metric("Suggested level", data.get("suggested_level_v1", "—"))
    c4.metric("Threshold", data.get("match_threshold", "—"))

    matched = data.get("matched_items") or []
    if matched:
        st.subheader("Top matched competencies")
        rows = [
            {
                "Item": m.get("item_id"),
                "Competency": m.get("competency_name"),
                "Similarity": round(float(m.get("similarity", 0)), 4),
            }
            for m in matched[:20]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.download_button(
        "Download scores JSON",
        data=scores_path.read_text(encoding="utf-8"),
        file_name=scores_path.name,
        mime="application/json",
        key=f"{key_prefix}_dl_{scores_path.name}",
    )


_AUDIT_STATUS_LABEL = {
    "ok": ("OK", "normal"),
    "review": ("Review", "off"),
    "high_risk": ("High risk", "inverse"),
    "no_data": ("No data", "secondary"),
    "error": ("Error", "inverse"),
}


def render_audit_result(audit: dict, *, key_prefix: str = "audit") -> None:
    """Show 7-day (or selected period) usage-pattern audit from auditor_agent."""
    status = audit.get("audit_status", "—")
    label, _badge = _AUDIT_STATUS_LABEL.get(status, (status, "secondary"))
    m = audit.get("metrics") or {}
    flags = audit.get("flags") or []
    daily = audit.get("daily") or []

    st.subheader("Usage audit (Auditor)")
    st.caption(
        f"Period {audit.get('period_start')} → {audit.get('period_end')} — "
        "day-wise pattern from `fact_cursor_daily_usage` (no long baseline)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audit status", label)
    c2.metric("Risk score", f"{audit.get('risk_score', 0):.2f}")
    c3.metric("Active days", m.get("active_days", "—"))
    c4.metric("Top day share", f"{int(round((m.get('top1_share') or 0) * 100))}%")

    if audit.get("error"):
        st.error(str(audit["error"]))
        return

    if status == "no_data":
        st.info("No meaningful daily usage in this window.")
        return

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Total activity", m.get("total_activity", "—"))
    c6.metric("Top 2 days share", f"{int(round((m.get('top2_share') or 0) * 100))}%")
    c7.metric("Last 2 days share", f"{int(round((m.get('last2_share') or 0) * 100))}%")
    c8.metric("Concentration", m.get("concentration_gini", "—"))

    if m.get("max_day"):
        st.markdown(
            f"**Peak day:** `{m.get('max_day')}` "
            f"({m.get('max_day_activity', '—')} activity units, "
            f"median day = {m.get('median_day', '—')})"
        )
    if audit.get("spike_days"):
        st.markdown(f"**Spike day(s):** {', '.join(audit['spike_days'])}")

    if flags:
        st.markdown("**Flags**")
        for f in flags:
            sev = f.get("severity", "")
            if sev == "high":
                st.error(f"**{f.get('code')}** — {f.get('detail')}")
            elif sev == "medium":
                st.warning(f"**{f.get('code')}** — {f.get('detail')}")
            else:
                st.info(f"**{f.get('code')}** — {f.get('detail')}")
    else:
        st.success("No usage-pattern flags for this period.")

    if daily:
        chart_rows = [
            {
                "date": d["date"],
                "activity": d.get("activity", 0),
                "lines_added": d.get("total_lines_added", 0),
                "applies": d.get("total_applies", 0),
            }
            for d in daily
        ]
        st.markdown("**Daily activity**")
        st.bar_chart(
            chart_rows,
            x="date",
            y="activity",
            use_container_width=True,
        )
        with st.expander("Daily breakdown table", expanded=False):
            st.dataframe(chart_rows, use_container_width=True, hide_index=True)

    st.download_button(
        "Download audit JSON",
        data=json.dumps(audit, indent=2, default=str),
        file_name=f"audit_{audit.get('user_email', 'user').replace('@', '_at_')}_"
        f"{audit.get('period_start')}_{audit.get('period_end')}.json",
        mime="application/json",
        key=f"{key_prefix}_dl_audit",
    )


def run_usage_audit(
    user: str, period_start: date, period_end: date
) -> dict | None:
    """Run auditor_agent for one user; returns result dict or None on failure."""
    try:
        load_mysql_settings()
        return _get_audit_user()(user.strip().lower(), period_start, period_end)
    except Exception as exc:
        return {
            "user_email": user,
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


def render_profile(profile_path: Path, *, key_prefix: str = "profile") -> None:
    data = load_json(profile_path)
    if not isinstance(data, dict):
        st.warning(f"Could not read profile: {profile_path.name}")
        return

    profile = data.get("profile") or {}
    sources = data.get("data_sources") or (data.get("context") or {}).get("data_sources")
    if sources:
        st.caption(f"Sources: {', '.join(sources)}")

    summary = profile.get("summary")
    narrative = profile.get("profile_narrative")

    if summary:
        st.subheader("Summary")
        st.write(summary)
    if narrative:
        st.subheader("Profile narrative")
        st.write(narrative)

    period = data.get("period") or {}
    if period:
        st.caption(f"Period: {period.get('start')} → {period.get('end')}")

    st.download_button(
        "Download profile JSON",
        data=profile_path.read_text(encoding="utf-8"),
        file_name=profile_path.name,
        mime="application/json",
        key=f"{key_prefix}_dl_{profile_path.name}",
    )


def run_pipeline_step(
    label: str,
    script: str,
    extra: list[str],
    progress: float,
    progress_bar,
    status,
) -> bool:
    status.update(label=f"{label}…")
    progress_bar.progress(progress)
    code, log = run_build(script, extra)
    show_log(log)
    if code != 0:
        st.error(f"{label} failed (exit {code}).")
        return False
    st.success(f"{label} completed.")
    return True


def page_sidebar() -> dict:
    st.sidebar.header("Settings")
    known = list_known_users()
    default_user = known[0] if known else "engineer@company.com"
    user = st.sidebar.text_input("User email", value=default_user).strip()

    period_mode = st.sidebar.radio("Period", ["Last N days", "Custom range"], horizontal=True)
    days = 30
    start: date | None = None
    end: date | None = None
    custom_start: date | None = None
    custom_end: date | None = None

    if period_mode == "Last N days":
        days = st.sidebar.slider("Days", min_value=1, max_value=90, value=30)
    else:
        today = date.today()
        col1, col2 = st.sidebar.columns(2)
        custom_start = col1.date_input("Start", value=today - timedelta(days=30))
        custom_end = col2.date_input("End", value=today - timedelta(days=1))
        if custom_start > custom_end:
            st.sidebar.error("Start must be on or before end.")
        start, end = custom_start, custom_end

    no_llm = st.sidebar.checkbox("Template profile only (no LLM)", value=False)

    return {
        "user": user,
        "days": days,
        "start": start,
        "end": end,
        "no_llm": no_llm,
        "period_args": _period_args(user, days, start, end) if user else [],
    }


def main() -> None:
    st.set_page_config(page_title="AI Adoption", page_icon="📊", layout="wide")
    st.title("AI Adoption — Profile Pipeline")
    st.caption(
        "Cursor + evidence + Copilot (MySQL) → profile → embed → scores; "
        "Auditor checks day-wise usage patterns in the selected period."
    )

    cfg = page_sidebar()
    user = cfg["user"]
    period_args = cfg["period_args"]

    if not user:
        st.warning("Enter a user email in the sidebar.")
        return

    tab_run, tab_batch, tab_steps, tab_results, tab_audit, tab_evidence, tab_admin = st.tabs(
        [
            "Run pipeline",
            "Batch (all users)",
            "Step by step",
            "Results",
            "Usage audit",
            "Evidence",
            "Admin",
        ]
    )

    with tab_run:
        st.subheader("Full pipeline")
        st.write(
            "Runs: merge (Cursor MySQL + evidence folder + Copilot MySQL when present) "
            "→ profile → embed → score. On success, saves to `fact_user_proficiency_snapshot`."
        )
        ev_folder = EVIDENCE_DIR / user
        if ev_folder.is_dir() and any(ev_folder.iterdir()):
            names = [f.name for f in ev_folder.iterdir() if f.is_file()]
            st.info(f"Evidence folder: {len(names)} file(s) — {', '.join(names[:5])}")
        else:
            st.caption(
                f"No evidence at `{ev_folder}` — pipeline uses telemetry only. "
                "Add PDFs/images there to include evidence."
            )

        period_start, period_end = resolve_period(
            cfg["days"], cfg["start"], cfg["end"]
        )

        run_col, audit_col = st.columns(2)
        run_pipeline_btn = run_col.button(
            "Run full pipeline",
            type="primary",
            use_container_width=True,
        )
        run_audit_btn = audit_col.button(
            "Run usage audit only",
            use_container_width=True,
            help="7-day-style pattern check from MySQL daily usage (uses sidebar period).",
        )

        if run_audit_btn:
            with st.spinner("Running usage audit…"):
                audit = run_usage_audit(user, period_start, period_end)
            if audit:
                st.session_state["last_audit"] = audit
                render_audit_result(audit, key_prefix="run_audit_only")

        if run_pipeline_btn:
            progress_bar = st.progress(0.0)
            status = st.status("Starting…", expanded=True)
            status.update(label="Running pipeline…")
            progress_bar.progress(0.25)
            pargs = build_period_args(
                user,
                cfg["days"],
                period_start.isoformat(),
                period_end.isoformat(),
            )
            code, log = run_full_pipeline_for_user(
                user,
                pargs,
                no_llm=cfg["no_llm"],
                start_iso=period_start.isoformat(),
                end_iso=period_end.isoformat(),
            )
            show_log(log)
            ok = code == 0
            progress_bar.progress(1.0 if ok else 0.5)
            if ok:
                status.update(label="Pipeline finished", state="complete")
                if profile_tables_exist():
                    try:
                        persist_user_snapshot(user, period_start, period_end)
                        st.success(
                            f"Saved to MySQL: `fact_user_proficiency_snapshot` "
                            f"({period_start} → {period_end})"
                        )
                    except Exception as exc:
                        st.warning(f"Pipeline OK but MySQL save failed: {exc}")
                else:
                    st.caption(
                        "MySQL snapshot table not found — run "
                        "`python batch_pipeline.py --init-profile-schema` or use "
                        "Batch tab → Apply profile MySQL schema."
                    )
                scores = latest_artifact(user, "scores")
                profile = latest_artifact(user, "profile")
                if scores:
                    st.divider()
                    render_scores(scores, key_prefix="pipeline_scores")
                if profile:
                    st.divider()
                    render_profile(profile, key_prefix="pipeline_profile")
                st.divider()
                with st.spinner("Running usage audit…"):
                    audit = run_usage_audit(user, period_start, period_end)
                if audit:
                    st.session_state["last_audit"] = audit
                    render_audit_result(audit, key_prefix="pipeline_audit")
            else:
                status.update(label="Pipeline failed", state="error")

    with tab_batch:
        st.subheader("Batch pipeline — all team members")
        st.caption(
            "Loads emails from `dim_cursor_team_members`, runs `batch_pipeline.py` "
            "as a subprocess (same as CLI), saves summaries and scores to MySQL."
        )
        try:
            load_mysql_settings()
            roster_count = count_team_members()
            st.metric("Team members in roster", roster_count)
        except Exception as exc:
            st.error(f"MySQL: {exc}")
            roster_count = 0

        b1, b2, b3 = st.columns(3)
        workers = b1.slider("Parallel workers", 1, 8, 4)
        require_events = b2.checkbox(
            "Only users with events in period",
            value=True,
            help="Recommended — skips inactive roster entries",
        )
        max_users = b3.number_input(
            "Max users (0 = all)", min_value=0, value=0, step=1
        )
        skip_existing = st.checkbox("Skip users already in snapshot table")
        batch_no_evidence = st.checkbox("Skip evidence folders")
        batch_no_copilot = st.checkbox("Skip Copilot MySQL data")
        init_schema = st.checkbox("Apply profile MySQL schema first")

        period_start, period_end = resolve_period(
            cfg["days"], cfg["start"], cfg["end"]
        )
        if require_events:
            preview = list_team_members(
                period_start=period_start,
                period_end=period_end,
                require_events=True,
            )
        else:
            preview = list_team_members()
        if skip_existing:
            preview = [
                u
                for u in preview
                if not snapshot_exists(u, period_start, period_end)
            ]
        if max_users > 0:
            preview = preview[: int(max_users)]
        st.write(f"**Would process:** {len(preview)} users for {period_start} → {period_end}")

        if st.button("Run batch (parallel)", type="primary", use_container_width=True):
            if not profile_tables_exist() and not init_schema:
                st.error("Run with 'Apply profile MySQL schema' or init via CLI first.")
            else:
                if init_schema:
                    init_profile_schema()
                    st.success("Profile schema applied.")
                if not profile_tables_exist():
                    st.error("Profile tables still missing.")
                elif not preview:
                    st.warning("No users to process for this period / filters.")
                else:
                    batch_cmd = _build_batch_cmd(
                        days=cfg["days"],
                        start=cfg["start"],
                        end=cfg["end"],
                        workers=workers,
                        require_events=require_events,
                        max_users=int(max_users),
                        skip_existing=skip_existing,
                        no_evidence=batch_no_evidence,
                        no_copilot=batch_no_copilot,
                        no_llm=cfg["no_llm"],
                    )
                    log_box = st.empty()
                    with st.spinner("Running batch_pipeline.py…"):
                        code, log = _run_streaming(
                            batch_cmd,
                            REPO_ROOT,
                            on_update=lambda text: log_box.code(text),
                        )
                    show_log(log)
                    if code == 0:
                        st.success("Batch finished successfully.")
                    else:
                        st.error(f"Batch finished with errors (exit {code}).")

        st.markdown("**CLI (recommended for 800+ users):**")
        st.code(
            f'python batch_pipeline.py --days {cfg["days"]} --workers 4 --require-events',
            language="powershell",
        )

    with tab_steps:
        st.subheader("Individual steps")
        cols = st.columns(2)
        step_defs = [
            ("Merge context", "build_merged_context.py", period_args),
            ("Profile", "create_profile.py", period_args + ["--use-saved-context"] + (["--no-llm"] if cfg["no_llm"] else [])),
            ("Embed", "embed_profile.py", ["--user", user]),
            ("Score", "compare/compare_competencies.py", ["--user", user]),
        ]
        for idx, (label, script, extra) in enumerate(step_defs):
            col = cols[idx % 2]
            with col:
                if st.button(label, key=f"step_{label}", use_container_width=True):
                    with st.spinner(f"Running {label}…"):
                        code, log = run_build(script, extra)
                    show_log(log)
                    if code == 0:
                        st.success(f"{label} done.")
                    else:
                        st.error(f"{label} failed (exit {code}).")

    with tab_audit:
        st.subheader("Usage audit (Auditor agent)")
        st.write(
            "Analyzes **day-wise** Cursor usage from MySQL for the sidebar period: "
            "which days drove the week, lumpy vs spread usage, end-of-period loading, "
            "and spike days. **Tip:** set **Days = 7** in the sidebar for the intended window."
        )
        period_start, period_end = resolve_period(
            cfg["days"], cfg["start"], cfg["end"]
        )
        st.caption(f"Current window: **{period_start}** → **{period_end}** · user **{user}**")
        if st.button("Run audit for this user & period", type="primary", use_container_width=True):
            with st.spinner("Auditing daily usage…"):
                audit = run_usage_audit(user, period_start, period_end)
            if audit:
                st.session_state["last_audit"] = audit
                render_audit_result(audit, key_prefix="tab_audit")
        elif st.session_state.get("last_audit"):
            cached = st.session_state["last_audit"]
            if (
                cached.get("user_email", "").lower() == user.lower()
                and cached.get("period_start") == period_start.isoformat()
                and cached.get("period_end") == period_end.isoformat()
            ):
                st.caption("Showing last audit for this user and period.")
                render_audit_result(cached, key_prefix="tab_audit_cached")
            else:
                st.info("Click **Run audit** to analyze this user and period.")

    with tab_results:
        st.subheader("Saved outputs")
        artifacts = {
            "Context": list_artifacts(user, "context"),
            "Profile": list_artifacts(user, "profile"),
            "Embedding": list_artifacts(user, "embedding"),
            "Scores": list_artifacts(user, "scores"),
        }
        if not any(artifacts.values()):
            st.info(f"No outputs yet for `{user}` under `build_user_profile/output/`.")
        else:
            for kind, paths in artifacts.items():
                if not paths:
                    continue
                labels = [p.name for p in paths]
                choice = st.selectbox(kind, labels, key=f"pick_{kind}")
                path = OUTPUT_DIR / _email_dir(user) / choice
                if kind == "Scores":
                    render_scores(path, key_prefix=f"results_scores_{choice}")
                elif kind == "Profile":
                    render_profile(path, key_prefix=f"results_profile_{choice}")
                else:
                    data = load_json(path)
                    st.json(data)
                st.divider()

    with tab_evidence:
        st.subheader("Evidence summarizer")
        ev_folder = EVIDENCE_DIR / user
        st.markdown(
            f"Upload files to **`{ev_folder}`** (PDF, DOCX, TXT, CSV, images), "
            "then run summarize below."
        )
        st.code(str(ev_folder), language=None)
        if ev_folder.is_dir():
            files = [f.name for f in ev_folder.iterdir() if f.is_file()]
            if files:
                st.write("Files in folder:", ", ".join(files))
            else:
                st.caption("Folder exists but has no files yet.")
        else:
            st.caption("Folder does not exist yet — create it and add files.")

        if st.button("Summarize evidence", use_container_width=True):
            with st.spinner("Summarizing evidence…"):
                code, log = run_evidence(period_args)
            show_log(log)
            if code == 0:
                st.success("Evidence summary written.")
                out_dir = EVIDENCE_OUTPUT / _email_dir(user)
                if out_dir.is_dir():
                    for p in sorted(out_dir.glob("evidence_*"), reverse=True)[:3]:
                        st.write(p.name)
                        if p.suffix == ".txt":
                            st.text(p.read_text(encoding="utf-8")[:8000])
                        elif p.suffix == ".json":
                            st.json(load_json(p))
            else:
                st.error(f"Evidence step failed (exit {code}).")

    with tab_admin:
        st.subheader("One-time: ingest competencies")
        c1, c2 = st.columns(2)
        dry_run = c1.checkbox("Dry run", key="ingest_dry")
        recreate = c2.checkbox("Recreate index", key="ingest_recreate")
        if st.button("Ingest 43 items → Pinecone"):
            extra: list[str] = []
            if dry_run:
                extra.append("--dry-run")
            if recreate:
                extra.append("--recreate")
            with st.spinner("Ingesting…"):
                code, log = run_build("competencies/ingest_competencies.py", extra)
            show_log(log)
            if code == 0:
                st.success("Ingest completed.")
            else:
                st.error(f"Ingest failed (exit {code}).")

        st.divider()
        st.subheader("Sync Cursor → MySQL")
        sync_days = st.number_input("Sync days", min_value=1, max_value=90, value=7)
        init_schema = st.checkbox("Apply schema first (--init-schema)")
        if st.button("Run Cursor sync"):
            with st.spinner("Syncing…"):
                code, log = run_sync(int(sync_days), init_schema)
            show_log(log)
            if code == 0:
                st.success("Sync completed.")
            else:
                st.error(f"Sync failed (exit {code}).")


if __name__ == "__main__":
    main()

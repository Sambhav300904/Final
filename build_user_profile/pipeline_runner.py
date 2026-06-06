"""Run the same 4-step pipeline as Streamlit (subprocess per step)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS: list[tuple[str, str]] = [
    ("merge", "build_merged_context.py"),
    ("profile", "create_profile.py"),
    ("embed", "embed_profile.py"),
    ("score", "compare/compare_competencies.py"),
]


def _run_script(
    python_exe: str,
    script: str,
    extra: list[str],
) -> tuple[int, str]:
    path = BUILD_DIR / script
    cmd = [python_exe, str(path), *extra]
    proc = subprocess.run(
        cmd,
        cwd=str(BUILD_DIR),
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


def build_period_args(
    email: str,
    days: int,
    start_iso: str | None,
    end_iso: str | None,
    *,
    no_evidence: bool = False,
    no_copilot: bool = False,
) -> list[str]:
    args = ["--user", email.strip().lower(), "--days", str(days)]
    if start_iso:
        args.extend(["--start", start_iso])
    if end_iso:
        args.extend(["--end", end_iso])
    if no_evidence:
        args.append("--no-evidence")
    if no_copilot:
        args.append("--no-copilot")
    return args


def _artifact_paths(
    email: str, start_iso: str | None, end_iso: str | None
) -> tuple[Path | None, Path | None]:
    if not start_iso or not end_iso:
        return None, None
    safe = email.strip().lower().replace("@", "_at_")
    folder = BUILD_DIR / "output" / safe
    return (
        folder / f"profile_{start_iso}_{end_iso}.json",
        folder / f"embedding_{start_iso}_{end_iso}.json",
    )


def _step_args(
    label: str,
    email: str,
    period_args: list[str],
    *,
    no_llm: bool,
    start_iso: str | None,
    end_iso: str | None,
) -> list[str]:
    if label == "merge":
        return list(period_args)
    if label == "profile":
        args = list(period_args) + ["--use-saved-context"]
        if no_llm:
            args.append("--no-llm")
        return args
    profile_path, embedding_path = _artifact_paths(email, start_iso, end_iso)
    if label == "embed":
        if profile_path and profile_path.is_file():
            return ["--input", str(profile_path)]
        return ["--user", email.strip().lower()]
    if label == "score":
        if embedding_path and embedding_path.is_file():
            return ["--input", str(embedding_path)]
        return ["--user", email.strip().lower()]
    return []


def run_full_pipeline_for_user(
    email: str,
    period_args: list[str],
    *,
    no_llm: bool = False,
    python_exe: str | None = None,
    start_iso: str | None = None,
    end_iso: str | None = None,
) -> tuple[int, str]:
    """Returns (exit_code, combined_log)."""
    py = python_exe or sys.executable
    logs: list[str] = []
    for label, script in PIPELINE_STEPS:
        extra = _step_args(
            label,
            email,
            period_args,
            no_llm=no_llm,
            start_iso=start_iso,
            end_iso=end_iso,
        )
        code, log = _run_script(py, script, extra)
        logs.append(f"=== {label} ===\n{log}")
        if code != 0:
            return code, "\n\n".join(logs)

    return 0, "\n\n".join(logs)

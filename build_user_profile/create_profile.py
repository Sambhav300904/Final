"""
Build an engineer profile from MySQL telemetry and optional uploaded evidence.

Usage:
    python create_profile.py --user engineer@company.com --days 30
    python create_profile.py --input output/user/context_....json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

_BUILD = Path(__file__).resolve().parent
sys.path.insert(0, str(_BUILD.parent / "cursor_mysql_sync"))
sys.path.insert(0, str(_BUILD))
sys.path.insert(0, str(_BUILD.parent / "evidence_evaluator"))

from competencies_data import COMPETENCIES  # noqa: E402
from build_merged_context import (  # noqa: E402
    build_merged_context,
    load_saved_context,
    normalize_merged_context,
    save_merged_context,
)
from config import OUTPUT_DIR, load_settings, validate_llm_settings  # noqa: E402
from fetch_user_data import resolve_period  # noqa: E402

try:
    from summarize_evidence import save_bundle as save_evidence_bundle  # noqa: E402
except ImportError:
    save_evidence_bundle = None  # type: ignore[misc, assignment]

_FRAMEWORK_COMPETENCY_LIST = "\n".join(
    f"  {cid}. {name}" for cid, name in COMPETENCIES.items()
)

MULTIMODAL_SYSTEM_PROMPT = f"""You are an analyst for the R Systems AI Coding Proficiency Framework.

You receive:
1) A JSON bundle (telemetry from Cursor MySQL, optional Copilot MySQL, optional evidence summaries).
2) Zero or more evidence IMAGE attachments (screenshots the engineer uploaded).

The 8 competency area names (use these exactly; do NOT use Cursor insight labels such as Configuration or Code Explanation):
{_FRAMEWORK_COMPETENCY_LIST}

Rules:
- Use ONLY facts from the JSON and attached images. Do not invent evidence.
- When images are attached, treat them as primary proof for Cursor/Copilot/AI-tool usage; cite filenames.
- When only telemetry JSON is provided (no images), phrase competency sections as observed signals from usage data.
- Combine Copilot MySQL data with Cursor telemetry when both exist.
- Write in third person about the engineer.
- Organize profile_narrative sections as "Competency: <exact name from list above>".
- Note gaps in evidence_gaps when proof is missing.
- Output valid JSON with keys: summary, profile_narrative, competency_signals, evidence_gaps, confidence.
- profile_narrative MUST be a single JSON string (plain text), NOT a nested object or map.
- profile_narrative: 400-700 words, embedding-friendly.
- competency_signals: list of {{competency_id, competency_name, observed_signals[], confidence}}; competency_id is 1-8.
- confidence: low | medium | high.
- Respond with raw JSON only. No markdown code fences.
"""

SYSTEM_PROMPT = f"""You are an analyst for the R Systems AI Coding Proficiency Framework.
You receive a JSON bundle with:
- telemetry: Cursor IDE usage (events, daily metrics, conversation insights)
- evidence (optional): summaries of uploaded files (PDFs, screenshots, PR exports, certs, etc.)
- copilot (optional): GitHub Copilot org usage from MySQL (interactions, code acceptance, LOC, breakdowns by IDE/feature/language)

The 8 competency area names (use these exactly; do NOT use Cursor insight labels such as Configuration or Code Explanation):
{_FRAMEWORK_COMPETENCY_LIST}

Rules:
- Use ONLY facts from the JSON input. Do not invent evidence or claim checkboxes are completed.
- When evidence summaries are present, treat them as verifiable artefacts; cite them for checklist-style signals.
- When copilot data is present, combine it with Cursor telemetry; note where Copilot vs Cursor signals differ.
- When only telemetry is present, phrase competency sections as "observed signals" or "usage patterns suggest".
- Write in third person about the engineer.
- Organize profile_narrative sections as "Competency: <exact name from list above>".
- If evidence is missing for a competency, note gaps in evidence_gaps rather than inferring proof.
- Include a short Limitations section when telemetry or evidence is incomplete.
- Output valid JSON with keys: summary, profile_narrative, competency_signals, evidence_gaps, confidence.
- profile_narrative MUST be a single JSON string (plain text), NOT a nested object or map.
- profile_narrative: 400-700 words, embedding-friendly, one section per competency area where data supports it.
- competency_signals: list of {{competency_id, competency_name, observed_signals[], confidence}}; competency_id is 1-8.
- confidence: low | medium | high based on data_quality, evidence volume, and telemetry volume.
- Respond with raw JSON only. No markdown code fences.
"""

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def normalize_narrative(value: Any) -> str:
    """Coerce LLM profile_narrative (str, dict, list, etc.) to plain text for .txt / embed."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: list[str] = []
        for key, val in value.items():
            header = str(key).strip()
            body = normalize_narrative(val)
            if body and body != header:
                parts.append(f"{header}\n{body}")
            elif header:
                parts.append(header)
        return "\n\n".join(parts)
    if isinstance(value, list):
        return "\n\n".join(
            normalize_narrative(item) for item in value if item is not None
        )
    return str(value).strip()


def _template_profile(merged: dict[str, Any]) -> dict[str, Any]:
    context = merged.get("telemetry") or merged
    evidence = merged.get("evidence")
    copilot = merged.get("copilot")
    email = merged.get("email") or context["email"]
    identity = context.get("identity") or {}
    daily = context.get("daily_usage") or {}
    events = context.get("events_summary") or {}
    insights = context.get("insights") or {}
    quality = context.get("data_quality", "low")

    has_evidence = bool(evidence and evidence.get("summaries"))
    has_copilot = bool(copilot and (copilot.get("daily_usage") or {}).get("days_in_window"))
    parts = ["telemetry"]
    if has_evidence:
        parts.append("evidence")
    if has_copilot:
        parts.append("copilot")
    source_label = " + ".join(parts)

    lines = [
        f"R Systems AI Coding Proficiency — engineer profile ({source_label}, not official sign-off).",
        f"Engineer: {identity.get('name') or email} ({email}).",
        f"Period: {merged['period']['start']} to {merged['period']['end']}.",
        f"Data quality: {quality}.",
        "",
        f"Usage: {events.get('total_events', 0)} events; "
        f"{daily.get('active_days', 0)} active days; "
        f"{events.get('total_tokens', 0)} tokens.",
        "",
    ]
    for dim, items in insights.items():
        lines.append(f"Conversation insights — {dim}:")
        for it in items[:6]:
            lines.append(f"  - {it['label']}: {it.get('pct', 0)}%")
        lines.append("")

    models = context.get("models") or []
    if models:
        lines.append("Top models:")
        for m in models[:5]:
            lines.append(f"  - {m.get('model')}: {m.get('events')} events")
        lines.append("")

    if has_evidence:
        lines.append("Uploaded evidence summaries:")
        for entry in evidence.get("summaries") or []:
            if entry.get("error"):
                lines.append(f"  - {entry['file']}: [error]")
            else:
                preview = (entry.get("summary") or "")[:400]
                lines.append(f"  - {entry['file']}: {preview}")
        lines.append("")

    if has_copilot:
        cd = copilot.get("daily_usage") or {}
        lines.append(
            f"GitHub Copilot: {cd.get('days_in_window', 0)} active days; "
            f"{cd.get('interactions', 0)} interactions; "
            f"{cd.get('code_acceptances', 0)} acceptances; "
            f"{cd.get('loc_added', 0)} LOC added."
        )
        for dim, items in (copilot.get("breakdowns") or {}).items():
            lines.append(f"Copilot breakdown — {dim}:")
            for it in items[:5]:
                lines.append(f"  - {it['label']}: {it.get('interactions', 0)} interactions")
            lines.append("")

    narrative = "\n".join(lines)
    if has_evidence:
        narrative += (
            "\nLimitations: Evidence summaries are LLM-derived from uploads; "
            "official framework sign-off may still require reviewer validation."
        )
    else:
        narrative += (
            "\nLimitations: Official framework items require PR/ticket/document evidence. "
            "This profile is provisional telemetry only."
        )

    gaps = [
        "Items 1-2 require training and safe-usage acknowledgement records.",
        "Merged PR and ticket evidence not available from Cursor telemetry alone.",
    ]
    if not has_evidence:
        gaps.insert(0, "No uploaded evidence files were found for this user.")
    if not has_copilot:
        gaps.append("No GitHub Copilot usage in MySQL for this period.")

    return {
        "summary": f"Profile for {email} ({source_label}, {quality} confidence).",
        "profile_narrative": narrative,
        "competency_signals": [],
        "evidence_gaps": gaps,
        "confidence": quality if quality in ("low", "medium", "high") else "low",
        "source": "template_fallback",
    }


def _parse_llm_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0].strip()
    return json.loads(text)


def _llm_enabled(settings, use_llm: bool) -> bool:
    if not use_llm:
        return False
    provider = settings.llm.provider
    if provider in {"template", "none", "off"}:
        return False
    # All non-template profiles use multimodal Qwen-VL (telemetry-only or with images).
    if provider in {"multimodal", "openai", "lmstudio", "openai_compatible", "vision"}:
        return bool(settings.vision_llm.model) and bool(settings.vision_llm.api_key)
    return False


def _collect_evidence_image_paths(merged: dict[str, Any]) -> list[tuple[str, Path]]:
    """Kept evidence image files to attach to the multimodal profile call."""
    evidence = merged.get("evidence") or {}
    folder = evidence.get("evidence_folder")
    if not folder:
        return []
    base = Path(folder)
    if not base.is_dir():
        return []
    try:
        max_images = int(os.getenv("PROFILE_MAX_EVIDENCE_IMAGES") or "10")
    except ValueError:
        max_images = 10
    paths: list[tuple[str, Path]] = []
    for entry in evidence.get("summaries") or []:
        if entry.get("error"):
            continue
        name = entry.get("file")
        if not name:
            continue
        path = base / name
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            paths.append((name, path))
        if len(paths) >= max_images:
            break
    return paths


def _make_vision_client(settings):
    from huggingface_hub import InferenceClient

    v = settings.vision_llm
    return InferenceClient(token=v.api_key, provider=v.provider, timeout=v.timeout_sec)


def _multimodal_llm_profile(merged: dict[str, Any], settings) -> dict[str, Any]:
    from summarize_evidence import call_with_retries, encode_image  # noqa: WPS433

    client = _make_vision_client(settings)
    v = settings.vision_llm
    sources = merged.get("data_sources") or ["telemetry"]
    images = _collect_evidence_image_paths(merged)

    user_text = (
        f"Build a proficiency-oriented profile from the data below "
        f"(sources: {', '.join(sources)}).\n"
    )
    if images:
        user_text += (
            f"\n{len(images)} evidence image(s) are attached after this text. "
            "Use them together with the JSON summaries.\n"
        )
    else:
        user_text += (
            "\nNo evidence images are attached for this run; use the JSON telemetry "
            "(and any evidence text summaries) only.\n"
        )
    user_text += "\n\nJSON bundle:\n" + json.dumps(merged, indent=2, default=str)

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for name, path in images:
        content.append({"type": "text", "text": f"\n--- Evidence file: {name} ---"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image(str(path))},
            }
        )

    print(
        f"Profile LLM (multimodal): {v.provider}, model={v.model}, "
        f"images={len(images)}",
        flush=True,
    )

    def _call():
        return client.chat.completions.create(
            model=v.model,
            messages=[
                {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=2500,
            temperature=settings.llm.temperature,
        )

    response = call_with_retries(_call, label="Multimodal profile")
    raw = response.choices[0].message.content or "{}"
    telemetry = merged.get("telemetry") or merged
    try:
        data = _parse_llm_json(raw)
    except json.JSONDecodeError:
        data = {
            "summary": "LLM profile (non-JSON response)",
            "profile_narrative": raw,
            "competency_signals": [],
            "evidence_gaps": [],
            "confidence": telemetry.get("data_quality", "low"),
        }
    data["source"] = f"multimodal:{v.provider}:{v.model}"
    if "profile_narrative" not in data:
        data["profile_narrative"] = data.get("summary", "")
    data["profile_narrative"] = normalize_narrative(data["profile_narrative"])
    return data


def _llm_profile(merged: dict[str, Any], settings) -> dict[str, Any]:
    if not _llm_enabled(settings, use_llm=True):
        return _template_profile(merged)
    return _multimodal_llm_profile(merged, settings)


def create_profile_from_context(
    context: dict[str, Any], *, use_llm: bool = True
) -> dict[str, Any]:
    merged = normalize_merged_context(context)
    settings = load_settings()
    if _llm_enabled(settings, use_llm):
        validate_llm_settings(settings)
    profile = (
        _llm_profile(merged, settings)
        if _llm_enabled(settings, use_llm)
        else _template_profile(merged)
    )
    return {
        "email": merged["email"],
        "period": merged["period"],
        "data_sources": merged.get("data_sources", ["telemetry"]),
        "context": merged,
        "profile": profile,
    }


def save_profile(bundle: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR
    email = bundle["email"].replace("@", "_at_")
    period = bundle["period"]
    path = out_dir / email / f"profile_{period['start']}_{period['end']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    narrative = normalize_narrative(bundle["profile"].get("profile_narrative", ""))
    bundle["profile"]["profile_narrative"] = narrative
    path.write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    path.with_suffix(".txt").write_text(narrative, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create engineer profile from MySQL telemetry")
    parser.add_argument("--user")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", type=lambda s: date.fromisoformat(s))
    parser.add_argument("--end", type=lambda s: date.fromisoformat(s))
    parser.add_argument("--input", type=Path, help="Existing context JSON (legacy or merged)")
    parser.add_argument("--no-llm", action="store_true", help="Use template only")
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Skip evidence folder when building context from --user",
    )
    parser.add_argument(
        "--use-saved-context",
        action="store_true",
        help="Load context JSON for this user/period if already on disk",
    )
    parser.add_argument(
        "--repair",
        type=Path,
        help="Re-save an existing profile_*.json (normalize profile_narrative to text)",
    )
    args = parser.parse_args()

    if args.repair:
        bundle = json.loads(args.repair.read_text(encoding="utf-8"))
        path = save_profile(bundle)
        print(f"Repaired: {path}")
        print(f"Narrative: {path.with_suffix('.txt')}")
        return 0

    if args.input:
        merged = normalize_merged_context(
            json.loads(args.input.read_text(encoding="utf-8"))
        )
    else:
        if not args.user:
            print("Provide --user or --input")
            return 1
        start, end = resolve_period(args.days, args.start, args.end)
        merged = None
        if args.use_saved_context:
            merged = load_saved_context(args.user, start, end)
        if merged is None:
            merged = build_merged_context(
                args.user,
                start,
                end,
                include_evidence=not args.no_evidence,
            )
            ctx_path = save_merged_context(merged)
            print(f"Context: {ctx_path} ({', '.join(merged['data_sources'])})")
            if merged.get("evidence") and save_evidence_bundle:
                ev_path = save_evidence_bundle(merged["evidence"])
                print(f"Evidence: {ev_path}")
        else:
            print(f"Loaded saved context ({', '.join(merged['data_sources'])})")

    bundle = create_profile_from_context(merged, use_llm=not args.no_llm)
    path = save_profile(bundle)
    print(f"Saved: {path}")
    print(f"Narrative: {path.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

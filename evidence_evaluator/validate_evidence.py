"""
Evidence relevance validator agent.

Given a per-file evidence summary (produced by ``summarize_evidence.py``), this
agent decides whether the evidence actually documents usage of Cursor, GitHub
Copilot, or other AI coding-assistant / AI-proficiency activity that is in
scope for the AI Adoption pipeline.

Out-of-scope evidence (random screenshots, unrelated billing, marketing decks,
personal documents, etc.) is rejected so it does not pollute the user's
profile or competency scores.

The agent uses a small LLM call (Hugging Face router / Novita by default) and
falls back to a deterministic keyword heuristic when the LLM is unreachable.

Public API
----------
``validate_evidence_item(client, model, file_name, file_type, summary)``
    Classify a single evidence summary.

``validate_raw_extract(client, model, file_name, file_type, raw_text)``
    Pre-summarize gate on locally extracted file text.

``validate_evidence_bundle(bundle, *, client=None, model=None)``
    Post-summarize pass: rejected files are moved from ``files_processed`` to
    ``files_rejected``; summaries excluded from ``combined_evidence_text``.
    Merges with any ``rejected_summaries`` from pre-validation.

``has_relevant_evidence(bundle)``
    True when at least one accepted summary exists for merge/profile.

CLI
---
    python validate_evidence.py --user engineer@company.com \
        --bundle output/<email_safe>/evidence_<start>_<end>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

try:
    from openai import OpenAI as _OpenAIClient
except ImportError:
    _OpenAIClient = None  # type: ignore[assignment]

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
ROOT_ENV = REPO_ROOT / ".env"
SYNC_ENV = REPO_ROOT / "cursor_mysql_sync" / ".env"

if ROOT_ENV.is_file():
    load_dotenv(ROOT_ENV)
if SYNC_ENV.is_file():
    load_dotenv(SYNC_ENV, override=False)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


RETRY_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.WriteTimeout,
)

# STRONG signals — any one of these is enough to accept the evidence as
# clearly about Cursor / Copilot / a named AI coding tool.
STRONG_KEYWORDS: tuple[str, ...] = (
    "cursor ide",
    "cursor editor",
    "cursor agent",
    "cursor chat",
    "cursor tab",
    "cursor composer",
    "cursor rules",
    "cursor settings",
    "cursor dashboard",
    "cursor.com",
    "github copilot",
    "copilot chat",
    "copilot workspace",
    "copilot completions",
    "ai pair programmer",
    "ai pair programming",
    "codeium",
    "windsurf",
    "tabnine",
    "amazon q developer",
    "jetbrains ai",
    "continue.dev",
    "agentic coding",
    "vibe coding",
    "pair programming with ai",
    # Cursor-specific model identifiers (these only appear on the Cursor
    # billing / models dashboards):
    "composer-1",
    "composer-1.5",
    "composer-2",
    "composer-2.5",
    "claude-opus-4-7-thinking",
    "claude-4.6-opus",
    "claude-4.6-sonnet",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5.3-codex-high",
    "your usage",
    "cumulative spend",
    "group by model",
)

# MEDIUM signals — common AI-coding terms. On their own these are NOT enough
# to accept (e.g. an SAP timesheet manual that just happens to say "AI
# proficiency" should be rejected). They need to co-occur with at least one
# other medium / strong signal, AND must not be canceled out by the
# summarizer's boilerplate (see _STRIP_PHRASES below).
MEDIUM_KEYWORDS: tuple[str, ...] = (
    "cursor",  # bare "cursor" — could also mean a UI cursor; medium signal
    "copilot",  # bare "copilot" — could mean other meanings
    "ai coding",
    "ai-assisted",
    "ai assisted",
    "ai adoption",
    "ai agent",
    "agentic",
    "prompt engineering",
    "claude",
    "anthropic",
    "openai",
    "gpt-4",
    "gpt-5",
    "codex",
    "qwen",
    "huggingface",
    "hugging face",
)

# Phrases injected by ``summarize_evidence.py`` into EVERY summary, regardless
# of the actual file content. They must be stripped before keyword matching
# or the heuristic will rubber-stamp anything (e.g. a timesheet manual whose
# summary starts with "R Systems AI Proficiency Evidence Review…").
_STRIP_PHRASES: tuple[str, ...] = (
    "ai proficiency evidence review",
    "r systems ai proficiency",
    "framework checklist evidence",
    "framework checklist",
    "ai proficiency review",
    "proficiency evidence",
    "verifiable engineering evidence",
    "engineering evidence",
    "evidence review",
    "llm-derived",
    "ai proficiency",  # last so it strips after the longer phrases
)

OUT_OF_SCOPE_PATTERNS: tuple[str, ...] = (
    "personal photo",
    "vacation",
    "wedding",
    "grocery receipt",
    "instagram",
    "facebook post",
    "tiktok",
    "telecom bill",
    "phone bill",
    "broadband bill",
    "airfiber",
    "jio",
    "airtel",
    "vodafone",
    "timesheet",
    "payslip",
    "salary slip",
    "hr policy",
    "leave application",
    "expense report",
    "passport",
    "aadhaar",
    "pan card",
    "rent agreement",
    "boarding pass",
    "menu card",
    "movie ticket",
)

# Backwards-compat alias for any external callers that imported the old name.
RELEVANT_KEYWORDS: tuple[str, ...] = STRONG_KEYWORDS + MEDIUM_KEYWORDS


def _compile_keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Build a single case-insensitive regex with word boundaries around each
    keyword. Whitespace inside multi-word keywords is matched flexibly so
    "cursor   ide" and "cursor\nide" both hit.
    """
    parts: list[str] = []
    for kw in keywords:
        escaped = re.escape(kw).replace(r"\ ", r"\s+")
        parts.append(rf"(?<!\w){escaped}(?!\w)")
    return re.compile("|".join(parts), re.IGNORECASE)


# Pre-compile keyword matchers once; word boundaries (`(?<!\w)` / `(?!\w)`)
# prevent false hits like "cursor" matching inside "precursor" or "jio" in
# "version".
_STRONG_RE = _compile_keyword_pattern(STRONG_KEYWORDS)
_MEDIUM_RE = _compile_keyword_pattern(MEDIUM_KEYWORDS)
_OUT_OF_SCOPE_RE = _compile_keyword_pattern(OUT_OF_SCOPE_PATTERNS)


def _find_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return the distinct, lower-cased matches of ``pattern`` in ``text``."""
    seen: list[str] = []
    for m in pattern.finditer(text):
        token = m.group(0).lower()
        token = re.sub(r"\s+", " ", token).strip()
        if token and token not in seen:
            seen.append(token)
    return seen


def _clean_for_matching(file_name: str, summary: str) -> str:
    """Lower-case + strip summarizer boilerplate so the heuristic only sees
    content that actually came from the uploaded file."""
    raw = f"{file_name}\n{summary}".lower()
    cleaned = raw
    for phrase in _STRIP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    return cleaned


VALIDATOR_SYSTEM_PROMPT = """You are a strict evidence-relevance classifier for an
AI Adoption proficiency pipeline at R Systems.

Each input is a SUMMARY of a single artifact a team member uploaded as evidence
of their AI tool usage.

Decide if the artifact actually documents usage / output / configuration /
training related to AI coding assistants and AI-assisted developer workflows.

IN SCOPE (return is_relevant = true):
- Cursor IDE: chats, agent runs, composer, tabs, rules, settings, dashboards,
  billing pages, usage analytics, "your usage", model spend, PRs authored with
  Cursor, screenshots of the Cursor editor.
- GitHub Copilot: completions, chat, workspace, dashboards, usage reports.
- Other AI coding assistants: Codeium, Windsurf, Tabnine, Amazon Q Developer,
  JetBrains AI, Continue.dev.
- LLM tooling used FOR coding/engineering: Claude/Anthropic, OpenAI/GPT/Codex,
  Qwen, Hugging Face, agentic frameworks, prompt engineering for code.
- Training certificates, course completions, internal docs, blog posts about
  AI tools / AI adoption / AI proficiency / agentic engineering.
- PRs, tickets, design docs, or commits that explicitly mention AI-assisted
  authoring or were produced with these tools.

OUT OF SCOPE (return is_relevant = false):
- Personal documents, holiday photos, marketing slides for unrelated products.
- Generic dashboards or invoices that do not mention any AI coding tool, AI
  model, or AI-assisted workflow.
- Random screenshots of unrelated apps (Slack chit-chat, Spotify, games).
- Office paperwork, HR forms, payslips, resumes without AI-tool context.

Be skeptical: if the summary only vaguely says "AI" or "automation" with no
named tool, named model, or coding context, mark it as NOT relevant.

You MUST respond with ONLY a single JSON object on one line, no prose, no
markdown fences. Schema:
{"is_relevant": bool, "confidence": number between 0 and 1,
 "categories": ["cursor"|"copilot"|"other_ai_coding_tool"|"llm_for_coding"|
                "ai_training"|"ai_adoption_doc"|"unrelated"|"unknown", ...],
 "reasoning": "<= 2 short sentences"}
"""

VALIDATOR_RAW_SYSTEM_PROMPT = """You are a strict evidence-relevance classifier for an
AI Adoption proficiency pipeline at R Systems.

Each input is RAW TEXT EXTRACTED from an uploaded file (PDF/DOCX/TXT/CSV preview).
The file has NOT been summarized yet. Judge whether the artifact itself is likely
evidence of Cursor, GitHub Copilot, or other AI coding-assistant usage.

Use the same IN SCOPE / OUT OF SCOPE rules as for summarized evidence. Be skeptical
of generic "AI" mentions, HR paperwork, bills, and personal documents.

You MUST respond with ONLY a single JSON object on one line, no prose, no
markdown fences. Schema:
{"is_relevant": bool, "confidence": number between 0 and 1,
 "categories": ["cursor"|"copilot"|"other_ai_coding_tool"|"llm_for_coding"|
                "ai_training"|"ai_adoption_doc"|"unrelated"|"unknown", ...],
 "reasoning": "<= 2 short sentences"}
"""


def _optional(name: str, default: str) -> str:
    return os.getenv(name) or default


def is_enabled() -> bool:
    raw = (os.getenv("EVIDENCE_VALIDATOR_ENABLED") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


class _ValidatorClient:
    """Thin shim so the validator can talk to either an OpenAI-compatible
    endpoint (e.g. HF router used by ``PROFILE_LLM_*``) or a
    ``huggingface_hub.InferenceClient`` provider (e.g. Novita).
    """

    def __init__(self, kind: str, inner: Any, model: str) -> None:
        self.kind = kind
        self.inner = inner
        self.model = model

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 300,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.kind == "openai":
            resp = self.inner.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""

        resp = self.inner.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content or ""


def make_validator_client() -> tuple["_ValidatorClient | None", str, str]:
    """Build the LLM client used by the validator.

    Preference order:
      1. ``EVIDENCE_VALIDATOR_*`` env (explicit overrides).
      2. ``PROFILE_LLM_*`` env (already proven OpenAI-compatible HF router).
      3. ``EVIDENCE_LLM_*`` env via ``huggingface_hub.InferenceClient``.

    Falls back to (None, "", "") if no token / endpoint is configured;
    callers then use the keyword-only heuristic.
    """
    explicit_base = os.getenv("EVIDENCE_VALIDATOR_BASE_URL")
    explicit_key = os.getenv("EVIDENCE_VALIDATOR_API_KEY")
    explicit_provider = os.getenv("EVIDENCE_VALIDATOR_PROVIDER")
    explicit_model = os.getenv("EVIDENCE_VALIDATOR_MODEL")
    timeout = int(_optional("EVIDENCE_VALIDATOR_TIMEOUT_SEC", "120"))

    if explicit_base and explicit_key and _OpenAIClient is not None:
        client = _OpenAIClient(
            base_url=explicit_base,
            api_key=explicit_key,
            timeout=timeout,
        )
        model = explicit_model or _optional("PROFILE_LLM_MODEL", "")
        return _ValidatorClient("openai", client, model), "openai_compatible", model

    profile_base = os.getenv("PROFILE_LLM_BASE_URL")
    profile_key = os.getenv("PROFILE_LLM_API_KEY")
    profile_model = os.getenv("PROFILE_LLM_MODEL")
    profile_provider = (os.getenv("PROFILE_LLM_PROVIDER") or "").lower()
    if (
        profile_base
        and profile_key
        and profile_model
        and profile_provider in {"openai", "lmstudio", "openai_compatible"}
        and _OpenAIClient is not None
    ):
        model = explicit_model or profile_model
        client = _OpenAIClient(
            base_url=profile_base,
            api_key=profile_key,
            timeout=timeout,
        )
        return _ValidatorClient("openai", client, model), "openai_compatible", model

    token = (
        explicit_key
        or os.getenv("NOVITA_API_KEY")
        or os.getenv("HF_TOKEN")
    )
    if not token:
        return None, "", ""

    provider = explicit_provider or _optional("EVIDENCE_LLM_PROVIDER", "novita")
    model = explicit_model or _optional("EVIDENCE_LLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    hf_client = InferenceClient(token=token, provider=provider, timeout=timeout)
    return _ValidatorClient("hf", hf_client, model), provider, model


def _call_with_retries(fn, label: str = "Validator"):
    last_exc = None
    for attempt in range(1, 3):
        try:
            return fn()
        except RETRY_EXCEPTIONS as exc:
            last_exc = exc
            if attempt == 2:
                break
            wait = 4 * attempt
            print(
                f"{label} call failed ({type(exc).__name__}), retrying in {wait}s",
                flush=True,
            )
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed without exception")


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _heuristic_scan(file_name: str, summary: str) -> dict[str, Any]:
    """Run the keyword/word-boundary scan once and return all findings.

    Returns a dict with ``cleaned`` text plus distinct lower-cased matches
    for strong / medium / out-of-scope. This is shared by the standalone
    heuristic verdict AND by the post-LLM defense-in-depth cross-check.
    """
    cleaned = _clean_for_matching(file_name, summary)
    return {
        "cleaned": cleaned,
        "strong_hits": _find_matches(_STRONG_RE, cleaned),
        "medium_hits": _find_matches(_MEDIUM_RE, cleaned),
        "out_of_scope_hits": _find_matches(_OUT_OF_SCOPE_RE, cleaned),
    }


def _infer_categories(hits: list[str]) -> list[str]:
    cats: list[str] = []
    if any("cursor" in h for h in hits):
        cats.append("cursor")
    if any("copilot" in h for h in hits):
        cats.append("copilot")
    if not cats and hits:
        cats.append("llm_for_coding")
    return cats


def _heuristic_validate(file_name: str, summary: str) -> dict[str, Any]:
    """Keyword-only fallback when the LLM is unavailable.

    The summarizer adds boilerplate framing ("R Systems AI Proficiency
    Evidence Review", "Framework Checklist Evidence", etc.) to every output,
    which used to falsely match the heuristic on every document. We strip
    those phrases first, then require either a STRONG keyword match (a named
    coding tool / Cursor-specific model) or at least 2 distinct MEDIUM
    matches. Out-of-scope patterns (telecom bills, timesheets, payslips...)
    veto everything. Keywords are matched with word boundaries so "cursor"
    no longer matches inside "precursor" / "discursive".
    """
    scan = _heuristic_scan(file_name, summary)
    strong_hits = scan["strong_hits"]
    medium_hits = scan["medium_hits"]
    out_hits = scan["out_of_scope_hits"]

    if out_hits:
        return {
            "is_relevant": False,
            "confidence": 0.85,
            "categories": ["unrelated"],
            "reasoning": (
                "Heuristic veto — out-of-scope content detected: "
                f"{', '.join(out_hits[:4])}."
            ),
            "method": "heuristic",
        }

    categories = _infer_categories(strong_hits + medium_hits)

    if strong_hits:
        return {
            "is_relevant": True,
            "confidence": 0.8,
            "categories": categories or ["other_ai_coding_tool"],
            "reasoning": (
                f"Heuristic strong match: {', '.join(strong_hits[:6])}."
            ),
            "method": "heuristic",
        }

    if len(medium_hits) >= 2:
        return {
            "is_relevant": True,
            "confidence": 0.55,
            "categories": categories or ["llm_for_coding"],
            "reasoning": (
                f"Heuristic medium matches (>=2 distinct): "
                f"{', '.join(medium_hits[:6])}."
            ),
            "method": "heuristic",
        }

    return {
        "is_relevant": False,
        "confidence": 0.7,
        "categories": ["unknown"],
        "reasoning": (
            "No clear mention of Cursor, Copilot, or other AI coding tools "
            "in the summary "
            f"(weak signals only: {', '.join(medium_hits) or 'none'})."
        ),
        "method": "heuristic",
    }


_REJECT_CATEGORIES = {"unrelated", "unknown"}

_VISION_JUNK_PHRASES: tuple[str, ...] = (
    "gpt researcher",
    "gpt-researcher",
    "localhost:3000",
    "localhost:3001",
    "localhost:30",
    "timesheet manual",
    "timesheet app",
    "user manual for a timesheet",
    "utility bill",
    "telecom bill",
    "phone bill",
    "electricity bill",
    "grocery receipt",
    "wedding photo",
    "passport",
    "payslip",
    "salary slip",
)

_CURSOR_ANALYTICS_PHRASES: tuple[str, ...] = (
    "work type",
    "intent distribution",
    "conversation insights",
    "usage analytics",
    "active days",
)


def _vision_source_of_truth_enabled() -> bool:
    raw = (os.getenv("EVIDENCE_VISION_SOURCE_OF_TRUTH") or "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _vision_junk_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [p for p in _VISION_JUNK_PHRASES if p in lowered]


def _cursor_analytics_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [p for p in _CURSOR_ANALYTICS_PHRASES if p in lowered]


def _should_accept_from_vision_text(
    file_name: str,
    reasoning: str,
    extra_text: str = "",
) -> tuple[bool, str, list[str]]:
    """Accept from vision reasoning + filename when Cursor/Copilot signals appear."""
    blob = f"{file_name}\n{reasoning}\n{extra_text}"
    if _vision_junk_hits(blob):
        return False, "", []

    scan = _heuristic_scan(file_name, blob)
    if scan["out_of_scope_hits"]:
        return False, "", []

    strong_hits = scan["strong_hits"]
    medium_hits = scan["medium_hits"]
    analytics_hits = _cursor_analytics_hits(blob)
    lowered = blob.lower()

    has_codex = "codex" in lowered or any("codex" in h for h in strong_hits)
    has_composer = "composer" in lowered or any("composer" in h for h in strong_hits)
    has_usage_ctx = any(
        w in lowered for w in ("usage", "spend", "billing", "your usage", "cumulative")
    )

    if strong_hits:
        return True, ", ".join(strong_hits[:6]), strong_hits
    if has_composer and has_usage_ctx:
        return True, "composer + usage/spend context", list(strong_hits) + list(medium_hits)
    if has_codex and has_usage_ctx:
        return True, "codex + usage/spend context", list(strong_hits) + list(medium_hits)
    # Cursor usage analytics UI (work type / intent charts) — common false negative
    if len(analytics_hits) >= 2:
        return True, ", ".join(analytics_hits[:4]), analytics_hits
    if len(medium_hits) >= 2:
        return True, ", ".join(medium_hits[:6]), medium_hits
    return False, "", []


def apply_vision_keyword_override(
    verdict: dict[str, Any],
    file_name: str,
    *,
    extra_text: str = "",
) -> dict[str, Any]:
    """Veto junk on accept; accept when vision said reject but text has Cursor signals."""
    reasoning = str(verdict.get("reasoning") or "")
    blob = f"{file_name}\n{reasoning}\n{extra_text}"
    junk = _vision_junk_hits(blob)
    if junk:
        if verdict.get("is_relevant"):
            return {
                **verdict,
                "is_relevant": False,
                "categories": ["unrelated"],
                "reasoning": (
                    f"Vision junk veto — {', '.join(junk[:3])}. "
                    f"(Vision said: {reasoning[:120]})"
                ),
                "cross_check": "vision_junk_veto",
                "phase": "pre_validate",
            }
        return verdict

    accept, label, hits = _should_accept_from_vision_text(
        file_name, reasoning, extra_text
    )
    if not accept:
        return verdict
    if verdict.get("is_relevant"):
        return verdict

    categories = _infer_categories(hits) or ["cursor"]
    return {
        **verdict,
        "is_relevant": True,
        "confidence": max(float(verdict.get("confidence") or 0), 0.78),
        "categories": categories,
        "reasoning": (
            f"Vision keyword override — {label}. "
            f"(Vision LLM said: {reasoning[:160]})"
        ),
        "method": "llm_vision_prevalidate",
        "cross_check": "vision_keyword_accept",
        "phase": "pre_validate",
    }


def finalize_vision_prevalidate_verdict(
    verdict: dict[str, Any],
    file_name: str,
    *,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Junk veto + keyword accept; final gate for image pre-validate."""
    extra = " ".join(categories or [])
    verdict = apply_vision_keyword_override(verdict, file_name, extra_text=extra)
    if verdict.get("is_relevant"):
        return verdict
    accept, label, hits = _should_accept_from_vision_text(
        file_name,
        str(verdict.get("reasoning") or ""),
        extra,
    )
    if not accept:
        return verdict
    return {
        **verdict,
        "is_relevant": True,
        "confidence": max(float(verdict.get("confidence") or 0), 0.78),
        "categories": _infer_categories(hits) or ["cursor"],
        "reasoning": (
            f"Vision finalize accept — {label}. "
            f"(Prior: {str(verdict.get('reasoning') or '')[:120]})"
        ),
        "method": "llm_vision_prevalidate",
        "cross_check": "vision_finalize_accept",
        "phase": "pre_validate",
    }


def entry_passed_vision_prevalidate(entry: dict[str, Any]) -> bool:
    """True when an image was accepted by Qwen-VL vision pre-validate."""
    pre = entry.get("pre_validation") or {}
    if not pre.get("is_relevant"):
        return False
    method = str(pre.get("method") or "")
    if method.startswith("llm_vision") or method == "llm_vision_prevalidate":
        return True
    return pre.get("cross_check") in {
        "vision_keyword_accept",
        "vision_finalize_accept",
    }


def _cross_check_llm_verdict(
    verdict: dict[str, Any],
    file_name: str,
    summary: str,
) -> dict[str, Any]:
    """Defense-in-depth on top of the LLM verdict.

    The LLM is generally good but can hallucinate "is_relevant=true" on a
    payslip or a totally unrelated PDF. We re-run the cheap keyword scan
    and override the verdict in these cases:

      1. The LLM accepted but the summary contains an out-of-scope
         pattern (telecom bill, payslip, passport, ...). Reject.
      2. The LLM accepted but only flagged ``unrelated`` / ``unknown``
         categories. Reject (contradictory output).
      3. The LLM accepted but the cleaned summary contains zero strong
         AND zero medium keywords. Demote to rejected — there is no
         evidence in the text for the LLM's claim.

    All other LLM verdicts are passed through unchanged. The verdict dict
    is annotated with ``cross_check`` describing what fired.
    """
    if not verdict.get("is_relevant"):
        return verdict

    scan = _heuristic_scan(file_name, summary)
    out_hits = scan["out_of_scope_hits"]
    strong_hits = scan["strong_hits"]
    medium_hits = scan["medium_hits"]

    cats = [c for c in (verdict.get("categories") or []) if c]
    only_reject_cats = bool(cats) and all(c in _REJECT_CATEGORIES for c in cats)

    if out_hits:
        return {
            **verdict,
            "is_relevant": False,
            "categories": ["unrelated"],
            "reasoning": (
                "Cross-check override — LLM accepted but out-of-scope "
                f"content detected: {', '.join(out_hits[:4])}. "
                f"(LLM said: {verdict.get('reasoning', '').strip()})"
            ),
            "cross_check": "out_of_scope_veto",
        }

    if only_reject_cats:
        return {
            **verdict,
            "is_relevant": False,
            "reasoning": (
                "Cross-check override — LLM said is_relevant=true but only "
                f"returned reject-style categories ({', '.join(cats)})."
            ),
            "cross_check": "contradictory_categories",
        }

    if not strong_hits and not medium_hits:
        return {
            **verdict,
            "is_relevant": False,
            "categories": ["unknown"],
            "reasoning": (
                "Cross-check override — LLM accepted but the summary "
                "contains no recognizable AI-coding-tool keyword "
                "(no Cursor / Copilot / Claude / Codex / etc.). "
                f"(LLM said: {verdict.get('reasoning', '').strip()})"
            ),
            "cross_check": "no_keyword_evidence",
        }

    return verdict


def _trim_summary(text: str, limit: int = 6000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... summary truncated for validator ...]"


def validate_evidence_item(
    client: "_ValidatorClient | None",
    model: str,
    file_name: str,
    file_type: str,
    summary: str,
    *,
    min_confidence: float = 0.5,
    content_kind: str = "summary",
) -> dict[str, Any]:
    """Classify evidence text as relevant / not relevant.

    ``content_kind`` is ``"summary"`` (default, post-summarize) or ``"raw_extract"``
    (pre-summarize gate on locally extracted file text).

    Returns a dict with at least:
        is_relevant : bool
        confidence  : float in [0, 1]
        categories  : list[str]
        reasoning   : str
        method      : "llm" | "heuristic"
    """
    if not summary or not summary.strip():
        return {
            "is_relevant": False,
            "confidence": 1.0,
            "categories": ["unknown"],
            "reasoning": "Empty content; nothing to validate.",
            "method": "trivial",
        }

    if client is None or not (model or getattr(client, "model", "")):
        return _heuristic_validate(file_name, summary)

    effective_model = model or client.model
    system_prompt = (
        VALIDATOR_RAW_SYSTEM_PROMPT
        if content_kind == "raw_extract"
        else VALIDATOR_SYSTEM_PROMPT
    )
    content_label = (
        "RAW EXTRACT (not summarized)" if content_kind == "raw_extract" else "SUMMARY"
    )

    user_prompt = (
        f"FILE NAME: {file_name}\n"
        f"FILE TYPE: {file_type}\n\n"
        f"{content_label}:\n\"\"\"\n{_trim_summary(summary)}\n\"\"\"\n\n"
        "Classify this evidence. Respond with the JSON object only."
    )

    def _call() -> str:
        prev_model = client.model
        client.model = effective_model
        try:
            return client.chat_json(system_prompt, user_prompt)
        finally:
            client.model = prev_model

    try:
        raw = _call_with_retries(_call, label="Validator LLM")
    except Exception as exc:
        heur = _heuristic_validate(file_name, summary)
        heur["reasoning"] = (
            f"LLM validator unavailable ({type(exc).__name__}: {exc}); "
            f"fell back to heuristic. {heur['reasoning']}"
        )
        heur["method"] = "heuristic_fallback"
        return heur

    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict) or "is_relevant" not in parsed:
        heur = _heuristic_validate(file_name, summary)
        heur["reasoning"] = (
            "LLM returned unparseable response; fell back to heuristic. "
            f"{heur['reasoning']}"
        )
        heur["method"] = "heuristic_fallback"
        heur["raw_response"] = raw[:400]
        return heur

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    categories = parsed.get("categories") or []
    if not isinstance(categories, list):
        categories = [str(categories)]
    categories = [str(c).strip().lower() for c in categories if str(c).strip()]

    is_relevant = bool(parsed.get("is_relevant"))
    if is_relevant and confidence < min_confidence:
        # Treat low-confidence positives as rejected to be safe.
        is_relevant = False

    verdict = {
        "is_relevant": is_relevant,
        "confidence": round(confidence, 3),
        "categories": categories or ["unknown"],
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "method": "llm",
    }

    # Defense-in-depth: even if the LLM said "relevant", veto using the
    # cheap keyword heuristic when the summary screams off-topic or
    # contains zero AI-coding-tool evidence. This catches LLM
    # hallucinations on payslips, telecom bills, holiday photos, etc.
    return _cross_check_llm_verdict(verdict, file_name, summary)


def validate_raw_extract(
    client: "_ValidatorClient | None",
    model: str,
    file_name: str,
    file_type: str,
    raw_text: str,
    *,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    """Pre-summarize relevance gate on locally extracted file text."""
    verdict = validate_evidence_item(
        client,
        model,
        file_name,
        file_type,
        raw_text,
        min_confidence=min_confidence,
        content_kind="raw_extract",
    )
    verdict["phase"] = "pre_validate"
    return verdict


def has_relevant_evidence(bundle: dict[str, Any] | None) -> bool:
    """True when the bundle has at least one accepted summary for the profile."""
    if not bundle:
        return False
    for entry in bundle.get("summaries") or []:
        if entry.get("error"):
            continue
        validation = entry.get("validation") or entry.get("pre_validation") or {}
        if validation.get("is_relevant") is False:
            continue
        if (entry.get("summary") or "").strip():
            return True
    return False


def _rebuild_combined_text(accepted: list[dict[str, Any]], bundle: dict[str, Any]) -> str:
    email = bundle.get("email", "")
    period = bundle.get("period") or {}
    folder = bundle.get("evidence_folder", "")
    parts = [
        f"Evidence summaries for {email}",
        f"Period: {period.get('start', '?')} to {period.get('end', '?')}",
        f"Source folder: {folder}",
        "",
    ]
    kept_any = False
    for entry in accepted:
        if entry.get("error"):
            continue
        kept_any = True
        parts.append(f"### {entry.get('file')}")
        parts.append(entry.get("summary") or "")
        parts.append("")

    if not kept_any:
        parts.append(
            "(All uploaded evidence was rejected by the validator as not "
            "related to Cursor / Copilot / AI coding tool usage.)"
        )
    return "\n".join(parts).strip()


def validate_evidence_bundle(
    bundle: dict[str, Any],
    *,
    client: "_ValidatorClient | None" = None,
    model: str | None = None,
    min_confidence: float | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Annotate and filter a bundle produced by ``summarize_evidence``.

    The bundle is mutated in place. Returns the same bundle for chaining.

    Adds to each entry in ``summaries``:
        entry["validation"] = {is_relevant, confidence, categories, reasoning,
                               method}

    Adds top-level:
        bundle["files_rejected"] = [file names that the validator rejected]
        bundle["validator"]      = {provider, model, enabled, min_confidence}

    Rebuilds ``combined_evidence_text`` from the accepted items only.
    Recomputes ``files_processed`` to exclude rejected files.
    """
    if not is_enabled():
        bundle["validator"] = {"enabled": False, "reason": "disabled via env"}
        bundle.setdefault("files_rejected", [])
        return bundle

    if client is None or not model:
        client, _provider, resolved_model = make_validator_client()
        model = model or resolved_model

    if min_confidence is None:
        try:
            min_confidence = float(
                os.getenv("EVIDENCE_VALIDATOR_MIN_CONFIDENCE") or "0.5"
            )
        except ValueError:
            min_confidence = 0.5

    summaries: list[dict[str, Any]] = bundle.get("summaries") or []
    prior_rejected: list[dict[str, Any]] = list(bundle.get("rejected_summaries") or [])
    prior_rejected_files: list[str] = list(bundle.get("files_rejected") or [])
    accepted_entries: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []
    rejected_files: list[str] = []
    processed_files: list[str] = []

    for i, entry in enumerate(summaries, start=1):
        file_name = entry.get("file", f"item_{i}")
        file_type = entry.get("type", "unknown")
        summary = entry.get("summary") or ""
        error = entry.get("error")

        if error:
            entry["validation"] = {
                "is_relevant": False,
                "confidence": 1.0,
                "categories": ["unknown"],
                "reasoning": f"Skipped — summarization failed: {error}",
                "method": "skipped",
            }
            rejected_entries.append(entry)
            rejected_files.append(file_name)
            continue

        if _vision_source_of_truth_enabled() and entry_passed_vision_prevalidate(entry):
            pre = dict(entry.get("pre_validation") or {})
            entry["validation"] = {
                **pre,
                "phase": "vision_source_of_truth",
                "post_validate": "skipped",
            }
            processed_files.append(file_name)
            accepted_entries.append(entry)
            if log:
                print(
                    f"  validating [{i}/{len(summaries)}] {file_name} ...",
                    flush=True,
                )
                print(
                    "    -> KEEP (vision pre-validate is source of truth; "
                    "skipped text post-validate)",
                    flush=True,
                )
            continue

        if log:
            print(f"  validating [{i}/{len(summaries)}] {file_name} ...", flush=True)
        try:
            verdict = validate_evidence_item(
                client,
                model or "",
                file_name=file_name,
                file_type=file_type,
                summary=summary,
                min_confidence=min_confidence,
            )
        except Exception as exc:
            verdict = _heuristic_validate(file_name, summary)
            verdict["method"] = "heuristic_error"
            verdict["reasoning"] = (
                f"Validator crashed ({type(exc).__name__}: {exc}); "
                f"fell back to heuristic. {verdict['reasoning']}"
            )

        verdict = {**verdict, "phase": verdict.get("phase") or "post_validate"}
        entry["validation"] = verdict
        if verdict.get("is_relevant"):
            processed_files.append(file_name)
            accepted_entries.append(entry)
        else:
            rejected_files.append(file_name)
            rejected_entries.append(entry)

        if log:
            tag = "KEEP" if verdict.get("is_relevant") else "REJECT"
            print(
                f"    -> {tag} (conf={verdict.get('confidence')}, "
                f"cats={','.join(verdict.get('categories') or [])})",
                flush=True,
            )

    bundle["summaries"] = accepted_entries
    bundle["rejected_summaries"] = prior_rejected + rejected_entries
    bundle["files_processed"] = processed_files
    bundle["files_rejected"] = list(
        dict.fromkeys(prior_rejected_files + rejected_files)
    )
    bundle["combined_evidence_text"] = _rebuild_combined_text(accepted_entries, bundle)
    bundle["validator"] = {
        "enabled": True,
        "model": model or "",
        "min_confidence": min_confidence,
        "rejected_count": len(rejected_files),
        "accepted_count": len(processed_files),
    }
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an existing evidence bundle JSON in place."
    )
    parser.add_argument("--bundle", required=True, help="Path to evidence_*.json")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Override EVIDENCE_VALIDATOR_MIN_CONFIDENCE (default 0.5)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the bundle JSON; default writes <stem>.validated.json",
    )
    args = parser.parse_args()

    path = Path(args.bundle).resolve()
    if not path.is_file():
        print(f"Bundle not found: {path}", file=sys.stderr)
        return 2

    bundle = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_evidence_bundle(bundle, min_confidence=args.min_confidence)

    out_path = path if args.in_place else path.with_name(path.stem + ".validated.json")
    out_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    accepted = len(bundle.get("files_processed") or [])
    rejected = len(bundle.get("files_rejected") or [])
    print(f"Validated -> {out_path}")
    print(f"  kept:     {accepted}")
    print(f"  rejected: {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

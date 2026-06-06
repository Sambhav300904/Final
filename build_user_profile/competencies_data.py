"""
R Systems AI Coding Proficiency Framework — full competency dataset.

Contains all 8 competencies and the 43 evidence-backed checklist items.
The text of each item is preserved exactly as defined in the framework
document and MUST NOT be paraphrased. This module is the single source
of truth used by the Pinecone ingest script and by downstream agents.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass(frozen=True)
class CompetencyItem:
    """A single checkbox item in the proficiency framework."""

    item_id: int
    competency_id: int
    competency_name: str
    text: str


COMPETENCIES: dict[int, str] = {
    1: "Responsible AI Usage and Compliance",
    2: "AI-Assisted Delivery Fundamentals",
    3: "Verification, Quality, and Engineering Rigor",
    4: "SDLC Breadth and Product Thinking",
    5: "Advanced AI Tooling Proficiency",
    6: "Complex Delivery and Modernization Impact",
    7: "Enablement and Talent Multiplication",
    8: "Business Impact and Reusable Assets",
}


_RAW_ITEMS: list[tuple[int, int, str]] = [
    # Competency 1: Responsible AI Usage and Compliance
    (
        1,
        1,
        "The engineer completed the internal AI basics and safe usage module or quiz, and the evidence is the completion record.",
    ),
    (
        2,
        1,
        "The engineer signed the safe usage declaration stating they will not share secrets or customer-sensitive data in AI tools, and the evidence is the acknowledgement.",
    ),
    # Competency 2: AI-Assisted Delivery Fundamentals
    (
        3,
        2,
        "The engineer delivered an AI-assisted code change that was merged, and the evidence is the pull request link.",
    ),
    (
        4,
        2,
        "The engineer delivered a user story or feature using AI assistance, and the evidence is the ticket link and merged pull request link.",
    ),
    (
        5,
        2,
        "The engineer fixed a defect or failing test using AI assistance, and the evidence is the defect reference and merged pull request link.",
    ),
    (
        6,
        2,
        "The engineer added or improved unit tests for an AI-assisted change, and the evidence is the pull request link showing the test additions.",
    ),
    (
        7,
        2,
        "The engineer improved readability or maintainability of code using AI assistance, and the evidence is a refactor pull request with reviewer approval.",
    ),
    (
        8,
        2,
        "The engineer used AI assistance to produce or improve developer-facing documentation for delivered work, and the evidence is the document link or pull request link.",
    ),
    # Competency 3: Verification, Quality, and Engineering Rigor
    (
        9,
        3,
        "Tests were executed for AI-assisted changes, and the evidence is CI status or test output.",
    ),
    (
        10,
        3,
        "Linting, formatting, or type checks were run for AI-assisted changes, and the evidence is CI status or output.",
    ),
    (
        11,
        3,
        "The engineer used AI to identify edge cases and updated code or tests accordingly, and the evidence is the pull request description and related changes.",
    ),
    (
        12,
        3,
        "The engineer used AI to generate a test plan or test checklist for a change, and the evidence is the note or ticket comment.",
    ),
    (
        13,
        3,
        "The engineer ran or referenced static analysis or security checks where applicable, and the evidence is scan output or CI status.",
    ),
    (
        14,
        3,
        "The engineer used AI to debug using logs or stack traces and documented root cause and fix rationale, and the evidence is the pull request description or incident note.",
    ),
    (
        15,
        3,
        "The engineer produced a clear review narrative for an AI-assisted change, including what AI generated and what was modified, and the evidence is the pull request template section.",
    ),
    # Competency 4: SDLC Breadth and Product Thinking
    (
        16,
        4,
        "The engineer used AI to clarify requirements or draft acceptance criteria, and the evidence is a ticket comment or short note.",
    ),
    (
        17,
        4,
        "The engineer used AI to break a story into implementation tasks or risks, and the evidence is the task breakdown in the ticket.",
    ),
    (
        18,
        4,
        "The engineer created a short design note or ADR using AI assistance, including tradeoffs and edge cases, and the evidence is the note link.",
    ),
    (
        19,
        4,
        "The engineer used AI to propose or refine an API contract, schema, interface, or integration contract, and the evidence is the specification or pull request link.",
    ),
    (
        20,
        4,
        "The engineer used AI to create or improve integration tests, contract tests, test data, or mocks, and the evidence is the pull request link.",
    ),
    (
        21,
        4,
        "The engineer used AI to improve CI, build scripts, pipelines, or DevOps automation, and the evidence is the pipeline or configuration change link.",
    ),
    (
        22,
        4,
        "The engineer used AI to update operational documentation such as runbooks, release notes, or troubleshooting guides, and the evidence is the document link.",
    ),
    # Competency 5: Advanced AI Tooling Proficiency
    (
        23,
        5,
        "The engineer completed a multi-file change using an advanced workflow such as an agent or composer approach, and the evidence is a pull request showing coordinated edits across multiple files.",
    ),
    (
        24,
        5,
        "The engineer demonstrated effective codebase context selection, such as grounding changes on the right modules and references, and the evidence is the pull request narrative describing how context was chosen.",
    ),
    (
        25,
        5,
        "The engineer used an iterative workflow that includes planning, implementation, testing, and refinement, and the evidence is a pull request showing multiple iterations with verification.",
    ),
    (
        26,
        5,
        "The engineer used AI to generate tests first or in parallel with implementation and showed passing results, and the evidence is the pull request and CI run.",
    ),
    (
        27,
        5,
        "The engineer used AI to perform a non-trivial refactor under constraints such as style rules, performance limits, or backward compatibility, and the evidence is the pull request with constraints stated.",
    ),
    (
        28,
        5,
        "The engineer created and shared a reusable prompt recipe or workflow pattern for advanced usage, and the evidence is the shared internal note.",
    ),
    (
        29,
        5,
        "The engineer demonstrated the ability to detect and correct hallucinations by validating AI output against source code, documentation, or tests, and the evidence is a pull request narrative or review comment showing the correction.",
    ),
    (
        30,
        5,
        "The engineer demonstrated good judgment about when not to use AI, such as for sensitive data, high-risk changes, or ambiguous requirements, and the evidence is a short decision note or a pull request comment explaining the choice.",
    ),
    # Competency 6: Complex Delivery and Modernization Impact
    (
        31,
        6,
        "The engineer delivered a modernization slice from a legacy codebase to a modern codebase, and the evidence is a migration pull request set and a short migration note.",
    ),
    (
        32,
        6,
        "The engineer contributed to a framework or platform upgrade with a phased rollout and rollback plan, and the evidence is an ADR or release plan plus related pull requests.",
    ),
    (
        33,
        6,
        "The engineer delivered a measurable performance or reliability improvement, and the evidence is a before-and-after benchmark, SLO note, or comparable proof.",
    ),
    (
        34,
        6,
        "The engineer improved automated test coverage or reduced flaky tests in a meaningful way, and the evidence is coverage reports, flake reduction notes, or CI metrics.",
    ),
    (
        35,
        6,
        "The engineer architected and delivered an AI-enabled feature or system, and the evidence is an architecture note and a delivered implementation link.",
    ),
    (
        36,
        6,
        "The engineer designed and implemented a multi-step or multi-agent orchestration workflow, and the evidence is a workflow diagram or design note and a repository or pull request link.",
    ),
    # Competency 7: Enablement and Talent Multiplication
    (
        37,
        7,
        "The engineer coached at least two Aspirants or Beginners and documented outcomes briefly, and the evidence is a short mentoring log or manager note.",
    ),
    (
        38,
        7,
        "The engineer ran at least one enablement session or two office-hour slots and captured attendees and takeaways, and the evidence is the invite and notes.",
    ),
    (
        39,
        7,
        "The engineer contributed to maintaining an internal playbook, standards page, or shared guidance that others reference, and the evidence is the page link and edit history.",
    ),
    (
        40,
        7,
        "The engineer created a reusable checklist or template that improved team consistency, and the evidence is the artifact link and adoption note.",
    ),
    # Competency 8: Business Impact and Reusable Assets
    (
        41,
        8,
        "The engineer contributed to an RFP response or proposal material by authoring or refining solution content, and the evidence is the internal document or slide link.",
    ),
    (
        42,
        8,
        "The engineer built a proof of concept or demo that supported a pursuit or unblocked delivery, and the evidence is the repository link and a short summary of the problem, approach, and outcome.",
    ),
    (
        43,
        8,
        "The engineer built an accelerator such as a scaffold, harness, automation script, or template that was adopted beyond a single engineer, and the evidence is the asset link and adoption record, plus an adoption note.",
    ),
]


COMPETENCY_ITEMS: List[CompetencyItem] = [
    CompetencyItem(
        item_id=item_id,
        competency_id=comp_id,
        competency_name=COMPETENCIES[comp_id],
        text=text,
    )
    for item_id, comp_id, text in _RAW_ITEMS
]


def all_items() -> List[CompetencyItem]:
    """Return all 43 competency items in framework order."""
    return list(COMPETENCY_ITEMS)


def items_by_competency(competency_id: int) -> List[CompetencyItem]:
    """Return items belonging to a single competency (1..8)."""
    if competency_id not in COMPETENCIES:
        raise ValueError(f"Unknown competency_id: {competency_id}")
    return [item for item in COMPETENCY_ITEMS if item.competency_id == competency_id]


def to_dict_list() -> list[dict]:
    """Return the dataset as a list of plain dicts (useful for JSON dumps)."""
    return [asdict(item) for item in COMPETENCY_ITEMS]


if __name__ == "__main__":
    assert len(COMPETENCY_ITEMS) == 43, "Expected exactly 43 framework items"
    for cid, name in COMPETENCIES.items():
        count = len(items_by_competency(cid))
        print(f"Competency {cid}: {name} ({count} items)")
    print(f"\nTotal items: {len(COMPETENCY_ITEMS)}")

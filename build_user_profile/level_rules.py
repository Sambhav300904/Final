"""
Framework level assignment from checked item IDs (telemetry or manager-approved).

Evaluates Champion -> Builder -> Beginner -> Aspirant; returns the highest level
where every rule passes. One item may satisfy multiple rules (double-counting
is allowed by design, matching the framework rubric).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Any, Iterable

from competencies_data import COMPETENCY_ITEMS

RULES_VERSION = "v2"

LEVEL_ORDER = ("Champion", "Builder", "Beginner", "Aspirant")
BELOW_ASPIRANT = "Below Aspirant"

_MANDATORY = frozenset({1, 2})
_ASPIRANT_STARTER = frozenset({3, 8, 12, 15, 16})
_BEGINNER_PR = frozenset({3, 4, 5})
_BEGINNER_ARTIFACT = frozenset({8, 12, 16})
_BUILDER_MULTIPLIER = frozenset({28, 40})


def items_for_competency(competency_id: int) -> frozenset[int]:
    return frozenset(
        item.item_id
        for item in COMPETENCY_ITEMS
        if item.competency_id == competency_id
    )


COMPETENCY_ITEM_IDS: dict[int, frozenset[int]] = {
    cid: items_for_competency(cid) for cid in range(1, 9)
}


@dataclass(frozen=True)
class Rule:
    """Single pass/fail condition for a level."""

    id: str
    label: str
    required: frozenset[int] | None = None
    min_count: int | None = None
    from_set: frozenset[int] | None = None
    from_competency: int | None = None

    def evaluate(self, checked: AbstractSet[int]) -> dict[str, Any]:
        if self.required is not None:
            matched = sorted(self.required & checked)
            missing = sorted(self.required - checked)
            return {
                "id": self.id,
                "label": self.label,
                "passed": len(missing) == 0,
                "required_count": len(self.required),
                "matched_count": len(matched),
                "matched_item_ids": matched,
                "missing_item_ids": missing,
            }

        if self.from_competency is not None:
            pool = COMPETENCY_ITEM_IDS[self.from_competency]
        elif self.from_set is not None:
            pool = self.from_set
        else:
            raise ValueError(f"Rule {self.id} has no item pool")

        need = self.min_count or 0
        matched = sorted(pool & checked)
        return {
            "id": self.id,
            "label": self.label,
            "passed": len(matched) >= need,
            "required_count": need,
            "matched_count": len(matched),
            "matched_item_ids": matched,
            "pool_item_ids": sorted(pool),
        }


def _mandatory_rule() -> Rule:
    return Rule(
        id="mandatory_1_2",
        label="Items 1 and 2 must be checked",
        required=_MANDATORY,
    )


LEVEL_RULES: dict[str, list[Rule]] = {
    "Aspirant": [
        _mandatory_rule(),
        Rule(
            id="aspirant_starter",
            label="Any 2 from items 3, 8, 12, 15, 16",
            min_count=2,
            from_set=_ASPIRANT_STARTER,
        ),
    ],
    "Beginner": [
        _mandatory_rule(),
        Rule(
            id="beginner_verification",
            label="At least 2 from Competency 3 (items 9-15)",
            min_count=2,
            from_competency=3,
        ),
        Rule(
            id="beginner_merged_pr",
            label="At least 1 from items 3, 4, or 5",
            min_count=1,
            from_set=_BEGINNER_PR,
        ),
        Rule(
            id="beginner_sdlc_artifact",
            label="At least 1 from items 8, 12, or 16",
            min_count=1,
            from_set=_BEGINNER_ARTIFACT,
        ),
    ],
    "Builder": [
        _mandatory_rule(),
        Rule(
            id="builder_verification",
            label="At least 4 from Competency 3 (items 9-15)",
            min_count=4,
            from_competency=3,
        ),
        Rule(
            id="builder_sdlc",
            label="At least 5 from Competency 4 (items 16-22)",
            min_count=5,
            from_competency=4,
        ),
        Rule(
            id="builder_advanced_tooling",
            label="At least 4 from Competency 5 (items 23-30)",
            min_count=4,
            from_competency=5,
        ),
        Rule(
            id="builder_multiplier",
            label="At least 1 from items 28 or 40",
            min_count=1,
            from_set=_BUILDER_MULTIPLIER,
        ),
    ],
    "Champion": [
        _mandatory_rule(),
        Rule(
            id="champion_verification",
            label="At least 5 from Competency 3 (items 9-15)",
            min_count=5,
            from_competency=3,
        ),
        Rule(
            id="champion_sdlc",
            label="At least 6 from Competency 4 (items 16-22)",
            min_count=6,
            from_competency=4,
        ),
        Rule(
            id="champion_advanced_tooling",
            label="At least 5 from Competency 5 (items 23-30)",
            min_count=5,
            from_competency=5,
        ),
        Rule(
            id="champion_complex_delivery",
            label="At least 2 from Competency 6 (items 31-36)",
            min_count=2,
            from_competency=6,
        ),
        Rule(
            id="champion_enablement",
            label="At least 3 from Competency 7 (items 37-40)",
            min_count=3,
            from_competency=7,
        ),
        Rule(
            id="champion_business_impact",
            label="At least 2 from Competency 8 (items 41-43)",
            min_count=2,
            from_competency=8,
        ),
    ],
}


def _normalize_checked(checked: Iterable[int]) -> frozenset[int]:
    return frozenset(int(i) for i in checked if int(i) > 0)


def evaluate_level_rules(checked: Iterable[int]) -> dict[str, Any]:
    """Return highest passing framework level and per-level rule breakdown."""
    checked_set = _normalize_checked(checked)
    levels_detail: dict[str, Any] = {}
    assigned: str | None = None

    for level in LEVEL_ORDER:
        rule_results = [r.evaluate(checked_set) for r in LEVEL_RULES[level]]
        passed = all(r["passed"] for r in rule_results)
        levels_detail[level] = {"passed": passed, "rules": rule_results}
        if passed and assigned is None:
            assigned = level

    return {
        "level": assigned or BELOW_ASPIRANT,
        "rules_version": RULES_VERSION,
        "checked_item_ids": sorted(checked_set),
        "levels": levels_detail,
    }


def suggested_level_from_checked(checked: Iterable[int]) -> str:
    return evaluate_level_rules(checked)["level"]

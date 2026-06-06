"""Tests for framework level_rules (v2)."""

from __future__ import annotations

import sys
from pathlib import Path

_BUILD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD))

from level_rules import (  # noqa: E402
    BELOW_ASPIRANT,
    evaluate_level_rules,
    suggested_level_from_checked,
)


def test_below_aspirant_only_mandatory():
    assert suggested_level_from_checked({1, 2}) == BELOW_ASPIRANT


def test_below_aspirant_starter_without_mandatory():
    # Two starter items but missing mandatory 1 and 2
    assert suggested_level_from_checked({3, 8, 12}) == BELOW_ASPIRANT


def test_aspirant_minimal():
    assert suggested_level_from_checked({1, 2, 3, 8}) == "Aspirant"


def test_beginner_requires_pr_and_artifact():
    checked = {1, 2, 3, 9, 10, 16}
    assert suggested_level_from_checked(checked) == "Beginner"


def test_beginner_item12_double_counts():
    # 12 satisfies both verification (>=2 with 9) and artifact (>=1)
    checked = {1, 2, 3, 9, 12}
    assert suggested_level_from_checked(checked) == "Beginner"


def test_builder_needs_multiplier():
    comp3 = {9, 10, 11, 12}
    comp4 = {16, 17, 18, 19, 20}
    comp5 = {23, 24, 25, 26}
    without_mult = {1, 2, 3, *comp3, *comp4, *comp5}
    assert suggested_level_from_checked(without_mult) == "Beginner"

    with_mult = without_mult | {28}
    assert suggested_level_from_checked(with_mult) == "Builder"


def test_champion_full_set():
    checked = {
        1,
        2,
        *range(9, 15),   # comp3 x6 (need 5)
        *range(16, 22),  # comp4 x6 (need 6)
        *range(23, 28),  # comp5 x5 (need 5)
        31,
        32,              # comp6 x2
        37,
        38,
        39,              # comp7 x3
        41,
        42,              # comp8 x2
    }
    result = evaluate_level_rules(checked)
    assert result["level"] == "Champion"
    assert result["levels"]["Champion"]["passed"] is True


def test_evaluate_includes_breakdown():
    result = evaluate_level_rules({1, 2, 3, 8})
    assert result["rules_version"] == "v2"
    assert result["levels"]["Aspirant"]["passed"] is True
    assert result["checked_item_ids"] == [1, 2, 3, 8]

from __future__ import annotations

import pytest

from reporting.pdf_report import (
    QUALITY_SCORE_WEIGHTS,
    balanced_quality_score,
    quality_score_components,
)


def test_balanced_quality_score_uses_documented_weights():
    metrics = {
        "cii": 1.2,
        "original_ridge_valley_clarity": 100.0,
        "processed_ridge_valley_clarity": 120.0,
        "original_edge_clarity": 50.0,
        "processed_edge_clarity": 60.0,
        "ssim": 0.95,
    }

    components = quality_score_components(metrics)
    expected = 100.0 * sum(
        QUALITY_SCORE_WEIGHTS[name] * components[name]
        for name in QUALITY_SCORE_WEIGHTS
    )

    assert sum(QUALITY_SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert balanced_quality_score(metrics) == pytest.approx(expected)


def test_balanced_quality_score_is_bounded():
    extreme = {
        "cii": 999.0,
        "original_ridge_valley_clarity": 1.0,
        "processed_ridge_valley_clarity": 999.0,
        "original_edge_clarity": 1.0,
        "processed_edge_clarity": 999.0,
        "ssim": 999.0,
    }
    empty = {}

    assert balanced_quality_score(extreme) == pytest.approx(100.0)
    assert 0.0 <= balanced_quality_score(empty) <= 100.0

"""Tests for JPA calibration analysis helpers."""

from __future__ import annotations

import numpy as np
import pytest

from qubex.contrib.experiment.jpa_calibration import (
    _centered_axis,
    _covariance_flatness,
    _NoValidPoint,
    _refined_axis,
    _select_mux_optimum,
    _state_separation_score,
)


def test_centered_axis_retains_configured_value_after_clipping() -> None:
    """A default clipped axis should still measure its configured center."""
    axis = _centered_axis(0.54, half_width=1.0, minimum=0.0, maximum=4.0)

    assert len(axis) == 9
    assert 0.54 in axis
    assert np.all(np.diff(axis) > 0.0)


def test_state_separation_score_is_invariant_to_iq_origin_and_phase() -> None:
    """State-separation score should ignore a common IQ offset and rotation."""
    ground = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.complex128)
    excited = ground + 2.0

    score = _state_separation_score(ground, excited)
    transformed_score = _state_separation_score(
        (ground + 3.0 - 4.0j) * np.exp(0.37j),
        (excited + 3.0 - 4.0j) * np.exp(0.37j),
    )

    assert score > 0.0
    assert transformed_score == pytest.approx(score, rel=1e-12, abs=1e-12)


def test_state_separation_score_rejects_nonfinite_samples() -> None:
    """State-separation score should reject nonfinite single-shot IQ data."""
    with pytest.raises(ValueError, match="finite"):
        _state_separation_score(
            np.array([0.0, np.nan], dtype=np.complex128),
            np.array([1.0, 2.0], dtype=np.complex128),
        )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        (np.array([1.0, -1.0, 1.0j, -1.0j]), 1.0),
        (np.array([2.0, -2.0, 1.0j, -1.0j]), 2.0),
    ],
)
def test_covariance_flatness_matches_principal_axis_ratio(
    samples: np.ndarray,
    expected: float,
) -> None:
    """Covariance flatness should equal the principal standard-deviation ratio."""
    assert _covariance_flatness(samples) == pytest.approx(
        expected,
        rel=1e-12,
        abs=1e-12,
    )


def test_covariance_flatness_marks_rank_deficient_cloud_as_infinite() -> None:
    """A line-shaped IQ cloud should have infinite covariance flatness."""
    samples = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.complex128)

    assert np.isinf(_covariance_flatness(samples))


def test_refined_axis_keeps_nonuniform_coarse_optimum() -> None:
    """A fine axis should retain the selected point from a nonuniform coarse axis."""
    refined = _refined_axis(
        np.array([0.0, 1.0, 3.0], dtype=np.float64),
        1,
        points=4,
    )

    assert len(refined) == 4
    assert 1.0 in refined
    assert np.all(np.diff(refined) > 0.0)


def test_select_mux_optimum_enforces_relative_flatness_for_every_peer() -> None:
    """A point should be rejected when one peer exceeds its baseline flatness ratio."""
    scores = np.array(
        [
            [[[10.0, 8.0]], [[7.0, 6.0]]],
            [[[10.0, 8.0]], [[7.0, 6.0]]],
        ]
    )
    flatness = np.full_like(scores, 1.5)
    flatness[1, 0, 0, 0] = 1.8

    optimum = _select_mux_optimum(
        scores,
        flatness,
        baseline_scores=[5.0, 5.0],
        baseline_flatness=[1.5, 1.5],
        minimum_score_gain=1.0,
        maximum_flatness_ratio=1.1,
        flatness_threshold=None,
    )

    assert optimum.index == (0, 0, 1)
    assert optimum.aggregate_score == pytest.approx(8.0)
    assert optimum.aggregate_score_gain == pytest.approx(1.6)
    assert not optimum.valid_mask[0, 0, 0]
    assert optimum.on_boundary


def test_select_mux_optimum_ranks_worst_peer_gain_instead_of_raw_score() -> None:
    """Selection should rank relative gain even when raw-score ranking disagrees."""
    scores = np.array(
        [
            [[[1.5, 2.0]]],
            [[[150.0, 140.0]]],
        ]
    )
    flatness = np.ones_like(scores)

    optimum = _select_mux_optimum(
        scores,
        flatness,
        baseline_scores=[1.0, 100.0],
        baseline_flatness=[1.0, 1.0],
        minimum_score_gain=1.0,
        maximum_flatness_ratio=1.1,
        flatness_threshold=None,
    )

    assert optimum.index == (0, 0, 0)
    assert optimum.aggregate_score == pytest.approx(1.5)
    assert optimum.aggregate_score_gain == pytest.approx(1.5)
    np.testing.assert_allclose(
        optimum.aggregate_score_map,
        np.array([[[1.5, 2.0]]]),
    )
    np.testing.assert_allclose(
        optimum.aggregate_score_gain_map,
        np.array([[[1.5, 1.4]]]),
    )


def test_select_mux_optimum_accepts_inclusive_gain_and_flatness_boundaries() -> None:
    """Gain and relative-flatness values equal to their limits should be valid."""
    scores = np.array([[[[1.1]]], [[[2.2]]]])
    flatness = np.array([[[[1.1]]], [[[1.1]]]])

    optimum = _select_mux_optimum(
        scores,
        flatness,
        baseline_scores=[1.0, 2.0],
        baseline_flatness=[1.0, 1.0],
        minimum_score_gain=1.1,
        maximum_flatness_ratio=1.1,
        flatness_threshold=None,
    )

    assert optimum.index == (0, 0, 0)
    assert optimum.valid_mask[0, 0, 0]
    assert optimum.aggregate_score_gain == pytest.approx(1.1)


def test_select_mux_optimum_applies_absolute_flatness_cap_only_when_requested() -> None:
    """An optional absolute cap should reject a relatively acceptable flatness."""
    scores = np.array([[[[2.0, 2.0]]]])
    flatness = np.array([[[[1.5, 1.2]]]])

    without_absolute_cap = _select_mux_optimum(
        scores,
        flatness,
        baseline_scores=[1.0],
        baseline_flatness=[1.5],
        minimum_score_gain=1.0,
        maximum_flatness_ratio=1.1,
        flatness_threshold=None,
    )
    with_absolute_cap = _select_mux_optimum(
        scores,
        flatness,
        baseline_scores=[1.0],
        baseline_flatness=[1.5],
        minimum_score_gain=1.0,
        maximum_flatness_ratio=1.1,
        flatness_threshold=1.2,
    )

    assert without_absolute_cap.index == (0, 0, 0)
    assert with_absolute_cap.index == (0, 0, 1)
    assert without_absolute_cap.valid_mask[0, 0, 0]
    assert not with_absolute_cap.valid_mask[0, 0, 0]


def test_select_mux_optimum_retains_diagnostics_when_every_point_is_invalid() -> None:
    """A failed selection should retain complete relative constraint maps."""
    scores = np.array(
        [
            [[[1.8, 2.4]]],
            [[[4.4, 3.2]]],
        ]
    )
    flatness = np.array(
        [
            [[[1.5, 1.5]]],
            [[[2.4, 2.0]]],
        ]
    )

    with pytest.raises(_NoValidPoint, match="No grid point") as exc_info:
        _select_mux_optimum(
            scores,
            flatness,
            baseline_scores=[2.0, 4.0],
            baseline_flatness=[1.5, 2.0],
            minimum_score_gain=1.0,
            maximum_flatness_ratio=1.1,
            flatness_threshold=None,
        )

    failure = exc_info.value
    np.testing.assert_allclose(
        failure.score_gains,
        np.array(
            [
                [[[0.9, 1.2]]],
                [[[1.1, 0.8]]],
            ]
        ),
    )
    np.testing.assert_allclose(
        failure.flatness_ratios,
        np.array(
            [
                [[[1.0, 1.0]]],
                [[[1.2, 1.0]]],
            ]
        ),
    )
    np.testing.assert_allclose(
        failure.aggregate_score_map,
        np.array([[[1.8, 2.4]]]),
    )
    np.testing.assert_allclose(
        failure.aggregate_score_gain_map,
        np.array([[[0.9, 0.8]]]),
    )
    assert failure.valid_mask.shape == (1, 1, 2)
    assert not np.any(failure.valid_mask)

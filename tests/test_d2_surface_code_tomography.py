"""Tests for d=2 surface-code logical-state tomography."""

from __future__ import annotations

from functools import reduce
from itertools import product
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose

import qubex.contrib.experiment.d2_surface_code_serial_cr as serial_cr
from qubex.contrib.experiment.d2_surface_code_serial_cr import SurfaceCodeConfig
from qubex.contrib.experiment.d2_surface_code_tomography import (
    execute_d2_logical_tomography,
    logical_codewords,
    mitigate_readout_probabilities,
    outcome_probabilities,
    plot_d2_logical_tomography,
    project_logical_density_matrix,
    reconstruct_d2_logical_tomography,
)

_IDENTITY = np.eye(2, dtype=np.complex128)
_PAULIS = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _qubit_map() -> dict[str, str]:
    """Return a deterministic symbolic-to-physical test mapping."""
    return {
        "D1": "Q00",
        "D2": "Q01",
        "D3": "Q02",
        "D4": "Q03",
        "A1": "Q04",
        "A2": "Q05",
        "A3": "Q06",
    }


def _density_matrix(state: np.ndarray) -> np.ndarray:
    """Return the pure-state density matrix for one state vector."""
    return np.outer(state, state.conj())


def _measurement_projector(axis: str, bit: int) -> np.ndarray:
    """Return the positive- or negative-Pauli measurement projector."""
    return (_IDENTITY + (-1) ** bit * _PAULIS[axis]) / 2


def _ideal_tomography_probabilities(
    density_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Generate all 81 exact XYZ-basis probability vectors independently."""
    probabilities: dict[str, np.ndarray] = {}
    for axes in product(("X", "Y", "Z"), repeat=4):
        basis = "".join(axes)
        basis_probabilities = np.empty(16, dtype=np.float64)
        for outcome in range(16):
            bits = (int(bit) for bit in f"{outcome:04b}")
            projector = reduce(
                np.kron,
                (
                    _measurement_projector(axis, bit)
                    for axis, bit in zip(axes, bits, strict=True)
                ),
            )
            basis_probabilities[outcome] = np.trace(density_matrix @ projector).real
        probabilities[basis] = basis_probabilities
    return probabilities


def test_logical_codewords_follow_the_d1_through_d4_convention() -> None:
    """Canonical codewords should use the documented D1,D2,D3,D4 amplitudes."""
    codewords = logical_codewords()
    expected_zero = np.zeros(16, dtype=np.complex128)
    expected_one = np.zeros(16, dtype=np.complex128)
    expected_zero[[0b0000, 0b1111]] = 1 / np.sqrt(2)
    expected_one[[0b0011, 0b1100]] = 1 / np.sqrt(2)

    assert tuple(codewords) == ("0L", "1L", "+L", "-L")
    assert_allclose(codewords["0L"], expected_zero, rtol=0.0, atol=1e-14)
    assert_allclose(codewords["1L"], expected_one, rtol=0.0, atol=1e-14)
    assert_allclose(
        codewords["+L"],
        (expected_zero + expected_one) / np.sqrt(2),
        rtol=0.0,
        atol=1e-14,
    )
    assert_allclose(
        codewords["-L"],
        (expected_zero - expected_one) / np.sqrt(2),
        rtol=0.0,
        atol=1e-14,
    )

    codeword_matrix = np.column_stack((codewords["0L"], codewords["1L"]))
    assert_allclose(
        codeword_matrix.conj().T @ codeword_matrix,
        np.eye(2),
        rtol=0.0,
        atol=1e-14,
    )
    assert_allclose(
        [np.vdot(state, state) for state in codewords.values()],
        np.ones(4),
        rtol=0.0,
        atol=1e-14,
    )


def test_outcome_probabilities_select_shots_and_use_binary_order() -> None:
    """Selected rows should map from D1,D2,D3,D4 bits to outcomes 0 through 15."""
    final_data_bits = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int8,
    )
    accepted_shots = np.array([True, False, True, True, True])
    expected = np.zeros(16)
    expected[[0b0000, 0b1010, 0b1111]] = [0.25, 0.25, 0.5]

    measured = outcome_probabilities(
        final_data_bits,
        accepted_shots=accepted_shots,
    )

    assert_allclose(measured, expected, rtol=0.0, atol=0.0)


def test_readout_mitigation_recovers_distribution_with_inverted_bit() -> None:
    """Four assignment matrices and one software inversion should recover truth."""
    confusion_matrices = np.array(
        [
            [[0.96, 0.04], [0.09, 0.91]],
            [[0.93, 0.07], [0.11, 0.89]],
            [[0.97, 0.03], [0.08, 0.92]],
            [[0.94, 0.06], [0.13, 0.87]],
        ],
    )
    true_probabilities = np.arange(1, 17, dtype=np.float64)
    true_probabilities /= true_probabilities.sum()
    combined_confusion = reduce(np.kron, confusion_matrices)
    inversion_mask = 0b0010
    inverted_columns = np.arange(16) ^ inversion_mask
    observed_probabilities = (
        combined_confusion[:, inverted_columns].T @ true_probabilities
    )

    mitigated = mitigate_readout_probabilities(
        observed_probabilities,
        confusion_matrices,
        readout_inversion=(False, False, True, False),
    )

    assert_allclose(
        mitigated,
        true_probabilities,
        rtol=1e-11,
        atol=1e-12,
    )


def test_ideal_zero_logical_state_reconstructs_physical_and_logical_rho() -> None:
    """Exact 81-basis data for zero-L should reconstruct unit fidelities."""
    zero_l = logical_codewords()["0L"]
    ideal_density = _density_matrix(zero_l)
    probabilities = _ideal_tomography_probabilities(ideal_density)

    result = reconstruct_d2_logical_tomography(
        probabilities,
        target_state="0L",
        mle_fit=False,
    )

    assert_allclose(
        result.physical_density_matrix,
        ideal_density,
        rtol=0.0,
        atol=2e-12,
    )
    assert_allclose(
        result.logical_density_matrix,
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        rtol=0.0,
        atol=2e-12,
    )
    assert result.code_space_probability == pytest.approx(
        1.0,
        rel=0.0,
        abs=2e-12,
    )
    assert result.physical_fidelity == pytest.approx(1.0, rel=0.0, abs=2e-12)
    assert result.logical_fidelity == pytest.approx(1.0, rel=0.0, abs=2e-12)
    assert result.fidelity_product_residual == pytest.approx(
        0.0,
        rel=0.0,
        abs=2e-12,
    )


def test_mle_reconstruction_returns_a_physical_ideal_state() -> None:
    """MLE should return a positive trace-one state with unit ideal fidelity."""
    zero_l = logical_codewords()["0L"]
    probabilities = _ideal_tomography_probabilities(_density_matrix(zero_l))

    result = reconstruct_d2_logical_tomography(
        probabilities,
        target_state="0L",
        mle_fit=True,
    )

    eigenvalues = np.linalg.eigvalsh(result.physical_density_matrix)
    assert np.trace(result.physical_density_matrix).real == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-10,
    )
    assert float(eigenvalues.min()) >= -1e-10
    assert result.physical_fidelity == pytest.approx(1.0, rel=0.0, abs=2e-5)
    assert result.logical_fidelity == pytest.approx(1.0, rel=0.0, abs=2e-5)


def test_logical_projection_separates_code_population_from_fidelity() -> None:
    """Orthogonal leakage should reduce P-L and F-phys but preserve F-L."""
    zero_l = logical_codewords()["0L"]
    leakage_state = np.zeros(16, dtype=np.complex128)
    leakage_state[0b0001] = 1.0
    density = 0.7 * _density_matrix(zero_l) + 0.3 * _density_matrix(leakage_state)

    metrics = project_logical_density_matrix(density, target_state="0L")

    assert metrics.code_space_probability == pytest.approx(
        0.7,
        rel=0.0,
        abs=1e-12,
    )
    assert metrics.physical_fidelity == pytest.approx(
        0.7,
        rel=0.0,
        abs=1e-12,
    )
    assert metrics.logical_fidelity == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-12,
    )
    assert metrics.fidelity_product_residual == pytest.approx(
        0.0,
        rel=0.0,
        abs=1e-12,
    )
    assert_allclose(
        metrics.logical_density_matrix,
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("invalid_kind", ["missing", "invalid"])
def test_reconstruction_rejects_incomplete_or_invalid_basis_sets(
    invalid_kind: str,
) -> None:
    """Reconstruction should require exactly the 81 four-qubit XYZ settings."""
    zero_l = logical_codewords()["0L"]
    probabilities = _ideal_tomography_probabilities(_density_matrix(zero_l))
    if invalid_kind == "missing":
        probabilities.pop("XXXX")
    else:
        probabilities["ABCD"] = probabilities.pop("XXXX")

    with pytest.raises(ValueError, match="exactly all 81"):
        reconstruct_d2_logical_tomography(
            probabilities,
            target_state="0L",
            mle_fit=False,
        )


def test_plot_returns_five_heatmaps_without_showing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plotting with show disabled should return five heatmaps without display."""
    zero_l = logical_codewords()["0L"]
    probabilities = _ideal_tomography_probabilities(_density_matrix(zero_l))
    result = reconstruct_d2_logical_tomography(
        probabilities,
        target_state="0L",
        mle_fit=False,
    )
    shown: list[go.Figure] = []
    monkeypatch.setattr(go.Figure, "show", lambda figure: shown.append(figure))

    figure = plot_d2_logical_tomography(result, show=False)
    traces = cast(tuple[Any, ...], figure.data)

    assert isinstance(figure, go.Figure)
    assert shown == []
    assert len(traces) == 5
    assert all(trace.type == "heatmap" for trace in traces)
    assert [np.asarray(trace.z).shape for trace in traces] == [
        (16, 16),
        (16, 16),
        (16, 16),
        (2, 2),
        (2, 2),
    ]


def test_execute_tomography_uses_all_bases_and_combined_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquisition should run 81 bases and use only herald-and-syndrome shots."""
    ideal_probabilities = _ideal_tomography_probabilities(
        _density_matrix(logical_codewords()["0L"])
    )
    built_axes: list[tuple[str, ...]] = []
    progress: list[tuple[int, int, str]] = []

    def fake_build(_exp: object, config: SurfaceCodeConfig) -> SimpleNamespace:
        built_axes.append(config.resolved_final_measurement_axes)
        return SimpleNamespace(config=config)

    def fake_execute(
        _exp: object,
        built: SimpleNamespace,
        *,
        plot: bool,
    ) -> SimpleNamespace:
        assert plot is False
        basis = "".join(built.config.resolved_final_measurement_axes)
        counts = np.rint(ideal_probabilities[basis] * 16).astype(np.int64)
        accepted_bits = np.array(
            [
                [int(bit) for bit in f"{outcome:04b}"]
                for outcome, count in enumerate(counts)
                for _ in range(count)
            ],
            dtype=np.int8,
        )
        assert accepted_bits.shape == (16, 4)

        final_data_bits = np.ones((20, 4), dtype=np.int8)
        final_data_bits[4:] = accepted_bits
        herald_mask = np.ones(20, dtype=np.bool_)
        herald_mask[:2] = False
        syndrome_mask = np.ones(20, dtype=np.bool_)
        syndrome_mask[2:4] = False
        combined_mask = herald_mask & syndrome_mask
        analysis = SimpleNamespace(
            initial_ground_accepted_shots=herald_mask,
            syndrome_accepted_shots=syndrome_mask,
            accepted_shots=combined_mask,
            final_data_bits=final_data_bits,
        )
        return SimpleNamespace(analysis=analysis, raw_result=object())

    monkeypatch.setattr(serial_cr, "build_d2_surface_code_experiment", fake_build)
    monkeypatch.setattr(serial_cr, "execute_d2_surface_code_experiment", fake_execute)
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        logical_state="0L",
        syndrome_mode="postselect",
        initial_ground_postselection=True,
        n_shots=20,
    )

    result = execute_d2_logical_tomography(
        cast(Any, object()),
        config,
        readout_mitigation=False,
        mle_fit=False,
        plot=False,
        progress_callback=lambda completed, total, basis: progress.append(
            (completed, total, basis)
        ),
    )

    assert len(built_axes) == 81
    assert built_axes[0] == ("X", "X", "X", "X")
    assert built_axes[-1] == ("Z", "Z", "Z", "Z")
    assert len(set(built_axes)) == 81
    assert progress[0] == (1, 81, "XXXX")
    assert progress[-1] == (81, 81, "ZZZZ")
    assert result.total_requested_shots == 81 * 20
    assert result.total_classified_shots == 81 * 20
    assert result.total_herald_accepted_shots == 81 * 18
    assert result.total_accepted_shots == 81 * 16
    assert result.classification_rate == pytest.approx(1.0)
    assert result.herald_acceptance_rate == pytest.approx(0.9)
    assert result.herald_acceptance_given_classified == pytest.approx(0.9)
    assert result.syndrome_acceptance_given_herald == pytest.approx(16 / 18)
    assert result.combined_acceptance_rate == pytest.approx(0.8)
    assert all(
        (
            statistics.classified_shots,
            statistics.herald_accepted_shots,
            statistics.syndrome_accepted_shots,
            statistics.combined_accepted_shots,
        )
        == (20, 18, 18, 16)
        for statistics in result.basis_statistics.values()
    )
    assert result.raw_probabilities is not None
    for basis, expected in ideal_probabilities.items():
        assert_allclose(
            result.raw_probabilities[basis],
            expected,
            rtol=0.0,
            atol=2e-15,
        )
    assert result.raw_probabilities["ZZZZ"][0b1111] == pytest.approx(
        0.5,
        rel=0.0,
        abs=2e-15,
    )

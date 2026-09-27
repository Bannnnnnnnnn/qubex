"""
Reconstruct and visualize logical-state tomography for the d=2 surface code.

The physical data-qubit order is always `D1,D2,D3,D4`.  Tomography therefore
uses the 81 tensor-product measurement settings in `{"X", "Y", "Z"}^4`, with
outcomes ordered from `0000` through `1111`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from qubex.analysis.state_tomography import create_density_matrix

if TYPE_CHECKING:
    from qubex.contrib.experiment.d2_surface_code_serial_cr import SurfaceCodeConfig
    from qubex.experiment import Experiment

LogicalTargetState = Literal["0L", "1L", "+L", "-L"]
TomographyProgressCallback = Callable[[int, int, str], None]

_DATA_QUBITS = ("D1", "D2", "D3", "D4")
_TOMOGRAPHY_BASES = tuple(
    "".join(axes) for axes in product(("X", "Y", "Z"), repeat=len(_DATA_QUBITS))
)
_DIMENSION = 2 ** len(_DATA_QUBITS)
_PROBABILITY_ATOL = 1e-8


@dataclass(frozen=True)
class LogicalProjectionMetrics:
    """
    Hold physical and code-space-projected logical-state metrics.

    Attributes
    ----------
    target_state
        Logical target state used for fidelity calculations.
    logical_density_matrix
        Normalized 2-by-2 density matrix in the `|0L>,|1L>` basis.
    code_space_probability
        Trace of the unnormalized projection into the logical code space.
    physical_fidelity
        Overlap between the physical density matrix and target codeword.
    logical_fidelity
        Target overlap after normalization within the logical code space.
    fidelity_product_residual
        Numerical residual of `F_phys - P_L * F_L`.
    """

    target_state: LogicalTargetState
    logical_density_matrix: NDArray[np.complex128]
    code_space_probability: float
    physical_fidelity: float
    logical_fidelity: float
    fidelity_product_residual: float


@dataclass(frozen=True)
class TomographyBasisStatistics:
    """
    Record shot acceptance for one four-qubit tomography basis.

    Counts for herald and syndrome acceptance are evaluated independently over
    all classified shots.  The combined count is their intersection and is the
    only subset used to estimate the final-data outcome probabilities.
    """

    basis: str
    requested_shots: int
    classified_shots: int
    herald_accepted_shots: int
    syndrome_accepted_shots: int
    combined_accepted_shots: int

    @property
    def combined_acceptance_rate(self) -> float:
        """Return the combined acceptance fraction over requested shots."""
        if self.requested_shots == 0:
            return 0.0
        return self.combined_accepted_shots / self.requested_shots


@dataclass(frozen=True)
class LogicalTomographyResult:
    """
    Hold reconstructed physical and projected logical density matrices.

    Attributes
    ----------
    target_state
        Logical target state used for fidelity calculations.
    probabilities
        Probabilities used for reconstruction, keyed by all 81 basis labels.
    raw_probabilities
        Unmitigated probabilities when collected from hardware, otherwise
        `None`.
    physical_density_matrix
        Reconstructed 16-by-16 density matrix in computational-basis order.
    ideal_physical_density_matrix
        Ideal target density matrix in the same physical basis.
    logical_density_matrix
        Code-space-projected 2-by-2 density matrix.
    ideal_logical_density_matrix
        Ideal target density matrix in the `|0L>,|1L>` basis.
    code_space_probability
        Probability `P_L` assigned to the two-dimensional code space.
    physical_fidelity
        Physical target-state fidelity `F_phys`.
    logical_fidelity
        Projected logical-state fidelity `F_L`.
    fidelity_product_residual
        Numerical residual of `F_phys - P_L * F_L`.
    basis_statistics
        Per-basis herald, syndrome, and combined shot counts.
    mle_fit
        Whether positive-semidefinite maximum-likelihood fitting was requested.
    readout_mitigation_applied
        Whether independent final-data readout correction was applied.
    figure
        Optional Fig. 4-like Plotly visualization.
    """

    target_state: LogicalTargetState
    probabilities: Mapping[str, NDArray[np.float64]]
    raw_probabilities: Mapping[str, NDArray[np.float64]] | None
    physical_density_matrix: NDArray[np.complex128]
    ideal_physical_density_matrix: NDArray[np.complex128]
    logical_density_matrix: NDArray[np.complex128]
    ideal_logical_density_matrix: NDArray[np.complex128]
    code_space_probability: float
    physical_fidelity: float
    logical_fidelity: float
    fidelity_product_residual: float
    basis_statistics: Mapping[str, TomographyBasisStatistics]
    mle_fit: bool
    readout_mitigation_applied: bool
    figure: go.Figure | None = None

    @property
    def total_requested_shots(self) -> int:
        """Return the requested shot count summed over all 81 bases."""
        return sum(item.requested_shots for item in self.basis_statistics.values())

    @property
    def total_accepted_shots(self) -> int:
        """Return the combined accepted-shot count over all 81 bases."""
        return sum(
            item.combined_accepted_shots for item in self.basis_statistics.values()
        )

    @property
    def total_classified_shots(self) -> int:
        """Return shots retained after optional classifier-confidence filtering."""
        return sum(item.classified_shots for item in self.basis_statistics.values())

    @property
    def total_herald_accepted_shots(self) -> int:
        """Return shots classified as all-zero by the initial seven-qubit readout."""
        return sum(
            item.herald_accepted_shots for item in self.basis_statistics.values()
        )

    @property
    def classification_rate(self) -> float:
        """Return classified shots divided by all requested hardware shots."""
        if not self.total_requested_shots:
            return 0.0
        return self.total_classified_shots / self.total_requested_shots

    @property
    def herald_acceptance_rate(self) -> float:
        """Return initial all-zero herald shots divided by all requested shots."""
        if not self.total_requested_shots:
            return 0.0
        return self.total_herald_accepted_shots / self.total_requested_shots

    @property
    def herald_acceptance_given_classified(self) -> float:
        """Return initial all-zero herald acceptance among classified shots."""
        if not self.total_classified_shots:
            return 0.0
        return self.total_herald_accepted_shots / self.total_classified_shots

    @property
    def syndrome_acceptance_given_herald(self) -> float:
        """Return combined herald/syndrome acceptance among heralded shots."""
        if not self.total_herald_accepted_shots:
            return 0.0
        return self.total_accepted_shots / self.total_herald_accepted_shots

    @property
    def combined_acceptance_rate(self) -> float:
        """Return combined herald/syndrome acceptance over requested shots."""
        if not self.total_requested_shots:
            return 0.0
        return self.total_accepted_shots / self.total_requested_shots


def logical_codewords() -> dict[str, NDArray[np.complex128]]:
    """
    Return the four canonical logical codewords in physical basis order.

    The convention is
    `|0L> = (|0000> + |1111>) / sqrt(2)` and
    `|1L> = (|0011> + |1100>) / sqrt(2)`.  The `+L` and `-L` vectors are their
    normalized sum and difference.

    Returns
    -------
    dict[str, NDArray]
        Fresh complex vectors keyed by `0L`, `1L`, `+L`, and `-L`.
    """
    zero = np.zeros(_DIMENSION, dtype=np.complex128)
    one = np.zeros(_DIMENSION, dtype=np.complex128)
    zero[[0b0000, 0b1111]] = 1.0 / np.sqrt(2.0)
    one[[0b0011, 0b1100]] = 1.0 / np.sqrt(2.0)
    return {
        "0L": zero.copy(),
        "1L": one.copy(),
        "+L": (zero + one) / np.sqrt(2.0),
        "-L": (zero - one) / np.sqrt(2.0),
    }


def outcome_probabilities(
    final_data_bits: NDArray[np.integer] | Sequence[Sequence[int]],
    accepted_shots: NDArray[np.bool_] | Sequence[bool] | None = None,
) -> NDArray[np.float64]:
    """
    Convert accepted final-data shots to a 16-outcome probability vector.

    Parameters
    ----------
    final_data_bits
        Binary array with shape `(shots, 4)` in `D1,D2,D3,D4` order.
    accepted_shots
        Optional one-dimensional Boolean mask selecting shots before counting.

    Returns
    -------
    NDArray
        Probability vector ordered from `0000` through `1111`.

    Raises
    ------
    ValueError
        If the input is empty, has the wrong shape, or contains non-binary data.
    """
    bits = np.asarray(final_data_bits)
    if bits.ndim != 2 or bits.shape[1] != len(_DATA_QUBITS):
        raise ValueError("final_data_bits must have shape (shots, 4).")
    if bits.shape[0] == 0:
        raise ValueError("At least one accepted final-data shot is required.")
    if not np.all(np.isin(bits, (0, 1))):
        raise ValueError("final_data_bits must contain only 0 and 1.")
    if accepted_shots is not None:
        mask = np.asarray(accepted_shots)
        if mask.shape != (bits.shape[0],) or mask.dtype.kind != "b":
            raise ValueError(
                "accepted_shots must be a bool vector matching final_data_bits."
            )
        bits = bits[mask]
        if bits.shape[0] == 0:
            raise ValueError("At least one accepted final-data shot is required.")
    indices = bits.astype(np.int64, copy=False) @ np.array([8, 4, 2, 1])
    counts = np.bincount(indices, minlength=_DIMENSION)
    return counts.astype(np.float64) / bits.shape[0]


def _as_probability_vector(
    probabilities: NDArray[np.floating] | Sequence[float],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Return one validated and normalized 16-outcome probability vector."""
    values = np.asarray(probabilities)
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued.")
    values = values.astype(np.float64, copy=True)
    if values.shape != (_DIMENSION,):
        raise ValueError(f"{name} must have shape ({_DIMENSION},).")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(values < -_PROBABILITY_ATOL):
        raise ValueError(f"{name} must not contain negative probabilities.")
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if not np.isclose(total, 1.0, atol=_PROBABILITY_ATOL, rtol=0.0):
        raise ValueError(f"{name} must sum to 1; received {total}.")
    return values / total


def _combined_confusion_matrix(
    confusion_matrices: NDArray[np.floating]
    | Sequence[NDArray[np.floating] | Sequence[Sequence[float]]],
) -> NDArray[np.float64]:
    """Validate and combine either one 16-by-16 or four 2-by-2 matrices."""
    array = np.asarray(confusion_matrices)
    if np.iscomplexobj(array):
        raise ValueError("confusion_matrices must be real-valued.")
    array = array.astype(np.float64, copy=True)
    if array.shape == (_DIMENSION, _DIMENSION):
        combined = array
    elif array.shape == (len(_DATA_QUBITS), 2, 2):
        normalized: list[NDArray[np.float64]] = []
        for index, matrix in enumerate(array):
            if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
                raise ValueError(
                    f"confusion_matrices[{index}] must be finite and non-negative."
                )
            row_sums = matrix.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                raise ValueError(
                    f"confusion_matrices[{index}] has an empty prepared-state row."
                )
            normalized.append(matrix / row_sums)
        combined = normalized[0]
        for matrix in normalized[1:]:
            combined = np.kron(combined, matrix)
    else:
        raise ValueError(
            "confusion_matrices must be one (16, 16) matrix or four (2, 2) "
            "matrices in D1,D2,D3,D4 order."
        )
    if not np.all(np.isfinite(combined)) or np.any(combined < 0):
        raise ValueError("confusion_matrices must be finite and non-negative.")
    row_sums = combined.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("confusion_matrices has an empty prepared-state row.")
    return combined / row_sums


def _project_probability_simplex(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the Euclidean projection of a vector onto the probability simplex."""
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    active = np.flatnonzero(ordered - cumulative / np.arange(1, ordered.size + 1) > 0)
    if active.size == 0:
        raise ValueError("Readout mitigation could not produce probabilities.")
    rho = int(active[-1])
    theta = cumulative[rho] / (rho + 1)
    projected = np.maximum(values - theta, 0.0)
    return projected / projected.sum()


def mitigate_readout_probabilities(
    probabilities: NDArray[np.floating] | Sequence[float],
    confusion_matrices: NDArray[np.floating]
    | Sequence[NDArray[np.floating] | Sequence[Sequence[float]]],
    readout_inversion: tuple[bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
    ),
) -> NDArray[np.float64]:
    """
    Correct independent four-qubit assignment error after shot postselection.

    Confusion-matrix rows represent prepared states and columns represent
    measured states.  If the classified bits were software-inverted before
    forming `probabilities`, the corresponding measured-state columns are
    permuted before inversion.  Negative quasi-probabilities from linear
    inversion are projected onto the probability simplex.

    Parameters
    ----------
    probabilities
        Observed 16-outcome vector ordered from `0000` through `1111`.
    confusion_matrices
        One combined 16-by-16 matrix, or four 2-by-2 matrices in
        `D1,D2,D3,D4` order.
    readout_inversion
        Whether each already-classified output bit was software-inverted.

    Returns
    -------
    NDArray
        Normalized, non-negative, readout-corrected outcome probabilities.

    Raises
    ------
    ValueError
        If an input is malformed or the confusion matrix is singular.
    """
    observed = _as_probability_vector(probabilities, name="probabilities")
    if len(readout_inversion) != len(_DATA_QUBITS) or any(
        not isinstance(value, bool) for value in readout_inversion
    ):
        raise ValueError("readout_inversion must contain exactly four bool values.")
    confusion = _combined_confusion_matrix(confusion_matrices)
    inversion_mask = sum(
        (1 << (len(_DATA_QUBITS) - index - 1))
        for index, invert in enumerate(readout_inversion)
        if invert
    )
    corrected_column_order = np.arange(_DIMENSION) ^ inversion_mask
    effective_confusion = confusion[:, corrected_column_order]
    try:
        mitigated = np.linalg.solve(effective_confusion.T, observed)
    except np.linalg.LinAlgError as exc:
        raise ValueError("confusion_matrices must form an invertible matrix.") from exc
    if not np.all(np.isfinite(mitigated)):
        raise ValueError("Readout mitigation produced non-finite values.")
    return _project_probability_simplex(mitigated)


def _target_state(
    target_state: str,
) -> tuple[
    LogicalTargetState,
    NDArray[np.complex128],
    NDArray[np.complex128],
]:
    """Return validated logical and physical target vectors."""
    codewords = logical_codewords()
    if target_state not in codewords:
        raise ValueError("target_state must be one of 0L, 1L, +L, or -L.")
    logical_vectors: dict[str, NDArray[np.complex128]] = {
        "0L": np.array([1.0, 0.0], dtype=np.complex128),
        "1L": np.array([0.0, 1.0], dtype=np.complex128),
        "+L": np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
        "-L": np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0),
    }
    name = cast(LogicalTargetState, target_state)
    return name, codewords[name], logical_vectors[name]


def _validated_density_matrix(
    rho: NDArray[Any],
) -> NDArray[np.complex128]:
    """Return a validated, explicitly Hermitian 16-by-16 density matrix."""
    density = np.asarray(rho, dtype=np.complex128)
    if density.shape != (_DIMENSION, _DIMENSION):
        raise ValueError(f"rho must have shape ({_DIMENSION}, {_DIMENSION}).")
    if not np.all(np.isfinite(density)):
        raise ValueError("rho must contain only finite values.")
    if not np.allclose(density, density.conj().T, atol=1e-8, rtol=0.0):
        raise ValueError("rho must be Hermitian.")
    density = (density + density.conj().T) / 2.0
    trace = np.trace(density)
    if not np.isclose(trace.imag, 0.0, atol=1e-8, rtol=0.0):
        raise ValueError("rho must have a real trace.")
    if not np.isclose(trace.real, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(f"rho must have trace 1; received {trace.real}.")
    return density / trace.real


def project_logical_density_matrix(
    rho: NDArray[Any],
    target_state: str,
) -> LogicalProjectionMetrics:
    """
    Project a physical four-data-qubit density matrix into the code space.

    Parameters
    ----------
    rho
        Hermitian, trace-one 16-by-16 physical density matrix.
    target_state
        One of `0L`, `1L`, `+L`, or `-L`.

    Returns
    -------
    LogicalProjectionMetrics
        Projected logical density matrix, code-space probability, and
        fidelities.

    Raises
    ------
    ValueError
        If the matrix is invalid or has no positive code-space population.
    """
    density = _validated_density_matrix(rho)
    name, physical_target, logical_target = _target_state(target_state)
    codewords = logical_codewords()
    isometry = np.column_stack((codewords["0L"], codewords["1L"]))
    projected = isometry.conj().T @ density @ isometry
    code_space_probability = float(np.trace(projected).real)
    if code_space_probability <= _PROBABILITY_ATOL:
        raise ValueError("rho has no positive population in the logical code space.")
    logical_density = projected / code_space_probability
    logical_density = (logical_density + logical_density.conj().T) / 2.0
    physical_fidelity = float(
        np.real(physical_target.conj() @ density @ physical_target)
    )
    logical_fidelity = float(
        np.real(logical_target.conj() @ logical_density @ logical_target)
    )
    residual = physical_fidelity - code_space_probability * logical_fidelity
    return LogicalProjectionMetrics(
        target_state=name,
        logical_density_matrix=logical_density,
        code_space_probability=code_space_probability,
        physical_fidelity=physical_fidelity,
        logical_fidelity=logical_fidelity,
        fidelity_product_residual=float(residual),
    )


def _validated_probability_mapping(
    probabilities: Mapping[str, NDArray[np.floating] | Sequence[float]],
) -> dict[str, NDArray[np.float64]]:
    """Return all 81 tomography vectors in deterministic basis order."""
    supplied = set(probabilities)
    expected = set(_TOMOGRAPHY_BASES)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            "probabilities must contain exactly all 81 XYZ basis settings; "
            f"missing={missing}, extra={extra}."
        )
    return {
        basis: _as_probability_vector(
            probabilities[basis],
            name=f"probabilities[{basis!r}]",
        )
        for basis in _TOMOGRAPHY_BASES
    }


def _validated_basis_statistics(
    basis_statistics: Mapping[str, TomographyBasisStatistics] | None,
) -> dict[str, TomographyBasisStatistics]:
    """Validate optional per-basis shot statistics."""
    if basis_statistics is None:
        return {}
    supplied = set(basis_statistics)
    expected = set(_TOMOGRAPHY_BASES)
    if supplied != expected:
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        raise ValueError(
            "basis_statistics must contain exactly all 81 basis settings; "
            f"missing={missing}, extra={extra}."
        )
    validated: dict[str, TomographyBasisStatistics] = {}
    for basis in _TOMOGRAPHY_BASES:
        item = basis_statistics[basis]
        if not isinstance(item, TomographyBasisStatistics):
            raise TypeError(
                "basis_statistics values must be TomographyBasisStatistics."
            )
        if item.basis != basis:
            raise ValueError(f"basis_statistics[{basis!r}].basis must equal {basis!r}.")
        counts = (
            item.requested_shots,
            item.classified_shots,
            item.herald_accepted_shots,
            item.syndrome_accepted_shots,
            item.combined_accepted_shots,
        )
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in counts
        ):
            raise TypeError("Shot statistics counts must be integers.")
        if any(value < 0 for value in counts):
            raise ValueError("Shot statistics counts must be non-negative.")
        if item.classified_shots > item.requested_shots:
            raise ValueError("classified_shots cannot exceed requested_shots.")
        if max(counts[2:]) > item.classified_shots:
            raise ValueError("Acceptance counts cannot exceed classified_shots.")
        if item.combined_accepted_shots > min(counts[2], counts[3]):
            raise ValueError(
                "combined_accepted_shots cannot exceed either individual count."
            )
        validated[basis] = item
    return validated


def reconstruct_d2_logical_tomography(
    probabilities: Mapping[str, NDArray[np.floating] | Sequence[float]],
    target_state: str = "0L",
    mle_fit: bool = True,
    basis_statistics: Mapping[str, TomographyBasisStatistics] | None = None,
) -> LogicalTomographyResult:
    """
    Reconstruct physical and projected logical density matrices from 81 bases.

    Parameters
    ----------
    probabilities
        Mapping from every four-character XYZ basis label to a 16-outcome
        probability vector in `0000` through `1111` order.
    target_state
        Logical fidelity target: `0L`, `1L`, `+L`, or `-L`.
    mle_fit
        Whether to use Qubex's positive-semidefinite CVXPY fit.  If `False`,
        Qubex linear inversion is used.
    basis_statistics
        Optional per-basis shot accounting.

    Returns
    -------
    LogicalTomographyResult
        Physical density matrix, logical projection, fidelities, and metadata.
    """
    if not isinstance(mle_fit, bool):
        raise TypeError("mle_fit must be bool.")
    normalized = _validated_probability_mapping(probabilities)
    statistics = _validated_basis_statistics(basis_statistics)
    density = np.asarray(
        create_density_matrix(normalized, mle_fit=mle_fit),
        dtype=np.complex128,
    )
    metrics = project_logical_density_matrix(density, target_state)
    name, physical_target, logical_target = _target_state(target_state)
    return LogicalTomographyResult(
        target_state=name,
        probabilities=normalized,
        raw_probabilities=None,
        physical_density_matrix=density,
        ideal_physical_density_matrix=np.outer(physical_target, physical_target.conj()),
        logical_density_matrix=metrics.logical_density_matrix,
        ideal_logical_density_matrix=np.outer(logical_target, logical_target.conj()),
        code_space_probability=metrics.code_space_probability,
        physical_fidelity=metrics.physical_fidelity,
        logical_fidelity=metrics.logical_fidelity,
        fidelity_product_residual=metrics.fidelity_product_residual,
        basis_statistics=statistics,
        mle_fit=mle_fit,
        readout_mitigation_applied=False,
    )


def _add_density_heatmap(
    figure: go.Figure,
    matrix: NDArray[np.complex128],
    *,
    component: Literal["real", "imag"],
    row: int,
    col: int,
    limit: float,
) -> None:
    """Add one density-matrix component to a subplot."""
    values = matrix.real if component == "real" else matrix.imag
    figure.add_trace(
        go.Heatmap(
            z=values,
            zmin=-limit,
            zmax=limit,
            zmid=0.0,
            colorscale="RdBu_r",
            showscale=False,
            hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.5f}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def plot_d2_logical_tomography(
    result: LogicalTomographyResult,
    show: bool = False,
) -> go.Figure:
    """
    Create a Fig. 4-like physical and projected logical density-matrix figure.

    Parameters
    ----------
    result
        Reconstructed d=2 tomography result.
    show
        Whether to display the figure immediately, including in Jupyter.

    Returns
    -------
    plotly.graph_objects.Figure
        Six-panel figure containing physical, ideal, logical, and metric views.
    """
    if not isinstance(result, LogicalTomographyResult):
        raise TypeError("result must be a LogicalTomographyResult.")
    if not isinstance(show, bool):
        raise TypeError("show must be bool.")
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Re(physical ρ)",
            "Im(physical ρ)",
            f"Ideal physical |{result.target_state}⟩",
            "Re(projected logical ρL)",
            "Im(projected logical ρL)",
            "Logical-state metrics",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.17,
    )
    _add_density_heatmap(
        figure,
        result.physical_density_matrix,
        component="real",
        row=1,
        col=1,
        limit=0.5,
    )
    _add_density_heatmap(
        figure,
        result.physical_density_matrix,
        component="imag",
        row=1,
        col=2,
        limit=0.5,
    )
    _add_density_heatmap(
        figure,
        result.ideal_physical_density_matrix,
        component="real",
        row=1,
        col=3,
        limit=0.5,
    )
    _add_density_heatmap(
        figure,
        result.logical_density_matrix,
        component="real",
        row=2,
        col=1,
        limit=1.0,
    )
    _add_density_heatmap(
        figure,
        result.logical_density_matrix,
        component="imag",
        row=2,
        col=2,
        limit=1.0,
    )

    physical_ticks = np.arange(_DIMENSION)
    physical_labels = [f"{value:04b}" for value in physical_ticks]
    for col in (1, 2, 3):
        figure.update_xaxes(
            tickmode="array",
            tickvals=physical_ticks,
            ticktext=physical_labels,
            tickangle=-90,
            row=1,
            col=col,
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=physical_ticks,
            ticktext=physical_labels,
            autorange="reversed",
            scaleanchor=f"x{col}" if col > 1 else "x",
            row=1,
            col=col,
        )
    logical_ticks = [0, 1]
    logical_labels = ["0L", "1L"]
    for col in (1, 2):
        figure.update_xaxes(
            tickmode="array",
            tickvals=logical_ticks,
            ticktext=logical_labels,
            row=2,
            col=col,
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=logical_ticks,
            ticktext=logical_labels,
            autorange="reversed",
            row=2,
            col=col,
        )
    figure.update_xaxes(visible=False, row=2, col=3)
    figure.update_yaxes(visible=False, row=2, col=3)
    acceptance_lines = "Shot statistics unavailable"
    if result.basis_statistics:
        acceptance_lines = (
            f"classified = {result.classification_rate:.2%}<br>"
            f"initial all-zero = {result.herald_acceptance_rate:.2%} "
            f"({result.herald_acceptance_given_classified:.2%} of classified)<br>"
            f"syndrome | initial = "
            f"{result.syndrome_acceptance_given_herald:.2%}<br>"
            f"combined = {result.total_accepted_shots:,} / "
            f"{result.total_requested_shots:,} "
            f"({result.combined_acceptance_rate:.2%})"
        )
    figure.add_annotation(
        x=0.84,
        y=0.24,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Target |{result.target_state}⟩</b><br>"
            f"P<sub>L</sub> = {result.code_space_probability:.4f}<br>"
            f"F<sub>phys</sub> = {result.physical_fidelity:.4f}<br>"
            f"F<sub>L</sub> = {result.logical_fidelity:.4f}<br>"
            f"{acceptance_lines}<br>"
            f"MLE = {result.mle_fit}<br>"
            f"readout mitigation = {result.readout_mitigation_applied}"
        ),
        align="left",
        showarrow=False,
        font={"size": 15},
    )
    figure.update_layout(
        title=(
            f"d=2 surface-code logical |{result.target_state}⟩ tomography — "
            f"Fphys={result.physical_fidelity:.2%}, "
            f"FL={result.logical_fidelity:.2%}, "
            f"PL={result.code_space_probability:.2%}"
        ),
        width=1_520,
        height=950,
        margin={"l": 70, "r": 40, "t": 105, "b": 90},
    )
    if show:
        figure.show()
    return figure


def _analysis_mask(analysis: object, name: str) -> NDArray[np.bool_]:
    """Return one validated shot-selection mask from surface-code analysis."""
    if not hasattr(analysis, name):
        raise RuntimeError(
            f"SurfaceCodeAnalysis is missing the required {name!r} mask."
        )
    mask = np.asarray(getattr(analysis, name))
    if mask.ndim != 1 or mask.dtype.kind != "b":
        raise RuntimeError(f"SurfaceCodeAnalysis.{name} must be a 1-D bool array.")
    return mask.astype(np.bool_, copy=False)


def execute_d2_logical_tomography(
    exp: Experiment,
    config: SurfaceCodeConfig,
    *,
    readout_mitigation: bool = True,
    mle_fit: bool = True,
    plot: bool = True,
    progress_callback: TomographyProgressCallback | None = None,
) -> LogicalTomographyResult:
    """
    Acquire all 81 bases and reconstruct one d=2 logical-state density matrix.

    Each basis executes a complete surface-code sequence, including the initial
    all-zero herald readout.  Only the intersection of the initial-ground and
    configured syndrome masks is used for final-data probabilities.  Final data
    bits themselves are never postselected.

    Parameters
    ----------
    exp
        Connected and calibrated Qubex experiment.
    config
        d=2 configuration with `syndrome_mode="postselect"` and
        `initial_ground_postselection=True`.
    readout_mitigation
        Whether to correct independent final-data assignment error.
    mle_fit
        Whether to use Qubex's positive-semidefinite CVXPY fit.
    plot
        Whether to display the resulting Plotly figure in the active frontend.
    progress_callback
        Optional callback invoked after every basis as
        `(completed_basis_count, 81, basis_label)`.

    Returns
    -------
    LogicalTomographyResult
        Reconstructed matrices, fidelities, acceptance statistics, and figure.

    Raises
    ------
    ValueError
        If required postselection is disabled, no shots survive a basis, or
        configuration values are incompatible with logical tomography.

    Notes
    -----
    This function executes hardware 81 times.  It does not alter `config`; each
    basis is built from an immutable dataclass replacement.
    """
    if config.syndrome_mode != "postselect":
        raise ValueError(
            "Logical tomography requires syndrome_mode='postselect' so code-space "
            "preparation uses the configured syndrome outcome."
        )
    if not config.initial_ground_postselection:
        raise ValueError(
            "Logical tomography requires initial_ground_postselection=True."
        )
    if config.logical_state not in ("0L", "1L", "+L", "-L"):
        raise ValueError(
            "Logical tomography requires logical_state to be 0L, 1L, +L, or -L."
        )
    for name, value in (
        ("readout_mitigation", readout_mitigation),
        ("mle_fit", mle_fit),
        ("plot", plot),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")

    from qubex.contrib.experiment.d2_surface_code_serial_cr import (
        build_d2_surface_code_experiment,
        execute_d2_surface_code_experiment,
    )

    raw_probabilities: dict[str, NDArray[np.float64]] = {}
    reconstruction_probabilities: dict[str, NDArray[np.float64]] = {}
    statistics: dict[str, TomographyBasisStatistics] = {}
    data_labels = [config.qubit_map[symbol] for symbol in _DATA_QUBITS]
    readout_inversion = cast(
        tuple[bool, bool, bool, bool],
        tuple(
            bool(config.data_readout_inversion.get(symbol, False))
            for symbol in _DATA_QUBITS
        ),
    )

    for completed, basis in enumerate(_TOMOGRAPHY_BASES, start=1):
        basis_config = replace(
            config,
            final_measurement_axes=cast(
                tuple[
                    Literal["X", "Y", "Z"],
                    Literal["X", "Y", "Z"],
                    Literal["X", "Y", "Z"],
                    Literal["X", "Y", "Z"],
                ],
                tuple(basis),
            ),
        )
        built = build_d2_surface_code_experiment(exp, basis_config)
        executed = execute_d2_surface_code_experiment(exp, built, plot=False)
        analysis = executed.analysis
        herald_mask = _analysis_mask(
            analysis,
            "initial_ground_accepted_shots",
        )
        syndrome_mask = _analysis_mask(analysis, "syndrome_accepted_shots")
        combined_mask = _analysis_mask(analysis, "accepted_shots")
        if not (herald_mask.shape == syndrome_mask.shape == combined_mask.shape):
            raise RuntimeError("Surface-code acceptance masks have different shapes.")
        expected_combined = herald_mask & syndrome_mask
        if not np.array_equal(combined_mask, expected_combined):
            raise RuntimeError(
                "accepted_shots must equal the intersection of initial-ground "
                "and syndrome acceptance during tomography."
            )
        final_data_bits = np.asarray(analysis.final_data_bits)
        if final_data_bits.shape != (combined_mask.size, len(_DATA_QUBITS)):
            raise RuntimeError(
                "SurfaceCodeAnalysis.final_data_bits has an unexpected shape."
            )
        if not np.any(combined_mask):
            raise ValueError(
                f"No shots survived initial-ground and syndrome postselection "
                f"for basis {basis}."
            )
        observed = outcome_probabilities(
            final_data_bits,
            accepted_shots=combined_mask,
        )
        raw_probabilities[basis] = observed
        if readout_mitigation:
            try:
                confusion = executed.raw_result.get_confusion_matrix(data_labels)
                corrected = mitigate_readout_probabilities(
                    observed,
                    confusion,
                    readout_inversion,
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Could not apply final-data readout mitigation for basis "
                    f"{basis}. Ensure classifiers with non-singular confusion "
                    "matrices are loaded for D1-D4."
                ) from exc
            reconstruction_probabilities[basis] = corrected
        else:
            reconstruction_probabilities[basis] = observed.copy()

        classified_shots = int(combined_mask.size)
        statistics[basis] = TomographyBasisStatistics(
            basis=basis,
            requested_shots=config.n_shots,
            classified_shots=classified_shots,
            herald_accepted_shots=int(np.count_nonzero(herald_mask)),
            syndrome_accepted_shots=int(np.count_nonzero(syndrome_mask)),
            combined_accepted_shots=int(np.count_nonzero(combined_mask)),
        )
        if progress_callback is not None:
            progress_callback(completed, len(_TOMOGRAPHY_BASES), basis)

    result = reconstruct_d2_logical_tomography(
        reconstruction_probabilities,
        target_state=config.logical_state,
        mle_fit=mle_fit,
        basis_statistics=statistics,
    )
    result = replace(
        result,
        raw_probabilities=raw_probabilities,
        readout_mitigation_applied=readout_mitigation,
    )
    figure = plot_d2_logical_tomography(result, show=plot)
    return replace(result, figure=figure)


__all__ = [
    "LogicalProjectionMetrics",
    "LogicalTargetState",
    "LogicalTomographyResult",
    "TomographyBasisStatistics",
    "TomographyProgressCallback",
    "execute_d2_logical_tomography",
    "logical_codewords",
    "mitigate_readout_probabilities",
    "outcome_probabilities",
    "plot_d2_logical_tomography",
    "project_logical_density_matrix",
    "reconstruct_d2_logical_tomography",
]

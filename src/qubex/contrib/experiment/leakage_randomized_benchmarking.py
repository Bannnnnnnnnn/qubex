"""Contributed leakage randomized benchmarking helpers."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Collection
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray

import qubex.visualization as viz
from qubex.analysis import FitResult, FitStatus
from qubex.clifford import Clifford
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_MAX_N_CLIFFORDS_2Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.pulse import Blank, FlatTop, PulseSchedule, RampType, Waveform
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options
from .stark_characterization import (
    StarkDriveQubit,
    _resolve_stark_ramptime,
    _stark_drive_amplitude,
    stark_cr_target,
    stark_rb_sequence_1q,
    stark_rb_sequence_2q,
    stark_target,
)

__all__ = [
    "fit_computational_leakage_rb",
    "fit_leakage_rb",
    "fit_stark_pulse_leakage",
    "interleaved_leakage_rb_experiment_1q",
    "interleaved_leakage_rb_experiment_2q",
    "leakage_randomized_benchmarking",
    "leakage_rb_experiment_1q",
    "leakage_rb_experiment_2q",
    "stark_interleaved_leakage_rb_experiment_1q",
    "stark_interleaved_leakage_rb_experiment_2q",
    "stark_leakage_randomized_benchmarking",
    "stark_leakage_rb_experiment_1q",
    "stark_leakage_rb_experiment_2q",
    "stark_pair_pulse_leakage_experiment",
    "stark_pulse_leakage_experiment",
]

logger = logging.getLogger(__name__)

COMPUTATIONAL_STATE_LABELS = ("0", "1")
LEAKAGE_FIT_MODEL = "leakage_rate_equation"
LEAKAGE_FIT_ASSUMPTIONS = (
    "single leakage subspace, Markovian gate-independent exchange, leakage "
    "population follows p_inf + (p0 - p_inf) * exp(-Gamma * m)"
)
COMPUTATIONAL_LEAKAGE_FIT_MODEL = "computational_subspace_decay"
COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS = (
    "paired SRB/IRB leakage estimate, computational population follows "
    "C + D * lambda_l**m and L = (1 - C) * (1 - lambda_l)"
)


def _curve_fit(*args: Any, **kwargs: Any) -> tuple[NDArray, NDArray]:
    from scipy.optimize import curve_fit as scipy_curve_fit

    return scipy_curve_fit(*args, **kwargs)


def _normalize_targets(targets: Collection[str] | str) -> list[str]:
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _normalize_int_sweep(values: ArrayLike, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if np.any(array < 0):
        raise ValueError(f"`{name}` must contain non-negative Clifford counts.")
    return array


def _normalize_float_series(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D array.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite values.")
    return array


def _auto_n_cliffords_range(max_n_cliffords: int) -> NDArray[np.int64]:
    if max_n_cliffords < 0:
        raise ValueError("`max_n_cliffords` must be non-negative.")

    values: list[int] = []
    idx = 0
    while True:
        n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
        if n_clifford > max_n_cliffords:
            break
        values.append(n_clifford)
        idx += 1
    return np.asarray(values, dtype=np.int64)


def _resolve_clifford(
    exp: Experiment,
    clifford: str | Clifford | None,
) -> Clifford | None:
    if clifford is None:
        return None
    if isinstance(clifford, Clifford):
        return clifford
    resolved = exp.benchmarking_service.clifford.get(clifford)
    if resolved is None:
        raise ValueError(f"Invalid Clifford: {clifford}")
    return resolved


def _require_1q_targets(exp: Experiment, targets: Collection[str]) -> None:
    for target in targets:
        target_object = exp.ctx.experiment_system.get_target(target)
        if target_object.is_cr:
            raise ValueError(f"`{target}` is not a 1Q target.")


def _require_2q_targets(
    exp: Experiment,
    targets: Collection[str],
) -> dict[str, tuple[str, str]]:
    cr_pairs: dict[str, tuple[str, str]] = {}
    for target in targets:
        target_object = exp.ctx.experiment_system.get_target(target)
        if not target_object.is_cr:
            raise ValueError(f"`{target}` is not a 2Q target.")
        cr_pairs[target] = exp.ctx.cr_pair(target)
    return cr_pairs


def _require_3_state_classifiers(exp: Experiment, targets: Collection[str]) -> None:
    for target in targets:
        classifier = exp.classifiers.get(target)
        if classifier is None:
            raise ValueError(
                f"State classifier for `{target}` is not built. "
                "Build a 3-state classifier before leakage RB."
            )
        if classifier.n_states != 3:
            raise ValueError(
                f"State classifier for `{target}` has {classifier.n_states} states; "
                "leakage RB requires a 3-state classifier."
            )


def _basis_labels(n_qubits: int) -> tuple[str, ...]:
    if n_qubits <= 0:
        raise ValueError("`n_qubits` must be positive.")
    return tuple(
        "".join(str(state) for state in basis)
        for basis in np.ndindex(*([3] * n_qubits))
    )


def _state_probabilities_from_result(
    result: Any,
    targets: Collection[str],
    *,
    state_labels: Collection[str],
    mitigate_readout: bool,
) -> NDArray[np.float64]:
    if mitigate_readout:
        probabilities = result.get_mitigated_probabilities(targets)
    else:
        probabilities = result.get_probabilities(targets)

    return np.asarray(
        [probabilities.get(label, 0.0) for label in state_labels],
        dtype=np.float64,
    )


def _summarize_population_trials(
    trials: NDArray[np.float64],
    state_labels: Collection[str],
) -> dict[str, NDArray[np.float64]]:
    labels = tuple(state_labels)
    computational_indices = [
        index
        for index, label in enumerate(labels)
        if all(state in COMPUTATIONAL_STATE_LABELS for state in label)
    ]
    leakage_indices = [
        index
        for index, label in enumerate(labels)
        if index not in computational_indices
    ]

    computational_trials = np.sum(trials[:, :, computational_indices], axis=2)
    leakage_trials = np.sum(trials[:, :, leakage_indices], axis=2)
    return {
        "state_population_mean": np.mean(trials, axis=1),
        "state_population_std": np.std(trials, axis=1),
        "computational_population_trials": computational_trials,
        "computational_population_mean": np.mean(computational_trials, axis=1),
        "computational_population_std": np.std(computational_trials, axis=1),
        "leakage_population_trials": leakage_trials,
        "leakage_population_mean": np.mean(leakage_trials, axis=1),
        "leakage_population_std": np.std(leakage_trials, axis=1),
    }


def _leakage_rate_equation(
    n_cliffords: ArrayLike,
    p0: float,
    gamma: float,
    p_inf: float,
) -> NDArray[np.float64]:
    n = np.asarray(n_cliffords, dtype=np.float64)
    return p_inf + (p0 - p_inf) * np.exp(-gamma * n)


def _computational_subspace_decay_equation(
    n_cliffords: ArrayLike,
    c: float,
    amplitude: float,
    lambda_l: float,
) -> NDArray[np.float64]:
    n = np.asarray(n_cliffords, dtype=np.float64)
    return c + amplitude * np.power(lambda_l, n)


def _parameter_errors(pcov: NDArray[np.float64]) -> NDArray[np.float64]:
    if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return np.full(3, np.nan, dtype=np.float64)
    diagonal = np.diag(pcov)
    if np.any(diagonal < 0):
        return np.full(3, np.nan, dtype=np.float64)
    return np.sqrt(diagonal)


def _derived_rate_errors(
    *,
    gamma: float,
    p_inf: float,
    pcov: NDArray[np.float64],
) -> tuple[float, float, float]:
    if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return np.nan, np.nan, np.nan

    rate_cov = pcov[1:3, 1:3]
    gamma_err = float(np.sqrt(max(rate_cov[0, 0], 0.0)))

    gamma_up_gradient = np.asarray([p_inf, gamma])
    gamma_down_gradient = np.asarray([1.0 - p_inf, -gamma])
    gamma_up_var = float(gamma_up_gradient @ rate_cov @ gamma_up_gradient)
    gamma_down_var = float(gamma_down_gradient @ rate_cov @ gamma_down_gradient)
    gamma_up_err = float(np.sqrt(max(gamma_up_var, 0.0)))
    gamma_down_err = float(np.sqrt(max(gamma_down_var, 0.0)))
    return gamma_err, gamma_up_err, gamma_down_err


def _computational_leakage_rate(
    *,
    c: float,
    lambda_l: float,
    pcov: NDArray[np.float64],
) -> tuple[float, float]:
    leakage_rate = (1.0 - c) * (1.0 - lambda_l)
    if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return float(leakage_rate), np.nan

    gradient = np.asarray(
        [
            -(1.0 - lambda_l),
            0.0,
            -(1.0 - c),
        ]
    )
    variance = float(gradient @ pcov @ gradient)
    return float(leakage_rate), float(np.sqrt(max(variance, 0.0)))


def _interleaved_gate_leakage_rate(
    *,
    reference_leakage_rate: float,
    interleaved_leakage_rate: float,
    reference_leakage_rate_err: float,
    interleaved_leakage_rate_err: float,
) -> tuple[float, float]:
    denominator = 1.0 - reference_leakage_rate
    if abs(denominator) <= np.finfo(float).eps:
        return np.nan, np.nan

    gate_leakage_rate = 1.0 - (1.0 - interleaved_leakage_rate) / denominator
    reference_gradient = -(1.0 - interleaved_leakage_rate) / denominator**2
    interleaved_gradient = 1.0 / denominator

    error_terms = []
    if np.isfinite(reference_leakage_rate_err):
        error_terms.append((reference_gradient * reference_leakage_rate_err) ** 2)
    if np.isfinite(interleaved_leakage_rate_err):
        error_terms.append((interleaved_gradient * interleaved_leakage_rate_err) ** 2)

    if not error_terms:
        return float(gate_leakage_rate), np.nan
    return float(gate_leakage_rate), float(np.sqrt(sum(error_terms)))


def _make_leakage_rb_figure(
    *,
    target: str,
    x: NDArray[np.int64],
    leakage_population: NDArray[np.float64],
    error_y: NDArray[np.float64] | None,
    x_fit: NDArray[np.float64] | None,
    leakage_fit: NDArray[np.float64] | None,
    title: str,
    xlabel: str,
    ylabel: str,
    xaxis_type: Literal["linear", "log"],
) -> go.Figure:
    fig = viz.make_figure()

    if x_fit is not None and leakage_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=leakage_fit,
                mode="lines",
                name="Leakage fit",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=leakage_population,
            error_y=dict(type="data", array=error_y) if error_y is not None else None,
            mode="markers",
            name="Leakage population",
        )
    )

    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type="linear",
    )
    return fig


def _make_computational_leakage_rb_figure(
    *,
    target: str,
    x: NDArray[np.int64],
    computational_population: NDArray[np.float64],
    error_y: NDArray[np.float64] | None,
    x_fit: NDArray[np.float64] | None,
    computational_fit: NDArray[np.float64] | None,
    title: str,
    xlabel: str,
    ylabel: str,
    xaxis_type: Literal["linear", "log"],
    trace_name: str,
) -> go.Figure:
    fig = viz.make_figure()

    if x_fit is not None and computational_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=computational_fit,
                mode="lines",
                name=f"{trace_name} fit",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=computational_population,
            error_y=dict(type="data", array=error_y) if error_y is not None else None,
            mode="markers",
            name=f"{trace_name} computational population",
        )
    )

    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type="linear",
    )
    return fig


def _make_interleaved_leakage_rb_figure(
    *,
    target: str,
    x: NDArray[np.int64],
    reference_population: NDArray[np.float64],
    reference_error_y: NDArray[np.float64] | None,
    reference_fit: FitResult,
    interleaved_population: NDArray[np.float64],
    interleaved_error_y: NDArray[np.float64] | None,
    interleaved_fit: FitResult,
    gate_leakage_rate: float,
    title: str,
    xlabel: str,
    ylabel: str,
    xaxis_type: Literal["linear", "log"],
) -> go.Figure:
    fig = viz.make_figure()

    for trace_name, fit_result in (
        ("SRB", reference_fit),
        ("IRB", interleaved_fit),
    ):
        x_fit = fit_result.data.get("x_fit")
        population_fit = fit_result.data.get("computational_population_fit")
        if x_fit is None or population_fit is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=population_fit,
                mode="lines",
                name=f"{trace_name} fit",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=reference_population,
            error_y=(
                dict(type="data", array=reference_error_y)
                if reference_error_y is not None
                else None
            ),
            mode="markers",
            name="SRB computational population",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=interleaved_population,
            error_y=(
                dict(type="data", array=interleaved_error_y)
                if interleaved_error_y is not None
                else None
            ),
            mode="markers",
            name="IRB computational population",
        )
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"L_gate = {gate_leakage_rate:.4g}",
        showarrow=False,
    )
    fig.update_layout(
        title=f"{title} : {target}",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        xaxis_type=xaxis_type,
        yaxis_type="linear",
    )
    return fig


def fit_computational_leakage_rb(
    *,
    target: str,
    x: ArrayLike,
    computational_population: ArrayLike,
    error_y: ArrayLike | None = None,
    p0: tuple[float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    plot: bool = True,
    title: str = "Leakage randomized benchmarking",
    xlabel: str = "Number of Cliffords",
    ylabel: str = "Computational population",
    xaxis_type: Literal["linear", "log"] = "linear",
    trace_name: str = "RB",
) -> FitResult:
    """
    Fit computational-subspace decay for SRB/IRB leakage estimates.

    The fitted model follows the SRB/IRB leakage estimate used in
    arXiv:2511.01260:
    ``P_comp(m) = C + D * lambda_l**m`` and
    ``L = (1 - C) * (1 - lambda_l)``.

    Returns
    -------
    FitResult
        Fit payload containing ``leakage_rate`` for the supplied RB protocol.
    """
    x_array = _normalize_int_sweep(x, name="x")
    comp = _normalize_float_series(
        computational_population,
        name="computational_population",
    )
    if len(x_array) != len(comp):
        raise ValueError("`x` and `computational_population` must have same length.")

    error = None
    if error_y is not None:
        error = _normalize_float_series(error_y, name="error_y")
        if len(error) != len(x_array):
            raise ValueError("`error_y` must have same length as `x`.")

    if p0 is None:
        c_guess = float(np.clip(comp[-1], 0.0, 1.0))
        amplitude_guess = float(np.clip(comp[0] - c_guess, -1.0, 1.0))
        p0 = (c_guess, amplitude_guess, 0.99)

    if bounds is None:
        bounds = ((0.0, -1.0, 0.0), (1.0, 1.0, 1.0 - 1e-12))

    try:
        popt, pcov = _curve_fit(
            _computational_subspace_decay_equation,
            x_array,
            comp,
            p0=p0,
            bounds=bounds,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "Failed to fit computational leakage RB data for %s: %s",
            target,
            exc,
        )
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit computational leakage RB data.",
        )

    c, amplitude, lambda_l = [float(value) for value in popt]
    c_err, amplitude_err, lambda_l_err = [
        float(value) for value in _parameter_errors(pcov)
    ]
    leakage_rate, leakage_rate_err = _computational_leakage_rate(
        c=c,
        lambda_l=lambda_l,
        pcov=pcov,
    )

    x_fit = np.linspace(float(np.min(x_array)), float(np.max(x_array)), 1000)
    comp_fit = _computational_subspace_decay_equation(
        x_fit,
        c,
        amplitude,
        lambda_l,
    )
    residual = comp - _computational_subspace_decay_equation(
        x_array,
        c,
        amplitude,
        lambda_l,
    )
    denom = np.sum((comp - np.mean(comp)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan

    fig = _make_computational_leakage_rb_figure(
        target=target,
        x=x_array,
        computational_population=comp,
        error_y=error,
        x_fit=x_fit,
        computational_fit=comp_fit,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xaxis_type=xaxis_type,
        trace_name=trace_name,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"L = {leakage_rate:.4g}",
        showarrow=False,
    )

    if plot:
        fig.show(config=viz.get_config(filename=f"computational_leakage_rb_{target}"))
        logger.info("Target: %s", target)
        logger.info("Fit: C + D * lambda_l^n")
        logger.info("  C = %.6g +/- %.1g", c, c_err)
        logger.info("  D = %.6g +/- %.1g", amplitude, amplitude_err)
        logger.info("  lambda_l = %.6g +/- %.1g", lambda_l, lambda_l_err)
        logger.info("  R^2 = %.6g", r2)
        logger.info("  L = %.6g +/- %.1g", leakage_rate, leakage_rate_err)

    return FitResult(
        status=FitStatus.SUCCESS,
        message="Fitting successful.",
        data={
            "C": c,
            "C_err": c_err,
            "amplitude": amplitude,
            "amplitude_err": amplitude_err,
            "lambda_l": lambda_l,
            "lambda_l_err": lambda_l_err,
            "leakage_rate": leakage_rate,
            "leakage_rate_err": leakage_rate_err,
            "r2": r2,
            "model": COMPUTATIONAL_LEAKAGE_FIT_MODEL,
            "assumptions": COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS,
            "x_fit": x_fit,
            "computational_population_fit": comp_fit,
            "computational_population": comp,
            # TODO: Remove this legacy payload key after callers migrate to .figure.
            "fig": fig,
        },
        figure=fig,
    )


def fit_leakage_rb(
    *,
    target: str,
    x: ArrayLike,
    leakage_population: ArrayLike,
    computational_population: ArrayLike | None = None,
    error_y: ArrayLike | None = None,
    p0: tuple[float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    plot: bool = True,
    title: str = "Leakage randomized benchmarking",
    xlabel: str = "Number of Cliffords",
    ylabel: str = "Population",
    xaxis_type: Literal["linear", "log"] = "linear",
) -> FitResult:
    """
    Fit leakage randomized benchmarking leakage population.

    The fitted model follows Chen et al., arXiv:1509.05470:
    ``p_leakage(m) = p_inf + (p0 - p_inf) * exp(-Gamma * m)``.
    It returns ``gamma_up = p_inf * Gamma`` and
    ``gamma_down = (1 - p_inf) * Gamma``.

    Parameters
    ----------
    target
        Target qubit label.
    x
        Clifford counts.
    leakage_population
        Measured leakage-subspace population values.
    computational_population
        Optional measured computational-subspace population values, retained in
        the result payload for convenience.
    error_y
        Optional uncertainty for leakage population.
    p0
        Optional initial guess ``(p0, Gamma, p_inf)``.
    bounds
        Optional parameter bounds.
    plot
        Whether to show the Plotly figure.
    title
        Figure title prefix.
    xlabel
        X-axis label.
    ylabel
        Y-axis label.
    xaxis_type
        Plotly x-axis type.

    Returns
    -------
    FitResult
        Fit payload and associated figure.
    """
    x_array = _normalize_int_sweep(x, name="x")
    leakage = _normalize_float_series(leakage_population, name="leakage_population")
    if len(x_array) != len(leakage):
        raise ValueError("`x` and `leakage_population` must have same length.")

    comp = None
    if computational_population is not None:
        comp = _normalize_float_series(
            computational_population,
            name="computational_population",
        )
        if len(comp) != len(x_array):
            raise ValueError("`computational_population` must have same length as `x`.")

    error = None
    if error_y is not None:
        error = _normalize_float_series(error_y, name="error_y")
        if len(error) != len(x_array):
            raise ValueError("`error_y` must have same length as `x`.")

    if p0 is None:
        leakage_p0_guess = float(np.clip(leakage[0], 0.0, 1.0))
        p_inf_guess = float(np.clip(leakage[-1], 0.0, 1.0))
        p0 = (leakage_p0_guess, 0.01, p_inf_guess)

    if bounds is None:
        bounds = ((0.0, 0.0, 0.0), (1.0, np.inf, 1.0))

    try:
        popt, pcov = _curve_fit(
            _leakage_rate_equation,
            x_array,
            leakage,
            p0=p0,
            bounds=bounds,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("Failed to fit leakage RB data for %s: %s", target, exc)
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit leakage randomized benchmarking data.",
        )

    leakage_p0, gamma, p_inf = [float(value) for value in popt]
    leakage_p0_err, gamma_err, p_inf_err = [
        float(value) for value in _parameter_errors(pcov)
    ]
    gamma_up = p_inf * gamma
    gamma_down = (1.0 - p_inf) * gamma
    gamma_err, gamma_up_err, gamma_down_err = _derived_rate_errors(
        gamma=gamma,
        p_inf=p_inf,
        pcov=pcov,
    )
    lambda_l = float(np.exp(-gamma))
    lambda_l_err = float(np.exp(-gamma) * gamma_err)

    x_fit = np.linspace(float(np.min(x_array)), float(np.max(x_array)), 1000)
    leakage_fit = _leakage_rate_equation(x_fit, leakage_p0, gamma, p_inf)
    residual = leakage - _leakage_rate_equation(x_array, leakage_p0, gamma, p_inf)
    denom = np.sum((leakage - np.mean(leakage)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan

    fig = _make_leakage_rb_figure(
        target=target,
        x=x_array,
        leakage_population=leakage,
        error_y=error,
        x_fit=x_fit,
        leakage_fit=leakage_fit,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xaxis_type=xaxis_type,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"gamma_up = {gamma_up:.4g}/Clifford",
        showarrow=False,
    )

    if plot:
        fig.show(config=viz.get_config(filename=f"leakage_rb_{target}"))
        logger.info("Target: %s", target)
        logger.info("Fit: p_inf + (p0 - p_inf) * exp(-Gamma * n)")
        logger.info("  p0 = %.6g +/- %.1g", leakage_p0, leakage_p0_err)
        logger.info("  Gamma = %.6g +/- %.1g", gamma, gamma_err)
        logger.info("  p_inf = %.6g +/- %.1g", p_inf, p_inf_err)
        logger.info("  R^2 = %.6g", r2)
        logger.info("  gamma_up = %.6g +/- %.1g", gamma_up, gamma_up_err)
        logger.info("  gamma_down = %.6g +/- %.1g", gamma_down, gamma_down_err)

    return FitResult(
        status=FitStatus.SUCCESS,
        message="Fitting successful.",
        data={
            "p0": leakage_p0,
            "p0_err": leakage_p0_err,
            "Gamma": gamma,
            "Gamma_err": gamma_err,
            "gamma_up": gamma_up,
            "gamma_up_err": gamma_up_err,
            "gamma_down": gamma_down,
            "gamma_down_err": gamma_down_err,
            "lambda_l": lambda_l,
            "lambda_l_err": lambda_l_err,
            "equilibrium_leakage_population": p_inf,
            "equilibrium_leakage_population_err": p_inf_err,
            "leakage_plus_seepage": gamma,
            "leakage_plus_seepage_err": gamma_err,
            "leakage_rate": gamma_up,
            "leakage_rate_err": gamma_up_err,
            "seepage_rate": gamma_down,
            "seepage_rate_err": gamma_down_err,
            "r2": r2,
            "model": LEAKAGE_FIT_MODEL,
            "assumptions": LEAKAGE_FIT_ASSUMPTIONS,
            "x_fit": x_fit,
            "leakage_population_fit": leakage_fit,
            "computational_population": comp,
            "leakage_population": leakage,
            # TODO: Remove this legacy payload key after callers migrate to .figure.
            "fig": fig,
        },
        figure=fig,
    )


def fit_stark_pulse_leakage(
    *,
    target: str,
    duration: ArrayLike,
    leakage_population: ArrayLike,
    computational_population: ArrayLike | None = None,
    error_y: ArrayLike | None = None,
    p0: tuple[float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    plot: bool = True,
    title: str = "Stark pulse leakage",
    xlabel: str = "Stark flat-top duration (ns)",
    ylabel: str = "Population",
    xaxis_type: Literal["linear", "log"] = "linear",
) -> FitResult:
    """
    Fit leakage population versus Stark-pulse duration.

    The model is the same leakage-rate equation used by leakage RB,
    ``p_leakage(t) = p_inf + (p0 - p_inf) * exp(-Gamma * t)``, but the
    independent variable is Stark flat-top duration in ns. Reported rates are
    therefore per ns.
    """
    duration_array = _normalize_float_series(duration, name="duration")
    if np.any(duration_array < 0):
        raise ValueError("`duration` must contain non-negative values.")

    leakage = _normalize_float_series(leakage_population, name="leakage_population")
    if len(duration_array) != len(leakage):
        raise ValueError("`duration` and `leakage_population` must have same length.")

    comp = None
    if computational_population is not None:
        comp = _normalize_float_series(
            computational_population,
            name="computational_population",
        )
        if len(comp) != len(duration_array):
            raise ValueError(
                "`computational_population` must have same length as `duration`."
            )

    error = None
    if error_y is not None:
        error = _normalize_float_series(error_y, name="error_y")
        if len(error) != len(duration_array):
            raise ValueError("`error_y` must have same length as `duration`.")

    if p0 is None:
        leakage_p0_guess = float(np.clip(leakage[0], 0.0, 1.0))
        p_inf_guess = float(np.clip(leakage[-1], 0.0, 1.0))
        p0 = (leakage_p0_guess, 1e-4, p_inf_guess)

    if bounds is None:
        bounds = ((0.0, 0.0, 0.0), (1.0, np.inf, 1.0))

    try:
        popt, pcov = _curve_fit(
            _leakage_rate_equation,
            duration_array,
            leakage,
            p0=p0,
            bounds=bounds,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("Failed to fit Stark pulse leakage data for %s: %s", target, exc)
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit Stark pulse leakage data.",
        )

    leakage_p0, gamma, p_inf = [float(value) for value in popt]
    leakage_p0_err, gamma_err, p_inf_err = [
        float(value) for value in _parameter_errors(pcov)
    ]
    gamma_up = p_inf * gamma
    gamma_down = (1.0 - p_inf) * gamma
    gamma_err, gamma_up_err, gamma_down_err = _derived_rate_errors(
        gamma=gamma,
        p_inf=p_inf,
        pcov=pcov,
    )
    lambda_l = float(np.exp(-gamma))
    lambda_l_err = float(np.exp(-gamma) * gamma_err)

    x_fit = np.linspace(
        float(np.min(duration_array)),
        float(np.max(duration_array)),
        1000,
    )
    leakage_fit = _leakage_rate_equation(x_fit, leakage_p0, gamma, p_inf)
    residual = leakage - _leakage_rate_equation(
        duration_array,
        leakage_p0,
        gamma,
        p_inf,
    )
    denom = np.sum((leakage - np.mean(leakage)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan

    fig = _make_leakage_rb_figure(
        target=target,
        x=duration_array,
        leakage_population=leakage,
        error_y=error,
        x_fit=x_fit,
        leakage_fit=leakage_fit,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xaxis_type=xaxis_type,
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.95,
        y=0.95,
        text=f"gamma_up = {gamma_up:.4g}/ns",
        showarrow=False,
    )

    if plot:
        fig.show(config=viz.get_config(filename=f"stark_pulse_leakage_{target}"))
        logger.info("Target: %s", target)
        logger.info("Fit: p_inf + (p0 - p_inf) * exp(-Gamma * t)")
        logger.info("  p0 = %.6g +/- %.1g", leakage_p0, leakage_p0_err)
        logger.info("  Gamma [/ns] = %.6g +/- %.1g", gamma, gamma_err)
        logger.info("  p_inf = %.6g +/- %.1g", p_inf, p_inf_err)
        logger.info("  R^2 = %.6g", r2)
        logger.info("  gamma_up [/ns] = %.6g +/- %.1g", gamma_up, gamma_up_err)
        logger.info("  gamma_down [/ns] = %.6g +/- %.1g", gamma_down, gamma_down_err)

    return FitResult(
        status=FitStatus.SUCCESS,
        message="Fitting successful.",
        data={
            "p0": leakage_p0,
            "p0_err": leakage_p0_err,
            "Gamma": gamma,
            "Gamma_err": gamma_err,
            "gamma_up": gamma_up,
            "gamma_up_err": gamma_up_err,
            "gamma_down": gamma_down,
            "gamma_down_err": gamma_down_err,
            "lambda_l": lambda_l,
            "lambda_l_err": lambda_l_err,
            "equilibrium_leakage_population": p_inf,
            "equilibrium_leakage_population_err": p_inf_err,
            "leakage_plus_seepage": gamma,
            "leakage_plus_seepage_err": gamma_err,
            "leakage_rate": gamma_up,
            "leakage_rate_err": gamma_up_err,
            "seepage_rate": gamma_down,
            "seepage_rate_err": gamma_down_err,
            "stark_leakage_rate_per_ns": gamma_up,
            "stark_leakage_rate_per_ns_err": gamma_up_err,
            "stark_seepage_rate_per_ns": gamma_down,
            "stark_seepage_rate_per_ns_err": gamma_down_err,
            "stark_leakage_plus_seepage_per_ns": gamma,
            "stark_leakage_plus_seepage_per_ns_err": gamma_err,
            "r2": r2,
            "model": LEAKAGE_FIT_MODEL,
            "assumptions": LEAKAGE_FIT_ASSUMPTIONS,
            "duration_fit": x_fit,
            "leakage_population_fit": leakage_fit,
            "computational_population": comp,
            "leakage_population": leakage,
            # TODO: Remove this legacy payload key after callers migrate to .figure.
            "fig": fig,
        },
        figure=fig,
    )


def leakage_rb_experiment_1q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run single-qubit leakage randomized benchmarking.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse, Clifford, and measurement services.
    targets
        Target qubits to benchmark.
    n_cliffords_range
        Optional Clifford-count sweep. Defaults to powers of two up to
        `max_n_cliffords`.
    n_trials
        Number of random Clifford sequences per sweep point.
    seeds
        Random seeds, one per trial.
    max_n_cliffords
        Maximum Clifford count for the automatic sweep.
    x90
        Optional per-target `X90` waveform overrides.
    interleaved_clifford
        Optional Clifford to interleave after every random Clifford. When set,
        this function dispatches to paired SRB/IRB leakage RB and returns the
        interleaved gate leakage estimate.
    interleaved_waveform
        Optional per-target waveforms used for the interleaved Clifford.
    in_parallel
        Whether to measure all targets in one parallel schedule.
    mitigate_readout
        Whether to apply classifier confusion-matrix mitigation.
    n_shots
        Number of shots per circuit.
    shot_interval
        Interval between shots in ns.
    xaxis_type
        Plotly x-axis type.
    plot
        Whether to show figures.
    save_image
        Whether to save figures.
    **deprecated_options
        Deprecated aliases such as `shots` and `interval`.

    Returns
    -------
    Result
        Per-target leakage RB data, fit payloads, and figures.
    """
    if interleaved_clifford is not None:
        return interleaved_leakage_rb_experiment_1q(
            exp,
            targets,
            interleaved_clifford=interleaved_clifford,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            interleaved_waveform=interleaved_waveform,
            in_parallel=in_parallel,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            xaxis_type=xaxis_type,
            plot=plot,
            save_image=save_image,
            **deprecated_options,
        )

    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)
    state_labels = _basis_labels(1)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q
    if xaxis_type is None:
        xaxis_type = "linear"

    if n_cliffords_range is None:
        sweep_range = _auto_n_cliffords_range(max_n_cliffords)
    else:
        sweep_range = _normalize_int_sweep(
            n_cliffords_range,
            name="n_cliffords_range",
        )

    if seeds is None:
        seed_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or len(seed_array) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="leakage_rb_experiment_1q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )

    def build_sequence(
        target_group: list[str],
        *,
        n_clifford: int,
        seed: int,
    ) -> PulseSchedule:
        with PulseSchedule(target_group) as ps:
            for target in target_group:
                rb_sequence = exp.benchmarking_service.rb_sequence_1q(
                    target,
                    n=n_clifford,
                    x90=x90.get(target) if x90 is not None else None,
                    interleaved_waveform=interleaved_waveform.get(target)
                    if interleaved_waveform is not None
                    else None,
                    interleaved_clifford=resolved_interleaved_clifford,
                    seed=seed,
                )
                ps.add(target, rb_sequence)
        return ps

    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        state_population_trials = defaultdict(list)

        for n_clifford in sweep_range:
            trial_populations = defaultdict(list)
            for seed in seed_array:
                result = exp.measurement_service.measure(
                    sequence=build_sequence(
                        target_group,
                        n_clifford=int(n_clifford),
                        seed=int(seed),
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                for target in target_group:
                    trial_populations[target].append(
                        _state_probabilities_from_result(
                            result,
                            [target],
                            state_labels=state_labels,
                            mitigate_readout=mitigate_readout,
                        )
                    )

            for target in target_group:
                state_population_trials[target].append(trial_populations[target])

        for target in target_group:
            trials = np.asarray(state_population_trials[target], dtype=np.float64)
            summary = _summarize_population_trials(trials, state_labels)

            title = (
                "Interleaved leakage randomized benchmarking"
                if resolved_interleaved_clifford is not None
                else "Leakage randomized benchmarking"
            )
            fit_result = fit_leakage_rb(
                target=target,
                x=sweep_range,
                leakage_population=summary["leakage_population_mean"],
                computational_population=summary["computational_population_mean"],
                error_y=summary["leakage_population_std"] if n_trials > 1 else None,
                title=title,
                ylabel="Leakage population",
                xaxis_type=xaxis_type,
                plot=plot,
            )

            if fit_result.figure is not None:
                figures[target] = fit_result.figure
                if save_image:
                    viz.save_figure(
                        fit_result.figure,
                        name=f"leakage_rb_experiment_1q_{target}",
                    )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "state_labels": np.asarray(state_labels, dtype=object),
                "state_population_trials": trials,
                **summary,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": (
                    None
                    if resolved_interleaved_clifford is None
                    else resolved_interleaved_clifford.name
                ),
                **fit_result,
            }

    return Result(data=return_data, figures=figures or None)


def interleaved_leakage_rb_experiment_1q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    interleaved_clifford: str | Clifford,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run paired SRB/IRB leakage RB for a specific single-qubit gate.

    This estimates the interleaved gate leakage rate using
    ``P_comp(m) = C + D * lambda_l**m``,
    ``L = (1 - C) * (1 - lambda_l)``, and
    ``L_gate = 1 - (1 - L_IRB) / (1 - L_SRB)``.
    """
    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)
    state_labels = _basis_labels(1)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q
    if xaxis_type is None:
        xaxis_type = "linear"

    if n_cliffords_range is None:
        sweep_range = _auto_n_cliffords_range(max_n_cliffords)
    else:
        sweep_range = _normalize_int_sweep(
            n_cliffords_range,
            name="n_cliffords_range",
        )

    if seeds is None:
        seed_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or len(seed_array) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="interleaved_leakage_rb_experiment_1q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    if resolved_interleaved_clifford is None:
        raise ValueError("`interleaved_clifford` must not be None.")

    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )

    def build_sequence(
        target_group: list[str],
        *,
        n_clifford: int,
        seed: int,
        interleaved: bool,
    ) -> PulseSchedule:
        with PulseSchedule(target_group) as ps:
            for target in target_group:
                rb_sequence = exp.benchmarking_service.rb_sequence_1q(
                    target,
                    n=n_clifford,
                    x90=x90.get(target) if x90 is not None else None,
                    interleaved_waveform=interleaved_waveform.get(target)
                    if interleaved and interleaved_waveform is not None
                    else None,
                    interleaved_clifford=resolved_interleaved_clifford
                    if interleaved
                    else None,
                    seed=seed,
                )
                ps.add(target, rb_sequence)
        return ps

    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        reference_population_trials = defaultdict(list)
        interleaved_population_trials = defaultdict(list)

        for n_clifford in sweep_range:
            reference_populations = defaultdict(list)
            interleaved_populations = defaultdict(list)

            for seed in seed_array:
                for interleaved, population_store in (
                    (False, reference_populations),
                    (True, interleaved_populations),
                ):
                    result = exp.measurement_service.measure(
                        sequence=build_sequence(
                            target_group,
                            n_clifford=int(n_clifford),
                            seed=int(seed),
                            interleaved=interleaved,
                        ),
                        mode="single",
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                    )
                    for target in target_group:
                        population_store[target].append(
                            _state_probabilities_from_result(
                                result,
                                [target],
                                state_labels=state_labels,
                                mitigate_readout=mitigate_readout,
                            )
                        )

            for target in target_group:
                reference_population_trials[target].append(
                    reference_populations[target]
                )
                interleaved_population_trials[target].append(
                    interleaved_populations[target]
                )

        for target in target_group:
            reference_trials = np.asarray(
                reference_population_trials[target],
                dtype=np.float64,
            )
            interleaved_trials = np.asarray(
                interleaved_population_trials[target],
                dtype=np.float64,
            )
            reference_summary = _summarize_population_trials(
                reference_trials,
                state_labels,
            )
            interleaved_summary = _summarize_population_trials(
                interleaved_trials,
                state_labels,
            )

            reference_error = (
                reference_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            interleaved_error = (
                interleaved_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            reference_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=reference_summary[
                    "computational_population_mean"
                ],
                error_y=reference_error,
                title="Standard leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="SRB",
                plot=False,
            )
            interleaved_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=interleaved_summary[
                    "computational_population_mean"
                ],
                error_y=interleaved_error,
                title="Interleaved leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="IRB",
                plot=False,
            )

            reference_leakage_rate = float(
                reference_fit.data.get("leakage_rate", np.nan)
            )
            interleaved_leakage_rate = float(
                interleaved_fit.data.get("leakage_rate", np.nan)
            )
            reference_leakage_rate_err = float(
                reference_fit.data.get("leakage_rate_err", np.nan)
            )
            interleaved_leakage_rate_err = float(
                interleaved_fit.data.get("leakage_rate_err", np.nan)
            )
            gate_leakage_rate, gate_leakage_rate_err = _interleaved_gate_leakage_rate(
                reference_leakage_rate=reference_leakage_rate,
                interleaved_leakage_rate=interleaved_leakage_rate,
                reference_leakage_rate_err=reference_leakage_rate_err,
                interleaved_leakage_rate_err=interleaved_leakage_rate_err,
            )

            fig = _make_interleaved_leakage_rb_figure(
                target=target,
                x=sweep_range,
                reference_population=reference_summary["computational_population_mean"],
                reference_error_y=reference_error,
                reference_fit=reference_fit,
                interleaved_population=interleaved_summary[
                    "computational_population_mean"
                ],
                interleaved_error_y=interleaved_error,
                interleaved_fit=interleaved_fit,
                gate_leakage_rate=gate_leakage_rate,
                title="Interleaved leakage randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Computational population",
                xaxis_type=xaxis_type,
            )
            figures[target] = fig
            if plot:
                fig.show(
                    config=viz.get_config(
                        filename=f"interleaved_leakage_rb_experiment_1q_{target}"
                    )
                )
            if save_image:
                viz.save_figure(
                    fig,
                    name=f"interleaved_leakage_rb_experiment_1q_{target}",
                )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray([target], dtype=object),
                "state_labels": np.asarray(state_labels, dtype=object),
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": resolved_interleaved_clifford.name,
                "reference": {
                    "state_population_trials": reference_trials,
                    **reference_summary,
                    "fit_status": reference_fit.status.value,
                    "fit": dict(reference_fit.data),
                },
                "interleaved": {
                    "state_population_trials": interleaved_trials,
                    **interleaved_summary,
                    "fit_status": interleaved_fit.status.value,
                    "fit": dict(interleaved_fit.data),
                },
                "L_SRB": reference_leakage_rate,
                "L_SRB_err": reference_leakage_rate_err,
                "L_IRB": interleaved_leakage_rate,
                "L_IRB_err": interleaved_leakage_rate_err,
                "L_gate": gate_leakage_rate,
                "L_gate_err": gate_leakage_rate_err,
                "gate_leakage_rate": gate_leakage_rate,
                "gate_leakage_rate_err": gate_leakage_rate_err,
                "model": COMPUTATIONAL_LEAKAGE_FIT_MODEL,
                "assumptions": COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS,
                # TODO: Remove this legacy payload key after callers migrate.
                "fig": fig,
            }

    return Result(data=return_data, figures=figures or None)


def leakage_rb_experiment_2q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: TargetMap[PulseSchedule] | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run two-qubit leakage randomized benchmarking.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse, Clifford, and measurement services.
    targets
        CR target labels to benchmark.
    n_cliffords_range
        Optional Clifford-count sweep. Defaults to powers of two up to
        `max_n_cliffords`.
    n_trials
        Number of random Clifford sequences per sweep point.
    seeds
        Random seeds, one per trial.
    max_n_cliffords
        Maximum Clifford count for the automatic sweep.
    x90
        Optional per-qubit `X90` waveform overrides.
    zx90
        Optional per-CR-target `ZX90` schedule overrides.
    interleaved_clifford
        Optional Clifford to interleave after every random Clifford. When set,
        this function dispatches to paired SRB/IRB leakage RB and returns the
        interleaved gate leakage estimate.
    interleaved_waveform
        Optional per-CR-target schedules used for the interleaved Clifford.
    in_parallel
        Whether to measure all CR targets in one parallel schedule.
    mitigate_readout
        Whether to apply classifier confusion-matrix mitigation.
    n_shots
        Number of shots per circuit.
    shot_interval
        Interval between shots in ns.
    xaxis_type
        Plotly x-axis type.
    plot
        Whether to show figures.
    save_image
        Whether to save figures.
    **deprecated_options
        Deprecated aliases such as `shots` and `interval`.

    Returns
    -------
    Result
        Per-target leakage RB data, fit payloads, and figures.
    """
    if interleaved_clifford is not None:
        return interleaved_leakage_rb_experiment_2q(
            exp,
            targets,
            interleaved_clifford=interleaved_clifford,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            interleaved_waveform=interleaved_waveform,
            in_parallel=in_parallel,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            xaxis_type=xaxis_type,
            plot=plot,
            save_image=save_image,
            **deprecated_options,
        )

    target_list = _normalize_targets(targets)
    cr_pairs = _require_2q_targets(exp, target_list)
    classifier_targets = {qubit for pair in cr_pairs.values() for qubit in pair}
    _require_3_state_classifiers(exp, classifier_targets)
    state_labels = _basis_labels(2)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q
    if xaxis_type is None:
        xaxis_type = "linear"

    if n_cliffords_range is None:
        sweep_range = _auto_n_cliffords_range(max_n_cliffords)
    else:
        sweep_range = _normalize_int_sweep(
            n_cliffords_range,
            name="n_cliffords_range",
        )

    if seeds is None:
        seed_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or len(seed_array) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="leakage_rb_experiment_2q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )

    def build_sequence(
        target_group: list[str],
        *,
        n_clifford: int,
        seed: int,
    ) -> PulseSchedule:
        with PulseSchedule() as ps:
            sequences: dict[str, PulseSchedule] = {}
            for target in target_group:
                sequences[target] = exp.benchmarking_service.rb_sequence_2q(
                    target,
                    n=n_clifford,
                    x90=x90,
                    zx90=zx90.get(target) if zx90 is not None else None,
                    interleaved_waveform=interleaved_waveform.get(target)
                    if interleaved_waveform is not None
                    else None,
                    interleaved_clifford=resolved_interleaved_clifford,
                    seed=seed,
                )
            max_duration = max(sequence.duration for sequence in sequences.values())
            for target in target_group:
                ps.call(
                    sequences[target].padded(
                        total_duration=max_duration,
                        pad_side="left",
                        deepcopy=False,
                    )
                )
        return ps

    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        state_population_trials = defaultdict(list)

        for n_clifford in sweep_range:
            trial_populations = defaultdict(list)
            for seed in seed_array:
                result = exp.measurement_service.measure(
                    sequence=build_sequence(
                        target_group,
                        n_clifford=int(n_clifford),
                        seed=int(seed),
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                for target in target_group:
                    control_qubit, target_qubit = cr_pairs[target]
                    trial_populations[target].append(
                        _state_probabilities_from_result(
                            result,
                            [control_qubit, target_qubit],
                            state_labels=state_labels,
                            mitigate_readout=mitigate_readout,
                        )
                    )

            for target in target_group:
                state_population_trials[target].append(trial_populations[target])

        for target in target_group:
            control_qubit, target_qubit = cr_pairs[target]
            trials = np.asarray(state_population_trials[target], dtype=np.float64)
            summary = _summarize_population_trials(trials, state_labels)

            title = (
                "Interleaved leakage randomized benchmarking"
                if resolved_interleaved_clifford is not None
                else "Leakage randomized benchmarking"
            )
            fit_result = fit_leakage_rb(
                target=target,
                x=sweep_range,
                leakage_population=summary["leakage_population_mean"],
                computational_population=summary["computational_population_mean"],
                error_y=summary["leakage_population_std"] if n_trials > 1 else None,
                title=title,
                ylabel="Leakage population",
                xaxis_type=xaxis_type,
                plot=plot,
            )

            if fit_result.figure is not None:
                figures[target] = fit_result.figure
                if save_image:
                    viz.save_figure(
                        fit_result.figure,
                        name=f"leakage_rb_experiment_2q_{target}",
                    )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray(
                    [control_qubit, target_qubit], dtype=object
                ),
                "state_labels": np.asarray(state_labels, dtype=object),
                "state_population_trials": trials,
                **summary,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": (
                    None
                    if resolved_interleaved_clifford is None
                    else resolved_interleaved_clifford.name
                ),
                **fit_result,
            }

    return Result(data=return_data, figures=figures or None)


def interleaved_leakage_rb_experiment_2q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    interleaved_clifford: str | Clifford,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: TargetMap[PulseSchedule] | None = None,
    interleaved_waveform: TargetMap[PulseSchedule] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run paired SRB/IRB leakage RB for a specific two-qubit gate.

    This estimates the interleaved gate leakage rate using the computational
    subspace fit from arXiv:2511.01260:
    ``P_comp(m) = C + D * lambda_l**m``,
    ``L = (1 - C) * (1 - lambda_l)``, and
    ``L_gate = 1 - (1 - L_IRB) / (1 - L_SRB)``.
    """
    target_list = _normalize_targets(targets)
    cr_pairs = _require_2q_targets(exp, target_list)
    classifier_targets = {qubit for pair in cr_pairs.values() for qubit in pair}
    _require_3_state_classifiers(exp, classifier_targets)
    state_labels = _basis_labels(2)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_2Q
    if xaxis_type is None:
        xaxis_type = "linear"

    if n_cliffords_range is None:
        sweep_range = _auto_n_cliffords_range(max_n_cliffords)
    else:
        sweep_range = _normalize_int_sweep(
            n_cliffords_range,
            name="n_cliffords_range",
        )

    if seeds is None:
        seed_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or len(seed_array) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="interleaved_leakage_rb_experiment_2q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    if resolved_interleaved_clifford is None:
        raise ValueError("`interleaved_clifford` must not be None.")

    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )

    def target_interleaved_waveform(target: str) -> PulseSchedule | None:
        waveform = (
            interleaved_waveform.get(target)
            if interleaved_waveform is not None
            else None
        )
        if (
            waveform is None
            and resolved_interleaved_clifford.name == "ZX90"
            and zx90 is not None
        ):
            waveform = zx90.get(target)
        return waveform

    def build_sequence(
        target_group: list[str],
        *,
        n_clifford: int,
        seed: int,
        interleaved: bool,
    ) -> PulseSchedule:
        with PulseSchedule() as ps:
            sequences: dict[str, PulseSchedule] = {}
            for target in target_group:
                sequences[target] = exp.benchmarking_service.rb_sequence_2q(
                    target,
                    n=n_clifford,
                    x90=x90,
                    zx90=zx90.get(target) if zx90 is not None else None,
                    interleaved_waveform=target_interleaved_waveform(target)
                    if interleaved
                    else None,
                    interleaved_clifford=resolved_interleaved_clifford
                    if interleaved
                    else None,
                    seed=seed,
                )
            max_duration = max(sequence.duration for sequence in sequences.values())
            for target in target_group:
                ps.call(
                    sequences[target].padded(
                        total_duration=max_duration,
                        pad_side="left",
                        deepcopy=False,
                    )
                )
        return ps

    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        reference_population_trials = defaultdict(list)
        interleaved_population_trials = defaultdict(list)

        for n_clifford in sweep_range:
            reference_populations = defaultdict(list)
            interleaved_populations = defaultdict(list)

            for seed in seed_array:
                for interleaved, population_store in (
                    (False, reference_populations),
                    (True, interleaved_populations),
                ):
                    result = exp.measurement_service.measure(
                        sequence=build_sequence(
                            target_group,
                            n_clifford=int(n_clifford),
                            seed=int(seed),
                            interleaved=interleaved,
                        ),
                        mode="single",
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                    )
                    for target in target_group:
                        control_qubit, target_qubit = cr_pairs[target]
                        population_store[target].append(
                            _state_probabilities_from_result(
                                result,
                                [control_qubit, target_qubit],
                                state_labels=state_labels,
                                mitigate_readout=mitigate_readout,
                            )
                        )

            for target in target_group:
                reference_population_trials[target].append(
                    reference_populations[target]
                )
                interleaved_population_trials[target].append(
                    interleaved_populations[target]
                )

        for target in target_group:
            control_qubit, target_qubit = cr_pairs[target]
            reference_trials = np.asarray(
                reference_population_trials[target],
                dtype=np.float64,
            )
            interleaved_trials = np.asarray(
                interleaved_population_trials[target],
                dtype=np.float64,
            )
            reference_summary = _summarize_population_trials(
                reference_trials,
                state_labels,
            )
            interleaved_summary = _summarize_population_trials(
                interleaved_trials,
                state_labels,
            )

            reference_error = (
                reference_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            interleaved_error = (
                interleaved_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            reference_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=reference_summary[
                    "computational_population_mean"
                ],
                error_y=reference_error,
                title="Standard leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="SRB",
                plot=False,
            )
            interleaved_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=interleaved_summary[
                    "computational_population_mean"
                ],
                error_y=interleaved_error,
                title="Interleaved leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="IRB",
                plot=False,
            )

            reference_leakage_rate = float(
                reference_fit.data.get("leakage_rate", np.nan)
            )
            interleaved_leakage_rate = float(
                interleaved_fit.data.get("leakage_rate", np.nan)
            )
            reference_leakage_rate_err = float(
                reference_fit.data.get("leakage_rate_err", np.nan)
            )
            interleaved_leakage_rate_err = float(
                interleaved_fit.data.get("leakage_rate_err", np.nan)
            )
            gate_leakage_rate, gate_leakage_rate_err = _interleaved_gate_leakage_rate(
                reference_leakage_rate=reference_leakage_rate,
                interleaved_leakage_rate=interleaved_leakage_rate,
                reference_leakage_rate_err=reference_leakage_rate_err,
                interleaved_leakage_rate_err=interleaved_leakage_rate_err,
            )

            fig = _make_interleaved_leakage_rb_figure(
                target=target,
                x=sweep_range,
                reference_population=reference_summary["computational_population_mean"],
                reference_error_y=reference_error,
                reference_fit=reference_fit,
                interleaved_population=interleaved_summary[
                    "computational_population_mean"
                ],
                interleaved_error_y=interleaved_error,
                interleaved_fit=interleaved_fit,
                gate_leakage_rate=gate_leakage_rate,
                title="Interleaved leakage randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Computational population",
                xaxis_type=xaxis_type,
            )
            figures[target] = fig
            if plot:
                fig.show(
                    config=viz.get_config(
                        filename=f"interleaved_leakage_rb_experiment_2q_{target}"
                    )
                )
            if save_image:
                viz.save_figure(
                    fig,
                    name=f"interleaved_leakage_rb_experiment_2q_{target}",
                )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray(
                    [control_qubit, target_qubit],
                    dtype=object,
                ),
                "state_labels": np.asarray(state_labels, dtype=object),
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": resolved_interleaved_clifford.name,
                "reference": {
                    "state_population_trials": reference_trials,
                    **reference_summary,
                    "fit_status": reference_fit.status.value,
                    "fit": dict(reference_fit.data),
                },
                "interleaved": {
                    "state_population_trials": interleaved_trials,
                    **interleaved_summary,
                    "fit_status": interleaved_fit.status.value,
                    "fit": dict(interleaved_fit.data),
                },
                "L_SRB": reference_leakage_rate,
                "L_SRB_err": reference_leakage_rate_err,
                "L_IRB": interleaved_leakage_rate,
                "L_IRB_err": interleaved_leakage_rate_err,
                "L_gate": gate_leakage_rate,
                "L_gate_err": gate_leakage_rate_err,
                "gate_leakage_rate": gate_leakage_rate,
                "gate_leakage_rate_err": gate_leakage_rate_err,
                "model": COMPUTATIONAL_LEAKAGE_FIT_MODEL,
                "assumptions": COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS,
                # TODO: Remove this legacy payload key after callers migrate.
                "fig": fig,
            }

    return Result(data=return_data, figures=figures or None)


def leakage_randomized_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    **kwargs: Any,
) -> Result:
    """
    Run one- or two-qubit leakage randomized benchmarking.

    The wrapper dispatches to the 1Q or 2Q implementation based on whether the
    supplied target labels are CR targets. For targets with
    ``interleaved_clifford`` set, it runs paired SRB/IRB leakage RB and returns
    an interleaved gate leakage estimate.
    """
    target_list = _normalize_targets(targets)
    if not target_list:
        raise ValueError("`targets` must not be empty.")

    target_is_2q = [
        exp.ctx.experiment_system.get_target(target).is_cr for target in target_list
    ]
    if any(target_is_2q) and not all(target_is_2q):
        raise ValueError("`targets` must not mix 1Q and 2Q labels.")
    if all(target_is_2q):
        if kwargs.get("interleaved_clifford") is not None:
            return interleaved_leakage_rb_experiment_2q(exp, target_list, **kwargs)
        return leakage_rb_experiment_2q(exp, target_list, **kwargs)
    return leakage_rb_experiment_1q(exp, target_list, **kwargs)


def _default_stark_pulse_duration_range() -> NDArray[np.float64]:
    return np.arange(0.0, 513.0, 64.0, dtype=np.float64)


def _normalize_duration_range(values: ArrayLike | None) -> NDArray[np.float64]:
    if values is None:
        return _default_stark_pulse_duration_range()
    durations = _normalize_float_series(values, name="duration_range")
    if np.any(durations < 0):
        raise ValueError("`duration_range` must contain non-negative durations.")
    return durations


def _normalize_initial_states(
    initial_states: Collection[str] | str | None,
) -> tuple[str | None, ...]:
    if initial_states is None:
        return (None,)
    states = (
        (initial_states,) if isinstance(initial_states, str) else tuple(initial_states)
    )
    if not states:
        raise ValueError("`initial_states` must not be empty.")
    invalid = [state for state in states if state not in {"0", "1"}]
    if invalid:
        raise ValueError("`initial_states` must contain only '0' and '1'.")
    return states


def _normalize_pair_initial_states(
    initial_states: Collection[str] | str | None,
) -> tuple[str | None, ...]:
    if initial_states is None:
        return (None,)
    states = (
        (initial_states,) if isinstance(initial_states, str) else tuple(initial_states)
    )
    if not states:
        raise ValueError("`initial_states` must not be empty.")
    invalid = [
        state
        for state in states
        if len(state) != 2 or any(bit not in {"0", "1"} for bit in state)
    ]
    if invalid:
        raise ValueError(
            "`initial_states` must contain only 2Q computational states such as "
            "'00', '01', '10', or '11'."
        )
    return states


def _initial_state_payload(states: Collection[str | None]) -> NDArray[np.object_]:
    return np.asarray(
        ["none" if state is None else state for state in states], dtype=object
    )


def _pair_initial_state_mapping(
    *,
    control_qubit: str,
    target_qubit: str,
    state: str | None,
) -> dict[str, str] | None:
    if state is None:
        return None
    return {control_qubit: state[0], target_qubit: state[1]}


def _resolve_stark_drive_target_label(
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
) -> str:
    if stark_drive_qubit == "control":
        return control_qubit
    if stark_drive_qubit == "target":
        return target_qubit
    raise ValueError("`stark_drive_qubit` must be 'control' or 'target'.")


def _stark_pulse_only_sequence(
    exp: Experiment,
    target: str,
    *,
    stark_amplitude: float,
    duration: float,
    stark_ramptime: float | None,
) -> PulseSchedule:
    target_label = exp.ctx.resolve_qubit_label(target)
    resolved_stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    stark_label = stark_target(exp, target_label)
    with PulseSchedule([stark_label]) as ps:
        ps.add(
            stark_label,
            FlatTop(
                duration=float(duration) + 2 * resolved_stark_ramptime,
                amplitude=_stark_drive_amplitude(
                    exp,
                    target=target_label,
                    stark_amplitude=stark_amplitude,
                ),
                tau=resolved_stark_ramptime,
            ),
        )
    return ps


def _stark_pair_pulse_only_sequence(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit,
    stark_amplitude: float,
    duration: float,
    stark_ramptime: float | None,
) -> PulseSchedule:
    resolved_stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    stark_drive_target = _resolve_stark_drive_target_label(
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_drive_qubit=stark_drive_qubit,
    )
    stark_label = stark_target(exp, stark_drive_target)
    total_duration = float(duration) + 2 * resolved_stark_ramptime
    labels = [control_qubit, stark_label, target_qubit]
    with PulseSchedule(labels) as ps:
        ps.add(control_qubit, Blank(total_duration))
        ps.add(
            stark_label,
            FlatTop(
                duration=total_duration,
                amplitude=_stark_drive_amplitude(
                    exp,
                    target=stark_drive_target,
                    stark_amplitude=stark_amplitude,
                ),
                tau=resolved_stark_ramptime,
            ),
        )
        ps.add(target_qubit, Blank(total_duration))
    return ps


def _idle_reference_sequence(
    exp: Experiment,
    target: str,
    *,
    duration: float,
    stark_ramptime: float | None,
) -> PulseSchedule:
    target_label = exp.ctx.resolve_qubit_label(target)
    total_duration = float(duration) + 2 * _resolve_stark_ramptime(stark_ramptime)
    with PulseSchedule([target_label]) as ps:
        ps.add(target_label, Blank(total_duration))
    return ps


def _pair_idle_reference_sequence(
    *,
    control_qubit: str,
    target_qubit: str,
    duration: float,
    stark_ramptime: float | None,
) -> PulseSchedule:
    total_duration = float(duration) + 2 * _resolve_stark_ramptime(stark_ramptime)
    with PulseSchedule([control_qubit, target_qubit]) as ps:
        ps.add(control_qubit, Blank(total_duration))
        ps.add(target_qubit, Blank(total_duration))
    return ps


def _collect_stark_pulse_leakage_trials(
    exp: Experiment,
    *,
    target: str,
    state_labels: Collection[str],
    duration_range: NDArray[np.float64],
    initial_states: Collection[str | None],
    n_trials: int,
    sequence_builder: Any,
    mitigate_readout: bool,
    n_shots: int,
    shot_interval: float,
    readout_amplitudes: dict[str, float] | None,
    readout_duration: float | None,
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
    readout_ramp_time: float | None,
    readout_drag_coeff: float | None,
    readout_ramp_type: RampType | None,
    readout_amplification: bool | None,
) -> NDArray[np.float64]:
    trials_by_duration: list[list[NDArray[np.float64]]] = []
    for duration in duration_range:
        point_trials = []
        for initial_state in initial_states:
            for _ in range(n_trials):
                result = exp.measurement_service.measure(
                    sequence=sequence_builder(float(duration)),
                    initial_states=(
                        None if initial_state is None else {target: initial_state}
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    readout_ramp_time=readout_ramp_time,
                    readout_drag_coeff=readout_drag_coeff,
                    readout_ramp_type=readout_ramp_type,
                    readout_amplification=readout_amplification,
                    plot=False,
                )
                point_trials.append(
                    _state_probabilities_from_result(
                        result,
                        [target],
                        state_labels=state_labels,
                        mitigate_readout=mitigate_readout,
                    )
                )
        trials_by_duration.append(point_trials)
    return np.asarray(trials_by_duration, dtype=np.float64)


def _collect_stark_pair_pulse_leakage_trials(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    state_labels: Collection[str],
    duration_range: NDArray[np.float64],
    initial_states: Collection[str | None],
    n_trials: int,
    sequence_builder: Any,
    mitigate_readout: bool,
    n_shots: int,
    shot_interval: float,
    readout_amplitudes: dict[str, float] | None,
    readout_duration: float | None,
    readout_pre_margin: float | None,
    readout_post_margin: float | None,
    readout_ramp_time: float | None,
    readout_drag_coeff: float | None,
    readout_ramp_type: RampType | None,
    readout_amplification: bool | None,
) -> NDArray[np.float64]:
    trials_by_duration: list[list[NDArray[np.float64]]] = []
    for duration in duration_range:
        point_trials = []
        for initial_state in initial_states:
            for _ in range(n_trials):
                result = exp.measurement_service.measure(
                    sequence=sequence_builder(float(duration)),
                    initial_states=_pair_initial_state_mapping(
                        control_qubit=control_qubit,
                        target_qubit=target_qubit,
                        state=initial_state,
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    readout_amplitudes=readout_amplitudes,
                    readout_duration=readout_duration,
                    readout_pre_margin=readout_pre_margin,
                    readout_post_margin=readout_post_margin,
                    readout_ramp_time=readout_ramp_time,
                    readout_drag_coeff=readout_drag_coeff,
                    readout_ramp_type=readout_ramp_type,
                    readout_amplification=readout_amplification,
                    plot=False,
                )
                point_trials.append(
                    _state_probabilities_from_result(
                        result,
                        [control_qubit, target_qubit],
                        state_labels=state_labels,
                        mitigate_readout=mitigate_readout,
                    )
                )
        trials_by_duration.append(point_trials)
    return np.asarray(trials_by_duration, dtype=np.float64)


def _stark_pulse_condition_payload(
    *,
    target: str,
    duration_range: NDArray[np.float64],
    trials: NDArray[np.float64],
    state_labels: Collection[str],
    n_trials: int,
    plot: bool,
    title: str,
    xaxis_type: Literal["linear", "log"],
) -> tuple[dict[str, Any], FitResult]:
    summary = _summarize_population_trials(trials, state_labels)
    fit_result = fit_stark_pulse_leakage(
        target=target,
        duration=duration_range,
        leakage_population=summary["leakage_population_mean"],
        computational_population=summary["computational_population_mean"],
        error_y=summary["leakage_population_std"]
        if trials.shape[1] > 1 or n_trials > 1
        else None,
        title=title,
        ylabel="Leakage population",
        xaxis_type=xaxis_type,
        plot=plot,
    )
    return (
        {
            "state_population_trials": trials,
            **summary,
            "fit_status": fit_result.status.value,
            "fit": dict(fit_result.data),
        },
        fit_result,
    )


def stark_pulse_leakage_experiment(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    stark_amplitude: float,
    duration_range: ArrayLike | None = None,
    stark_ramptime: float | None = None,
    initial_states: Collection[str] | str | None = ("0", "1"),
    measure_reference: bool | None = None,
    mitigate_readout: bool | None = None,
    n_trials: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplitudes: dict[str, float] | None = None,
    readout_duration: float | None = None,
    readout_pre_margin: float | None = None,
    readout_post_margin: float | None = None,
    readout_ramp_time: float | None = None,
    readout_drag_coeff: float | None = None,
    readout_ramp_type: RampType | None = None,
    readout_amplification: bool | None = None,
    reset_awg_and_capunits: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure leakage from Stark pulses without applying RB or gate pulses.

    ``duration_range`` is the Stark flat-top duration in ns. The actual elapsed
    pulse window is ``duration + 2 * stark_ramptime``. By default the method
    measures both ``|0>`` and ``|1>`` initial states and averages their
    computational-subspace leakage. Set ``measure_reference=False`` to skip the
    same-duration no-Stark idle reference.
    """
    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)
    state_labels = _basis_labels(1)

    if measure_reference is None:
        measure_reference = True
    if mitigate_readout is None:
        mitigate_readout = False
    if n_trials is None:
        n_trials = 1
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if xaxis_type is None:
        xaxis_type = "linear"

    duration_values = _normalize_duration_range(duration_range)
    initial_state_values = _normalize_initial_states(initial_states)
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_pulse_leakage_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits=set(target_list))

    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}
    resolved_stark_ramptime = _resolve_stark_ramptime(stark_ramptime)

    for target in target_list:
        target_label = exp.ctx.resolve_qubit_label(target)
        stark_trials = _collect_stark_pulse_leakage_trials(
            exp,
            target=target_label,
            state_labels=state_labels,
            duration_range=duration_values,
            initial_states=initial_state_values,
            n_trials=n_trials,
            sequence_builder=lambda duration, target_label=target_label: (
                _stark_pulse_only_sequence(
                    exp,
                    target_label,
                    stark_amplitude=stark_amplitude,
                    duration=duration,
                    stark_ramptime=resolved_stark_ramptime,
                )
            ),
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
        )
        stark_payload, stark_fit = _stark_pulse_condition_payload(
            target=target_label,
            duration_range=duration_values,
            trials=stark_trials,
            state_labels=state_labels,
            n_trials=n_trials,
            plot=plot,
            title="Stark pulse leakage",
            xaxis_type=xaxis_type,
        )

        if stark_fit.figure is not None:
            figures[target_label] = stark_fit.figure
            if save_image:
                viz.save_figure(
                    stark_fit.figure,
                    name=f"stark_pulse_leakage_experiment_{target_label}",
                )

        reference_payload = None
        reference_fit = None
        excess_leakage_population_mean = None
        excess_leakage_population_std = None
        if measure_reference:
            reference_trials = _collect_stark_pulse_leakage_trials(
                exp,
                target=target_label,
                state_labels=state_labels,
                duration_range=duration_values,
                initial_states=initial_state_values,
                n_trials=n_trials,
                sequence_builder=lambda duration, target_label=target_label: (
                    _idle_reference_sequence(
                        exp,
                        target_label,
                        duration=duration,
                        stark_ramptime=resolved_stark_ramptime,
                    )
                ),
                mitigate_readout=mitigate_readout,
                n_shots=n_shots,
                shot_interval=shot_interval,
                readout_amplitudes=readout_amplitudes,
                readout_duration=readout_duration,
                readout_pre_margin=readout_pre_margin,
                readout_post_margin=readout_post_margin,
                readout_ramp_time=readout_ramp_time,
                readout_drag_coeff=readout_drag_coeff,
                readout_ramp_type=readout_ramp_type,
                readout_amplification=readout_amplification,
            )
            reference_payload, reference_fit = _stark_pulse_condition_payload(
                target=target_label,
                duration_range=duration_values,
                trials=reference_trials,
                state_labels=state_labels,
                n_trials=n_trials,
                plot=plot,
                title="Idle reference leakage",
                xaxis_type=xaxis_type,
            )
            if reference_fit.figure is not None:
                reference_key = f"{target_label}_idle_reference"
                figures[reference_key] = reference_fit.figure
                if save_image:
                    viz.save_figure(
                        reference_fit.figure,
                        name=f"stark_pulse_leakage_experiment_reference_{target_label}",
                    )
            excess_trials = (
                stark_payload["leakage_population_trials"]
                - reference_payload["leakage_population_trials"]
            )
            excess_leakage_population_mean = np.mean(excess_trials, axis=1)
            excess_leakage_population_std = np.std(excess_trials, axis=1)

        fit_data = stark_fit.data
        return_data[target_label] = {
            "duration": duration_values,
            "duration_range": duration_values,
            "total_duration": duration_values + 2 * resolved_stark_ramptime,
            "state_labels": np.asarray(state_labels, dtype=object),
            "initial_states": _initial_state_payload(initial_state_values),
            "stark_amplitude": stark_amplitude,
            "stark_ramptime": resolved_stark_ramptime,
            "mitigate_readout": mitigate_readout,
            "n_trials": n_trials,
            "n_shots": n_shots,
            "shot_interval": shot_interval,
            "stark": stark_payload,
            "reference": reference_payload,
            "excess_leakage_population_mean": excess_leakage_population_mean,
            "excess_leakage_population_std": excess_leakage_population_std,
            "stark_leakage_rate_per_ns": fit_data.get("stark_leakage_rate_per_ns"),
            "stark_leakage_rate_per_ns_err": fit_data.get(
                "stark_leakage_rate_per_ns_err"
            ),
            "stark_seepage_rate_per_ns": fit_data.get("stark_seepage_rate_per_ns"),
            "stark_seepage_rate_per_ns_err": fit_data.get(
                "stark_seepage_rate_per_ns_err"
            ),
            "stark_leakage_plus_seepage_per_ns": fit_data.get(
                "stark_leakage_plus_seepage_per_ns"
            ),
            "stark_leakage_plus_seepage_per_ns_err": fit_data.get(
                "stark_leakage_plus_seepage_per_ns_err"
            ),
            "fit_status": stark_fit.status.value,
            "fit": dict(fit_data),
        }

    return Result(data=return_data, figures=figures or None)


def stark_pair_pulse_leakage_experiment(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_amplitude: float,
    duration_range: ArrayLike | None = None,
    stark_ramptime: float | None = None,
    initial_states: Collection[str] | str | None = ("00", "01", "10", "11"),
    measure_reference: bool | None = None,
    mitigate_readout: bool | None = None,
    n_trials: int | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    readout_amplitudes: dict[str, float] | None = None,
    readout_duration: float | None = None,
    readout_pre_margin: float | None = None,
    readout_post_margin: float | None = None,
    readout_ramp_time: float | None = None,
    readout_drag_coeff: float | None = None,
    readout_ramp_type: RampType | None = None,
    readout_amplification: bool | None = None,
    reset_awg_and_capunits: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Measure pair-level leakage caused by a Stark pulse without RB or gate pulses.

    ``duration_range`` is the Stark flat-top duration in ns. The measured state
    probabilities are classified jointly over ``[control_qubit, target_qubit]``,
    so the returned leakage population is directly comparable with 2Q leakage
    RB payloads. The result key follows the dressed CR target naming used by
    :func:`stark_leakage_rb_experiment_2q`.
    """
    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    if control_label == target_label:
        raise ValueError("`control_qubit` and `target_qubit` must be different.")
    _resolve_stark_drive_target_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    _require_3_state_classifiers(exp, [control_label, target_label])
    state_labels = _basis_labels(2)

    if measure_reference is None:
        measure_reference = True
    if mitigate_readout is None:
        mitigate_readout = False
    if n_trials is None:
        n_trials = 1
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if xaxis_type is None:
        xaxis_type = "linear"

    duration_values = _normalize_duration_range(duration_range)
    initial_state_values = _normalize_pair_initial_states(initial_states)
    resolved_stark_ramptime = _resolve_stark_ramptime(stark_ramptime)
    result_target = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    stark_drive_target = _resolve_stark_drive_target_label(
        control_qubit=control_label,
        target_qubit=target_label,
        stark_drive_qubit=stark_drive_qubit,
    )

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_pair_pulse_leakage_experiment",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    if reset_awg_and_capunits:
        exp.ctx.reset_awg_and_capunits(qubits={control_label, target_label})

    stark_trials = _collect_stark_pair_pulse_leakage_trials(
        exp,
        control_qubit=control_label,
        target_qubit=target_label,
        state_labels=state_labels,
        duration_range=duration_values,
        initial_states=initial_state_values,
        n_trials=n_trials,
        sequence_builder=lambda duration: _stark_pair_pulse_only_sequence(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            stark_drive_qubit=stark_drive_qubit,
            stark_amplitude=stark_amplitude,
            duration=duration,
            stark_ramptime=resolved_stark_ramptime,
        ),
        mitigate_readout=mitigate_readout,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplitudes=readout_amplitudes,
        readout_duration=readout_duration,
        readout_pre_margin=readout_pre_margin,
        readout_post_margin=readout_post_margin,
        readout_ramp_time=readout_ramp_time,
        readout_drag_coeff=readout_drag_coeff,
        readout_ramp_type=readout_ramp_type,
        readout_amplification=readout_amplification,
    )
    stark_payload, stark_fit = _stark_pulse_condition_payload(
        target=result_target,
        duration_range=duration_values,
        trials=stark_trials,
        state_labels=state_labels,
        n_trials=n_trials,
        plot=plot,
        title="Stark pair pulse leakage",
        xaxis_type=xaxis_type,
    )

    figures: dict[str, go.Figure] = {}
    if stark_fit.figure is not None:
        figures[result_target] = stark_fit.figure
        if save_image:
            viz.save_figure(
                stark_fit.figure,
                name=f"stark_pair_pulse_leakage_experiment_{result_target}",
            )

    reference_payload = None
    reference_fit = None
    excess_leakage_population_mean = None
    excess_leakage_population_std = None
    if measure_reference:
        reference_trials = _collect_stark_pair_pulse_leakage_trials(
            exp,
            control_qubit=control_label,
            target_qubit=target_label,
            state_labels=state_labels,
            duration_range=duration_values,
            initial_states=initial_state_values,
            n_trials=n_trials,
            sequence_builder=lambda duration: _pair_idle_reference_sequence(
                control_qubit=control_label,
                target_qubit=target_label,
                duration=duration,
                stark_ramptime=resolved_stark_ramptime,
            ),
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_amplitudes=readout_amplitudes,
            readout_duration=readout_duration,
            readout_pre_margin=readout_pre_margin,
            readout_post_margin=readout_post_margin,
            readout_ramp_time=readout_ramp_time,
            readout_drag_coeff=readout_drag_coeff,
            readout_ramp_type=readout_ramp_type,
            readout_amplification=readout_amplification,
        )
        reference_payload, reference_fit = _stark_pulse_condition_payload(
            target=result_target,
            duration_range=duration_values,
            trials=reference_trials,
            state_labels=state_labels,
            n_trials=n_trials,
            plot=plot,
            title="Pair idle reference leakage",
            xaxis_type=xaxis_type,
        )
        if reference_fit.figure is not None:
            reference_key = f"{result_target}_idle_reference"
            figures[reference_key] = reference_fit.figure
            if save_image:
                viz.save_figure(
                    reference_fit.figure,
                    name=f"stark_pair_pulse_leakage_experiment_reference_{result_target}",
                )
        excess_trials = (
            stark_payload["leakage_population_trials"]
            - reference_payload["leakage_population_trials"]
        )
        excess_leakage_population_mean = np.mean(excess_trials, axis=1)
        excess_leakage_population_std = np.std(excess_trials, axis=1)

    fit_data = stark_fit.data
    return Result(
        data={
            result_target: {
                "duration": duration_values,
                "duration_range": duration_values,
                "total_duration": duration_values + 2 * resolved_stark_ramptime,
                "measured_qubits": np.asarray(
                    [control_label, target_label],
                    dtype=object,
                ),
                "state_labels": np.asarray(state_labels, dtype=object),
                "initial_states": _initial_state_payload(initial_state_values),
                "stark_amplitude": stark_amplitude,
                "stark_drive_qubit": stark_drive_qubit,
                "stark_drive_target": stark_drive_target,
                "stark_ramptime": resolved_stark_ramptime,
                "mitigate_readout": mitigate_readout,
                "n_trials": n_trials,
                "n_shots": n_shots,
                "shot_interval": shot_interval,
                "stark": stark_payload,
                "reference": reference_payload,
                "excess_leakage_population_mean": excess_leakage_population_mean,
                "excess_leakage_population_std": excess_leakage_population_std,
                "stark_leakage_rate_per_ns": fit_data.get("stark_leakage_rate_per_ns"),
                "stark_leakage_rate_per_ns_err": fit_data.get(
                    "stark_leakage_rate_per_ns_err"
                ),
                "stark_seepage_rate_per_ns": fit_data.get("stark_seepage_rate_per_ns"),
                "stark_seepage_rate_per_ns_err": fit_data.get(
                    "stark_seepage_rate_per_ns_err"
                ),
                "stark_leakage_plus_seepage_per_ns": fit_data.get(
                    "stark_leakage_plus_seepage_per_ns"
                ),
                "stark_leakage_plus_seepage_per_ns_err": fit_data.get(
                    "stark_leakage_plus_seepage_per_ns_err"
                ),
                "fit_status": stark_fit.status.value,
                "fit": dict(fit_data),
            }
        },
        figures=figures or None,
    )


def _normalize_lrb_sweep_and_seeds(
    *,
    n_cliffords_range: ArrayLike | None,
    n_trials: int | None,
    seeds: ArrayLike | None,
    max_n_cliffords: int | None,
    default_max_n_cliffords: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if n_trials <= 0:
        raise ValueError("`n_trials` must be positive.")
    if max_n_cliffords is None:
        max_n_cliffords = default_max_n_cliffords

    if n_cliffords_range is None:
        sweep_range = _auto_n_cliffords_range(max_n_cliffords)
    else:
        sweep_range = _normalize_int_sweep(
            n_cliffords_range,
            name="n_cliffords_range",
        )

    if seeds is None:
        seed_array = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_array = np.asarray(seeds, dtype=np.int64)
        if seed_array.ndim != 1 or len(seed_array) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )
    return sweep_range, seed_array, n_trials


def _stark_1q_lrb_sequence(
    exp: Experiment,
    target_group: list[str],
    *,
    stark_amplitude: float,
    stark_ramptime: float | None,
    n_clifford: int,
    seed: int,
    x90: TargetMap[Waveform] | None,
    interleaved_clifford: Clifford | None,
    interleaved_waveform: TargetMap[Waveform] | None,
) -> PulseSchedule:
    sequences = {
        target: stark_rb_sequence_1q(
            exp,
            target,
            stark_amplitude=stark_amplitude,
            stark_ramptime=stark_ramptime,
            n=n_clifford,
            seed=seed,
            x90=x90,
            interleaved_clifford=interleaved_clifford,
            interleaved_waveform=interleaved_waveform,
            plot=False,
        )
        for target in target_group
    }
    if len(sequences) == 1:
        return next(iter(sequences.values()))

    max_duration = max(sequence.duration for sequence in sequences.values())
    with PulseSchedule() as ps:
        for target in target_group:
            ps.call(
                sequences[target].padded(
                    total_duration=max_duration,
                    pad_side="left",
                    deepcopy=False,
                )
            )
    return ps


def stark_leakage_rb_experiment_1q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    stark_amplitude: float,
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """
    Run single-qubit leakage RB while each target's Stark tone is applied.

    This follows :func:`leakage_rb_experiment_1q`, but builds each circuit with
    :func:`stark_rb_sequence_1q` so the Clifford sequence is executed on the
    in-situ target under a continuous Stark drive.
    """
    if interleaved_clifford is not None:
        return stark_interleaved_leakage_rb_experiment_1q(
            exp,
            targets,
            stark_amplitude=stark_amplitude,
            interleaved_clifford=interleaved_clifford,
            stark_ramptime=stark_ramptime,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            interleaved_waveform=interleaved_waveform,
            in_parallel=in_parallel,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            xaxis_type=xaxis_type,
            plot=plot,
            save_image=save_image,
            **deprecated_options,
        )

    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)
    state_labels = _basis_labels(1)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if xaxis_type is None:
        xaxis_type = "linear"

    sweep_range, seed_array, n_trials = _normalize_lrb_sweep_and_seeds(
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        default_max_n_cliffords=DEFAULT_MAX_N_CLIFFORDS_1Q,
    )
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_leakage_rb_experiment_1q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )
    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        state_population_trials = defaultdict(list)
        for n_clifford in sweep_range:
            trial_populations = defaultdict(list)
            for seed in seed_array:
                result = exp.measurement_service.measure(
                    sequence=_stark_1q_lrb_sequence(
                        exp,
                        target_group,
                        stark_amplitude=stark_amplitude,
                        stark_ramptime=stark_ramptime,
                        n_clifford=int(n_clifford),
                        seed=int(seed),
                        x90=x90,
                        interleaved_clifford=None,
                        interleaved_waveform=None,
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                for target in target_group:
                    trial_populations[target].append(
                        _state_probabilities_from_result(
                            result,
                            [target],
                            state_labels=state_labels,
                            mitigate_readout=mitigate_readout,
                        )
                    )
            for target in target_group:
                state_population_trials[target].append(trial_populations[target])

        for target in target_group:
            trials = np.asarray(state_population_trials[target], dtype=np.float64)
            summary = _summarize_population_trials(trials, state_labels)
            fit_result = fit_leakage_rb(
                target=target,
                x=sweep_range,
                leakage_population=summary["leakage_population_mean"],
                computational_population=summary["computational_population_mean"],
                error_y=summary["leakage_population_std"] if n_trials > 1 else None,
                title="Stark leakage randomized benchmarking",
                ylabel="Leakage population",
                xaxis_type=xaxis_type,
                plot=plot,
            )

            if fit_result.figure is not None:
                figures[target] = fit_result.figure
                if save_image:
                    viz.save_figure(
                        fit_result.figure,
                        name=f"stark_leakage_rb_experiment_1q_{target}",
                    )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "state_labels": np.asarray(state_labels, dtype=object),
                "state_population_trials": trials,
                **summary,
                "stark_amplitude": stark_amplitude,
                "stark_ramptime": stark_ramptime,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": None,
                **fit_result,
            }

    return Result(data=return_data, figures=figures or None)


def stark_interleaved_leakage_rb_experiment_1q(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run paired SRB/IRB single-qubit leakage RB under a Stark tone."""
    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)
    state_labels = _basis_labels(1)

    if in_parallel is None:
        in_parallel = False
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if xaxis_type is None:
        xaxis_type = "linear"

    sweep_range, seed_array, n_trials = _normalize_lrb_sweep_and_seeds(
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        default_max_n_cliffords=DEFAULT_MAX_N_CLIFFORDS_1Q,
    )
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_interleaved_leakage_rb_experiment_1q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    if resolved_interleaved_clifford is None:
        raise ValueError("`interleaved_clifford` must not be None.")

    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )
    return_data: dict[str, dict[str, Any]] = {}
    figures: dict[str, go.Figure] = {}

    for target_group in target_groups:
        reference_population_trials = defaultdict(list)
        interleaved_population_trials = defaultdict(list)
        for n_clifford in sweep_range:
            reference_populations = defaultdict(list)
            interleaved_populations = defaultdict(list)
            for seed in seed_array:
                for interleaved, population_store in (
                    (False, reference_populations),
                    (True, interleaved_populations),
                ):
                    result = exp.measurement_service.measure(
                        sequence=_stark_1q_lrb_sequence(
                            exp,
                            target_group,
                            stark_amplitude=stark_amplitude,
                            stark_ramptime=stark_ramptime,
                            n_clifford=int(n_clifford),
                            seed=int(seed),
                            x90=x90,
                            interleaved_clifford=resolved_interleaved_clifford
                            if interleaved
                            else None,
                            interleaved_waveform=interleaved_waveform
                            if interleaved
                            else None,
                        ),
                        mode="single",
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        plot=False,
                    )
                    for target in target_group:
                        population_store[target].append(
                            _state_probabilities_from_result(
                                result,
                                [target],
                                state_labels=state_labels,
                                mitigate_readout=mitigate_readout,
                            )
                        )
            for target in target_group:
                reference_population_trials[target].append(
                    reference_populations[target]
                )
                interleaved_population_trials[target].append(
                    interleaved_populations[target]
                )

        for target in target_group:
            reference_trials = np.asarray(
                reference_population_trials[target],
                dtype=np.float64,
            )
            interleaved_trials = np.asarray(
                interleaved_population_trials[target],
                dtype=np.float64,
            )
            reference_summary = _summarize_population_trials(
                reference_trials,
                state_labels,
            )
            interleaved_summary = _summarize_population_trials(
                interleaved_trials,
                state_labels,
            )
            reference_error = (
                reference_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            interleaved_error = (
                interleaved_summary["computational_population_std"]
                if n_trials > 1
                else None
            )
            reference_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=reference_summary[
                    "computational_population_mean"
                ],
                error_y=reference_error,
                title="Standard Stark leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="SRB",
                plot=False,
            )
            interleaved_fit = fit_computational_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=interleaved_summary[
                    "computational_population_mean"
                ],
                error_y=interleaved_error,
                title="Interleaved Stark leakage randomized benchmarking",
                xaxis_type=xaxis_type,
                trace_name="IRB",
                plot=False,
            )
            reference_leakage_rate = float(
                reference_fit.data.get("leakage_rate", np.nan)
            )
            interleaved_leakage_rate = float(
                interleaved_fit.data.get("leakage_rate", np.nan)
            )
            reference_leakage_rate_err = float(
                reference_fit.data.get("leakage_rate_err", np.nan)
            )
            interleaved_leakage_rate_err = float(
                interleaved_fit.data.get("leakage_rate_err", np.nan)
            )
            gate_leakage_rate, gate_leakage_rate_err = _interleaved_gate_leakage_rate(
                reference_leakage_rate=reference_leakage_rate,
                interleaved_leakage_rate=interleaved_leakage_rate,
                reference_leakage_rate_err=reference_leakage_rate_err,
                interleaved_leakage_rate_err=interleaved_leakage_rate_err,
            )
            fig = _make_interleaved_leakage_rb_figure(
                target=target,
                x=sweep_range,
                reference_population=reference_summary["computational_population_mean"],
                reference_error_y=reference_error,
                reference_fit=reference_fit,
                interleaved_population=interleaved_summary[
                    "computational_population_mean"
                ],
                interleaved_error_y=interleaved_error,
                interleaved_fit=interleaved_fit,
                gate_leakage_rate=gate_leakage_rate,
                title="Interleaved Stark leakage randomized benchmarking",
                xlabel="Number of Cliffords",
                ylabel="Computational population",
                xaxis_type=xaxis_type,
            )
            figures[target] = fig
            if plot:
                fig.show(
                    config=viz.get_config(
                        filename=f"stark_interleaved_leakage_rb_experiment_1q_{target}"
                    )
                )
            if save_image:
                viz.save_figure(
                    fig,
                    name=f"stark_interleaved_leakage_rb_experiment_1q_{target}",
                )

            return_data[target] = {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray([target], dtype=object),
                "state_labels": np.asarray(state_labels, dtype=object),
                "stark_amplitude": stark_amplitude,
                "stark_ramptime": stark_ramptime,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": resolved_interleaved_clifford.name,
                "reference": {
                    "state_population_trials": reference_trials,
                    **reference_summary,
                    "fit_status": reference_fit.status.value,
                    "fit": dict(reference_fit.data),
                },
                "interleaved": {
                    "state_population_trials": interleaved_trials,
                    **interleaved_summary,
                    "fit_status": interleaved_fit.status.value,
                    "fit": dict(interleaved_fit.data),
                },
                "L_SRB": reference_leakage_rate,
                "L_SRB_err": reference_leakage_rate_err,
                "L_IRB": interleaved_leakage_rate,
                "L_IRB_err": interleaved_leakage_rate_err,
                "L_gate": gate_leakage_rate,
                "L_gate_err": gate_leakage_rate_err,
                "gate_leakage_rate": gate_leakage_rate,
                "gate_leakage_rate_err": gate_leakage_rate_err,
                "model": COMPUTATIONAL_LEAKAGE_FIT_MODEL,
                "assumptions": COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS,
                "fig": fig,
            }

    return Result(data=return_data, figures=figures or None)


def _stark_2q_lrb_sequence(
    exp: Experiment,
    *,
    control_qubit: str,
    target_qubit: str,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit,
    stark_ramptime: float | None,
    n_clifford: int,
    seed: int,
    x90: TargetMap[Waveform] | None,
    zx90: PulseSchedule | None,
    interleaved_clifford: Clifford | None,
    interleaved_waveform: PulseSchedule | None,
) -> PulseSchedule:
    return stark_rb_sequence_2q(
        exp,
        control_qubit=control_qubit,
        target_qubit=target_qubit,
        stark_amplitude=stark_amplitude,
        stark_drive_qubit=stark_drive_qubit,
        stark_ramptime=stark_ramptime,
        n=n_clifford,
        seed=seed,
        x90=x90,
        zx90=zx90,
        interleaved_clifford=interleaved_clifford,
        interleaved_waveform=interleaved_waveform,
        plot=False,
    )


def stark_leakage_rb_experiment_2q(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: PulseSchedule | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run two-qubit leakage RB under a Stark tone on control or target."""
    if interleaved_clifford is not None:
        return stark_interleaved_leakage_rb_experiment_2q(
            exp,
            control_qubit=control_qubit,
            target_qubit=target_qubit,
            stark_amplitude=stark_amplitude,
            interleaved_clifford=interleaved_clifford,
            stark_drive_qubit=stark_drive_qubit,
            stark_ramptime=stark_ramptime,
            n_cliffords_range=n_cliffords_range,
            n_trials=n_trials,
            seeds=seeds,
            max_n_cliffords=max_n_cliffords,
            x90=x90,
            zx90=zx90,
            interleaved_waveform=interleaved_waveform,
            mitigate_readout=mitigate_readout,
            n_shots=n_shots,
            shot_interval=shot_interval,
            xaxis_type=xaxis_type,
            plot=plot,
            save_image=save_image,
            **deprecated_options,
        )

    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if xaxis_type is None:
        xaxis_type = "linear"

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    _require_3_state_classifiers(exp, [control_label, target_label])
    state_labels = _basis_labels(2)
    result_target = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    sweep_range, seed_array, n_trials = _normalize_lrb_sweep_and_seeds(
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        default_max_n_cliffords=DEFAULT_MAX_N_CLIFFORDS_2Q,
    )
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_leakage_rb_experiment_2q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    state_population_trials: list[list[NDArray[np.float64]]] = []
    for n_clifford in sweep_range:
        trial_populations = []
        for seed in seed_array:
            result = exp.measurement_service.measure(
                sequence=_stark_2q_lrb_sequence(
                    exp,
                    control_qubit=control_label,
                    target_qubit=target_label,
                    stark_amplitude=stark_amplitude,
                    stark_drive_qubit=stark_drive_qubit,
                    stark_ramptime=stark_ramptime,
                    n_clifford=int(n_clifford),
                    seed=int(seed),
                    x90=x90,
                    zx90=zx90,
                    interleaved_clifford=None,
                    interleaved_waveform=None,
                ),
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )
            trial_populations.append(
                _state_probabilities_from_result(
                    result,
                    [control_label, target_label],
                    state_labels=state_labels,
                    mitigate_readout=mitigate_readout,
                )
            )
        state_population_trials.append(trial_populations)

    trials = np.asarray(state_population_trials, dtype=np.float64)
    summary = _summarize_population_trials(trials, state_labels)
    fit_result = fit_leakage_rb(
        target=result_target,
        x=sweep_range,
        leakage_population=summary["leakage_population_mean"],
        computational_population=summary["computational_population_mean"],
        error_y=summary["leakage_population_std"] if n_trials > 1 else None,
        title="Stark two-qubit leakage randomized benchmarking",
        ylabel="Leakage population",
        xaxis_type=xaxis_type,
        plot=plot,
    )

    figures = {}
    if fit_result.figure is not None:
        figures[result_target] = fit_result.figure
        if save_image:
            viz.save_figure(
                fit_result.figure,
                name=f"stark_leakage_rb_experiment_2q_{result_target}",
            )

    return Result(
        data={
            result_target: {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray(
                    [control_label, target_label],
                    dtype=object,
                ),
                "state_labels": np.asarray(state_labels, dtype=object),
                "state_population_trials": trials,
                **summary,
                "stark_amplitude": stark_amplitude,
                "stark_drive_qubit": stark_drive_qubit,
                "stark_ramptime": stark_ramptime,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": None,
                **fit_result,
            }
        },
        figures=figures or None,
    )


def stark_interleaved_leakage_rb_experiment_2q(
    exp: Experiment,
    control_qubit: str,
    target_qubit: str,
    *,
    stark_amplitude: float,
    interleaved_clifford: str | Clifford,
    stark_drive_qubit: StarkDriveQubit = "target",
    stark_ramptime: float | None = None,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    zx90: PulseSchedule | None = None,
    interleaved_waveform: PulseSchedule | None = None,
    mitigate_readout: bool | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    **deprecated_options: Any,
) -> Result:
    """Run paired SRB/IRB two-qubit leakage RB under a Stark tone."""
    if mitigate_readout is None:
        mitigate_readout = False
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if xaxis_type is None:
        xaxis_type = "linear"

    control_label = exp.ctx.resolve_qubit_label(control_qubit)
    target_label = exp.ctx.resolve_qubit_label(target_qubit)
    _require_3_state_classifiers(exp, [control_label, target_label])
    state_labels = _basis_labels(2)
    result_target = stark_cr_target(
        exp,
        control_label,
        target_label,
        stark_drive_qubit=stark_drive_qubit,
    )
    sweep_range, seed_array, n_trials = _normalize_lrb_sweep_and_seeds(
        n_cliffords_range=n_cliffords_range,
        n_trials=n_trials,
        seeds=seeds,
        max_n_cliffords=max_n_cliffords,
        default_max_n_cliffords=DEFAULT_MAX_N_CLIFFORDS_2Q,
    )
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="stark_interleaved_leakage_rb_experiment_2q",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    if resolved_interleaved_clifford is None:
        raise ValueError("`interleaved_clifford` must not be None.")

    reference_population_trials = []
    interleaved_population_trials = []
    for n_clifford in sweep_range:
        reference_populations = []
        interleaved_populations = []
        for seed in seed_array:
            for interleaved, population_store in (
                (False, reference_populations),
                (True, interleaved_populations),
            ):
                result = exp.measurement_service.measure(
                    sequence=_stark_2q_lrb_sequence(
                        exp,
                        control_qubit=control_label,
                        target_qubit=target_label,
                        stark_amplitude=stark_amplitude,
                        stark_drive_qubit=stark_drive_qubit,
                        stark_ramptime=stark_ramptime,
                        n_clifford=int(n_clifford),
                        seed=int(seed),
                        x90=x90,
                        zx90=zx90,
                        interleaved_clifford=resolved_interleaved_clifford
                        if interleaved
                        else None,
                        interleaved_waveform=interleaved_waveform
                        if interleaved
                        else None,
                    ),
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )
                population_store.append(
                    _state_probabilities_from_result(
                        result,
                        [control_label, target_label],
                        state_labels=state_labels,
                        mitigate_readout=mitigate_readout,
                    )
                )
        reference_population_trials.append(reference_populations)
        interleaved_population_trials.append(interleaved_populations)

    reference_trials = np.asarray(reference_population_trials, dtype=np.float64)
    interleaved_trials = np.asarray(interleaved_population_trials, dtype=np.float64)
    reference_summary = _summarize_population_trials(reference_trials, state_labels)
    interleaved_summary = _summarize_population_trials(interleaved_trials, state_labels)
    reference_error = (
        reference_summary["computational_population_std"] if n_trials > 1 else None
    )
    interleaved_error = (
        interleaved_summary["computational_population_std"] if n_trials > 1 else None
    )
    reference_fit = fit_computational_leakage_rb(
        target=result_target,
        x=sweep_range,
        computational_population=reference_summary["computational_population_mean"],
        error_y=reference_error,
        title="Standard Stark two-qubit leakage randomized benchmarking",
        xaxis_type=xaxis_type,
        trace_name="SRB",
        plot=False,
    )
    interleaved_fit = fit_computational_leakage_rb(
        target=result_target,
        x=sweep_range,
        computational_population=interleaved_summary["computational_population_mean"],
        error_y=interleaved_error,
        title="Interleaved Stark two-qubit leakage randomized benchmarking",
        xaxis_type=xaxis_type,
        trace_name="IRB",
        plot=False,
    )
    reference_leakage_rate = float(reference_fit.data.get("leakage_rate", np.nan))
    interleaved_leakage_rate = float(interleaved_fit.data.get("leakage_rate", np.nan))
    reference_leakage_rate_err = float(
        reference_fit.data.get("leakage_rate_err", np.nan)
    )
    interleaved_leakage_rate_err = float(
        interleaved_fit.data.get("leakage_rate_err", np.nan)
    )
    gate_leakage_rate, gate_leakage_rate_err = _interleaved_gate_leakage_rate(
        reference_leakage_rate=reference_leakage_rate,
        interleaved_leakage_rate=interleaved_leakage_rate,
        reference_leakage_rate_err=reference_leakage_rate_err,
        interleaved_leakage_rate_err=interleaved_leakage_rate_err,
    )
    fig = _make_interleaved_leakage_rb_figure(
        target=result_target,
        x=sweep_range,
        reference_population=reference_summary["computational_population_mean"],
        reference_error_y=reference_error,
        reference_fit=reference_fit,
        interleaved_population=interleaved_summary["computational_population_mean"],
        interleaved_error_y=interleaved_error,
        interleaved_fit=interleaved_fit,
        gate_leakage_rate=gate_leakage_rate,
        title="Interleaved Stark two-qubit leakage randomized benchmarking",
        xlabel="Number of Cliffords",
        ylabel="Computational population",
        xaxis_type=xaxis_type,
    )
    if plot:
        fig.show(
            config=viz.get_config(
                filename=f"stark_interleaved_leakage_rb_experiment_2q_{result_target}"
            )
        )
    if save_image:
        viz.save_figure(
            fig,
            name=f"stark_interleaved_leakage_rb_experiment_2q_{result_target}",
        )

    return Result(
        data={
            result_target: {
                "n_cliffords": sweep_range,
                "measured_qubits": np.asarray(
                    [control_label, target_label],
                    dtype=object,
                ),
                "state_labels": np.asarray(state_labels, dtype=object),
                "stark_amplitude": stark_amplitude,
                "stark_drive_qubit": stark_drive_qubit,
                "stark_ramptime": stark_ramptime,
                "mitigate_readout": mitigate_readout,
                "interleaved_clifford": resolved_interleaved_clifford.name,
                "reference": {
                    "state_population_trials": reference_trials,
                    **reference_summary,
                    "fit_status": reference_fit.status.value,
                    "fit": dict(reference_fit.data),
                },
                "interleaved": {
                    "state_population_trials": interleaved_trials,
                    **interleaved_summary,
                    "fit_status": interleaved_fit.status.value,
                    "fit": dict(interleaved_fit.data),
                },
                "L_SRB": reference_leakage_rate,
                "L_SRB_err": reference_leakage_rate_err,
                "L_IRB": interleaved_leakage_rate,
                "L_IRB_err": interleaved_leakage_rate_err,
                "L_gate": gate_leakage_rate,
                "L_gate_err": gate_leakage_rate_err,
                "gate_leakage_rate": gate_leakage_rate,
                "gate_leakage_rate_err": gate_leakage_rate_err,
                "model": COMPUTATIONAL_LEAKAGE_FIT_MODEL,
                "assumptions": COMPUTATIONAL_LEAKAGE_FIT_ASSUMPTIONS,
                "fig": fig,
            }
        },
        figures={result_target: fig},
    )


def stark_leakage_randomized_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    stark_amplitude: float,
    stark_drive_qubit: StarkDriveQubit = "target",
    **kwargs: Any,
) -> Result:
    """
    Dispatch Stark leakage RB based on 1Q labels or CR target labels.

    For two-qubit Stark LRB this wrapper accepts CR target labels and resolves
    each CR pair through the experiment context. Use
    :func:`stark_leakage_rb_experiment_2q` directly when passing separate
    control/target qubit labels.
    """
    target_list = _normalize_targets(targets)
    if not target_list:
        raise ValueError("`targets` must not be empty.")

    target_is_2q = [
        exp.ctx.experiment_system.get_target(target).is_cr for target in target_list
    ]
    if any(target_is_2q) and not all(target_is_2q):
        raise ValueError("`targets` must not mix 1Q and 2Q labels.")
    if not all(target_is_2q):
        return stark_leakage_rb_experiment_1q(
            exp,
            target_list,
            stark_amplitude=stark_amplitude,
            **kwargs,
        )

    data: dict[str, Any] = {}
    figures: dict[str, go.Figure] = {}
    for cr_target in target_list:
        control_qubit, target_qubit = exp.ctx.cr_pair(cr_target)
        if kwargs.get("interleaved_clifford") is not None:
            result = stark_interleaved_leakage_rb_experiment_2q(
                exp,
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                stark_amplitude=stark_amplitude,
                stark_drive_qubit=stark_drive_qubit,
                **kwargs,
            )
        else:
            result = stark_leakage_rb_experiment_2q(
                exp,
                control_qubit=control_qubit,
                target_qubit=target_qubit,
                stark_amplitude=stark_amplitude,
                stark_drive_qubit=stark_drive_qubit,
                **kwargs,
            )
        data.update(result.data)
        if result.figures is not None:
            figures.update(result.figures)
    return Result(data=data, figures=figures or None)

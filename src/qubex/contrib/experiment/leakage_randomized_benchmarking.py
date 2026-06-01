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
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.pulse import PulseArray, PulseSchedule, VirtualZ, Waveform
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options

__all__ = [
    "fit_leakage_rb",
    "leakage_randomized_benchmarking",
    "leakage_rb_experiment_1q",
    "leakage_rb_sequence_1q",
]

logger = logging.getLogger(__name__)

COMPUTATIONAL_STATE_LABELS = ("0", "1")
LEAKAGE_STATE_LABEL = "2"
LEAKAGE_FIT_MODEL = "single_exchange"
LEAKAGE_FIT_ASSUMPTIONS = (
    "single leakage subspace, Markovian gate-independent exchange, "
    "computational population follows p_inf + A * lambda_l**m"
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


def _state_probabilities_from_result(
    result: Any,
    target: str,
    *,
    mitigate_readout: bool,
) -> NDArray[np.float64]:
    if mitigate_readout:
        probabilities = result.get_mitigated_probabilities([target])
    else:
        probabilities = result.get_probabilities([target])

    return np.asarray(
        [
            *(probabilities.get(label, 0.0) for label in COMPUTATIONAL_STATE_LABELS),
            probabilities.get(LEAKAGE_STATE_LABEL, 0.0),
        ],
        dtype=np.float64,
    )


def _single_exchange_model(
    n_cliffords: ArrayLike,
    amplitude: float,
    lambda_l: float,
    p_inf: float,
) -> NDArray[np.float64]:
    n = np.asarray(n_cliffords, dtype=np.float64)
    return p_inf + amplitude * np.power(lambda_l, n)


def _parameter_errors(pcov: NDArray[np.float64]) -> NDArray[np.float64]:
    if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return np.full(3, np.nan, dtype=np.float64)
    diagonal = np.diag(pcov)
    if np.any(diagonal < 0):
        return np.full(3, np.nan, dtype=np.float64)
    return np.sqrt(diagonal)


def _derived_rate_errors(
    *,
    lambda_l: float,
    p_inf: float,
    pcov: NDArray[np.float64],
) -> tuple[float, float, float]:
    if pcov.shape != (3, 3) or not np.all(np.isfinite(pcov)):
        return np.nan, np.nan, np.nan

    rate_cov = pcov[1:3, 1:3]
    leakage_plus_seepage_err = float(np.sqrt(max(rate_cov[0, 0], 0.0)))
    exchange_rate = 1.0 - lambda_l

    leakage_gradient = np.asarray([-(1.0 - p_inf), -exchange_rate])
    seepage_gradient = np.asarray([-p_inf, exchange_rate])
    leakage_var = float(leakage_gradient @ rate_cov @ leakage_gradient)
    seepage_var = float(seepage_gradient @ rate_cov @ seepage_gradient)
    leakage_rate_err = float(np.sqrt(max(leakage_var, 0.0)))
    seepage_rate_err = float(np.sqrt(max(seepage_var, 0.0)))
    return leakage_plus_seepage_err, leakage_rate_err, seepage_rate_err


def _make_leakage_rb_figure(
    *,
    target: str,
    x: NDArray[np.int64],
    computational_population: NDArray[np.float64],
    leakage_population: NDArray[np.float64] | None,
    error_y: NDArray[np.float64] | None,
    x_fit: NDArray[np.float64] | None,
    computational_fit: NDArray[np.float64] | None,
    title: str,
    xlabel: str,
    ylabel: str,
    xaxis_type: Literal["linear", "log"],
) -> go.Figure:
    fig = viz.make_figure()

    if x_fit is not None and computational_fit is not None:
        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=computational_fit,
                mode="lines",
                name="Computational fit",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=computational_population,
            error_y=dict(type="data", array=error_y) if error_y is not None else None,
            mode="markers",
            name="Computational population",
        )
    )

    if leakage_population is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=leakage_population,
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


def leakage_rb_sequence_1q(
    exp: Experiment,
    target: str,
    *,
    n: int,
    x90: Waveform | None = None,
    interleaved_clifford: str | Clifford | None = None,
    interleaved_waveform: Waveform | None = None,
    seed: int | None = None,
    include_inverse: bool = True,
) -> PulseArray:
    """
    Build a single-qubit leakage randomized benchmarking sequence.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse and Clifford services.
    target
        Target qubit label.
    n
        Number of random Clifford gates.
    x90
        Optional `X90` waveform override.
    interleaved_clifford
        Optional Clifford to interleave after every random Clifford.
    interleaved_waveform
        Optional waveform used for the interleaved Clifford.
    seed
        Seed for random Clifford generation.
    include_inverse
        Whether to append the RB inverse Clifford.

    Returns
    -------
    PulseArray
        Pulse sequence for one target qubit.
    """
    if n < 0:
        raise ValueError("`n` must be non-negative.")

    x90_waveform = x90 or exp.pulse.x90(target)
    z90 = VirtualZ(np.pi / 2)
    sequence: list[Waveform | VirtualZ] = []

    resolved_interleaved_clifford = _resolve_clifford(exp, interleaved_clifford)
    clifford_generator = exp.benchmarking_service.clifford_generator

    if resolved_interleaved_clifford is None:
        cliffords, inverse = clifford_generator.create_rb_sequences(
            n=n,
            type="1Q",
            seed=seed,
        )
    else:
        if interleaved_waveform is None:
            if resolved_interleaved_clifford.name == "X90":
                interleaved_waveform = exp.pulse.x90(target)
            elif resolved_interleaved_clifford.name == "X180":
                interleaved_waveform = exp.pulse.x180(target)
            else:
                raise ValueError("interleaved_waveform must be provided.")
        cliffords, inverse = clifford_generator.create_irb_sequences(
            n=n,
            interleave=resolved_interleaved_clifford,
            type="1Q",
            seed=seed,
        )

    def add_gate(gate: str) -> None:
        if gate == "X90":
            sequence.append(x90_waveform)
        elif gate == "Z90":
            sequence.append(z90)
        else:
            raise ValueError(f"Invalid 1Q Clifford gate: {gate}")

    for clifford in cliffords:
        for gate in clifford:
            add_gate(gate)
        if interleaved_waveform is not None:
            sequence.append(interleaved_waveform)

    if include_inverse:
        for gate in inverse:
            add_gate(gate)

    return PulseArray(sequence)


def fit_leakage_rb(
    *,
    target: str,
    x: ArrayLike,
    computational_population: ArrayLike,
    leakage_population: ArrayLike | None = None,
    error_y: ArrayLike | None = None,
    p0: tuple[float, float, float] | None = None,
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    | None = None,
    plot: bool = True,
    title: str = "Leakage randomized benchmarking",
    xlabel: str = "Number of Cliffords",
    ylabel: str = "Population",
    xaxis_type: Literal["linear", "log"] = "linear",
) -> FitResult:
    """
    Fit leakage randomized benchmarking computational-subspace population.

    The fitted model is ``p_c(m) = p_inf + A * lambda_l**m``. It estimates
    ``L + S = 1 - lambda_l`` directly and derives separated leakage/seepage
    rates from the fitted equilibrium population.

    Parameters
    ----------
    target
        Target qubit label.
    x
        Clifford counts.
    computational_population
        Measured ``P(0) + P(1)`` values.
    leakage_population
        Optional measured ``P(2)`` values for plotting.
    error_y
        Optional uncertainty for computational population.
    p0
        Optional initial guess ``(A, lambda_l, p_inf)``.
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
    comp = _normalize_float_series(
        computational_population,
        name="computational_population",
    )
    if len(x_array) != len(comp):
        raise ValueError("`x` and `computational_population` must have same length.")

    leakage = None
    if leakage_population is not None:
        leakage = _normalize_float_series(leakage_population, name="leakage_population")
        if len(leakage) != len(x_array):
            raise ValueError("`leakage_population` must have same length as `x`.")

    error = None
    if error_y is not None:
        error = _normalize_float_series(error_y, name="error_y")
        if len(error) != len(x_array):
            raise ValueError("`error_y` must have same length as `x`.")

    if p0 is None:
        p_inf_guess = float(np.clip(comp[-1], 0.0, 1.0))
        amplitude_guess = float(np.clip(comp[0] - p_inf_guess, -1.0, 1.0))
        p0 = (amplitude_guess, 0.99, p_inf_guess)

    if bounds is None:
        bounds = ((-1.0, 0.0, 0.0), (1.0, 1.0, 1.0))

    try:
        popt, pcov = _curve_fit(
            _single_exchange_model,
            x_array,
            comp,
            p0=p0,
            bounds=bounds,
        )
    except (RuntimeError, ValueError) as exc:
        logger.warning("Failed to fit leakage RB data for %s: %s", target, exc)
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit leakage randomized benchmarking data.",
        )

    amplitude, lambda_l, p_inf = [float(value) for value in popt]
    amplitude_err, lambda_l_err, p_inf_err = [
        float(value) for value in _parameter_errors(pcov)
    ]
    leakage_plus_seepage = 1.0 - lambda_l
    leakage_rate = (1.0 - p_inf) * leakage_plus_seepage
    seepage_rate = p_inf * leakage_plus_seepage
    (
        leakage_plus_seepage_err,
        leakage_rate_err,
        seepage_rate_err,
    ) = _derived_rate_errors(lambda_l=lambda_l, p_inf=p_inf, pcov=pcov)

    x_fit = np.linspace(float(np.min(x_array)), float(np.max(x_array)), 1000)
    comp_fit = _single_exchange_model(x_fit, amplitude, lambda_l, p_inf)
    residual = comp - _single_exchange_model(x_array, amplitude, lambda_l, p_inf)
    denom = np.sum((comp - np.mean(comp)) ** 2)
    r2 = float(1.0 - np.sum(residual**2) / denom) if denom > 0 else np.nan

    fig = _make_leakage_rb_figure(
        target=target,
        x=x_array,
        computational_population=comp,
        leakage_population=leakage,
        error_y=error,
        x_fit=x_fit,
        computational_fit=comp_fit,
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
        text=f"L+S = {leakage_plus_seepage:.4g}",
        showarrow=False,
    )

    if plot:
        fig.show(config=viz.get_config(filename=f"leakage_rb_{target}"))
        logger.info("Target: %s", target)
        logger.info("Fit: p_inf + A * lambda_l^n")
        logger.info("  A = %.6g +/- %.1g", amplitude, amplitude_err)
        logger.info("  lambda_l = %.6g +/- %.1g", lambda_l, lambda_l_err)
        logger.info("  p_inf = %.6g +/- %.1g", p_inf, p_inf_err)
        logger.info("  R^2 = %.6g", r2)
        logger.info("  L + S = %.6g +/- %.1g", leakage_plus_seepage, leakage_plus_seepage_err)
        logger.info("  L = %.6g +/- %.1g", leakage_rate, leakage_rate_err)
        logger.info("  S = %.6g +/- %.1g", seepage_rate, seepage_rate_err)

    return FitResult(
        status=FitStatus.SUCCESS,
        message="Fitting successful.",
        data={
            "amplitude": amplitude,
            "amplitude_err": amplitude_err,
            "lambda_l": lambda_l,
            "lambda_l_err": lambda_l_err,
            "equilibrium_computational_population": p_inf,
            "equilibrium_computational_population_err": p_inf_err,
            "leakage_plus_seepage": leakage_plus_seepage,
            "leakage_plus_seepage_err": leakage_plus_seepage_err,
            "leakage_rate": leakage_rate,
            "leakage_rate_err": leakage_rate_err,
            "seepage_rate": seepage_rate,
            "seepage_rate_err": seepage_rate_err,
            "r2": r2,
            "model": LEAKAGE_FIT_MODEL,
            "assumptions": LEAKAGE_FIT_ASSUMPTIONS,
            "x_fit": x_fit,
            "computational_population_fit": comp_fit,
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
    include_inverse: bool | None = None,
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
        Optional Clifford to interleave after every random Clifford.
    interleaved_waveform
        Optional per-target waveforms used for the interleaved Clifford.
    include_inverse
        Whether to append the RB inverse Clifford.
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
    target_list = _normalize_targets(targets)
    _require_1q_targets(exp, target_list)
    _require_3_state_classifiers(exp, target_list)

    if include_inverse is None:
        include_inverse = True
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
    target_groups = [target_list] if in_parallel else [[target] for target in target_list]

    def build_sequence(
        target_group: list[str],
        *,
        n_clifford: int,
        seed: int,
    ) -> PulseSchedule:
        with PulseSchedule(target_group) as ps:
            for target in target_group:
                rb_sequence = leakage_rb_sequence_1q(
                    exp,
                    target,
                    n=n_clifford,
                    x90=x90.get(target) if x90 is not None else None,
                    interleaved_waveform=interleaved_waveform.get(target)
                    if interleaved_waveform is not None
                    else None,
                    interleaved_clifford=resolved_interleaved_clifford,
                    seed=seed,
                    include_inverse=include_inverse,
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
                    state_classification=True,
                    plot=False,
                )
                for target in target_group:
                    trial_populations[target].append(
                        _state_probabilities_from_result(
                            result,
                            target,
                            mitigate_readout=mitigate_readout,
                        )
                    )

            for target in target_group:
                state_population_trials[target].append(trial_populations[target])

        for target in target_group:
            trials = np.asarray(state_population_trials[target], dtype=np.float64)
            state_mean = np.mean(trials, axis=1)
            state_std = np.std(trials, axis=1)
            computational_mean = np.sum(state_mean[:, :2], axis=1)
            computational_std = np.sqrt(np.sum(state_std[:, :2] ** 2, axis=1))
            leakage_mean = state_mean[:, 2]
            leakage_std = state_std[:, 2]

            title = (
                "Interleaved leakage randomized benchmarking"
                if resolved_interleaved_clifford is not None
                else "Leakage randomized benchmarking"
            )
            fit_result = fit_leakage_rb(
                target=target,
                x=sweep_range,
                computational_population=computational_mean,
                leakage_population=leakage_mean,
                error_y=computational_std if n_trials > 1 else None,
                title=title,
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
                "state_labels": np.asarray(
                    [*COMPUTATIONAL_STATE_LABELS, LEAKAGE_STATE_LABEL],
                    dtype=object,
                ),
                "state_population_trials": trials,
                "state_population_mean": state_mean,
                "state_population_std": state_std,
                "computational_population_mean": computational_mean,
                "computational_population_std": computational_std,
                "leakage_population_mean": leakage_mean,
                "leakage_population_std": leakage_std,
                "mitigate_readout": mitigate_readout,
                "include_inverse": include_inverse,
                "interleaved_clifford": (
                    None
                    if resolved_interleaved_clifford is None
                    else resolved_interleaved_clifford.name
                ),
                **fit_result,
            }

    return Result(data=return_data, figures=figures or None)


def leakage_randomized_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    **kwargs: Any,
) -> Result:
    """
    Run single-qubit leakage randomized benchmarking.

    This convenience wrapper delegates to :func:`leakage_rb_experiment_1q`.
    """
    return leakage_rb_experiment_1q(exp, targets, **kwargs)

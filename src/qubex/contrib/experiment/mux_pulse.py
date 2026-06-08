"""Contributed mux pulse calibration helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit, least_squares

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.analysis.fit_result import FitResult, FitStatus
from qubex.clifford import Clifford
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
    DRAG_COEFF,
    DRAG_HPI_DURATION,
    DRAG_PI_DURATION,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models import Result
from qubex.pulse import Drag, FlatTop, PulseSchedule, Waveform
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options


def _normalize_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    if targets is None:
        return list(exp.ctx.qubit_labels)
    if isinstance(targets, str):
        return [targets]
    return list(targets)


def _require_mux_targets(
    exp: Experiment,
    targets: Collection[str] | str | None,
) -> list[str]:
    target_list = _normalize_targets(exp, targets)
    if len(target_list) != 2:
        raise ValueError("Mux pulse needs just 2 targets")
    return target_list


def _measurement_sampling_period_ns(exp: Experiment) -> float:
    measurement = getattr(exp.ctx, "measurement", None)
    sampling_period = getattr(measurement, "sampling_period", None)
    if sampling_period is not None:
        return float(sampling_period)
    return float(getattr(FlatTop, "SAMPLING_PERIOD", 1.0))


def _drag_param(
    exp: Experiment,
    target: str,
    pulse_type: Literal["pi", "hpi"],
) -> Mapping[str, Any] | None:
    if pulse_type == "hpi":
        return exp.calib_note.get_drag_hpi_param(target)
    if pulse_type == "pi":
        return exp.calib_note.get_drag_pi_param(target)
    raise ValueError("Invalid pulse type.")


def _drag_duration(
    pulse_type: Literal["pi", "hpi"],
    duration: float | None,
) -> float:
    if duration is not None:
        return duration
    if pulse_type == "hpi":
        return DRAG_HPI_DURATION
    if pulse_type == "pi":
        return DRAG_PI_DURATION
    raise ValueError("Invalid pulse type.")


def _drag_rotation_rate(
    *,
    pulse: Waveform,
    pulse_type: Literal["pi", "hpi"],
    sampling_period_ns: float,
) -> float:
    area = pulse.real.sum() * sampling_period_ns
    if pulse_type == "hpi":
        return 0.25 / area
    if pulse_type == "pi":
        return 0.5 / area
    raise ValueError("Invalid pulse type.")


def _initial_drag_beta(
    exp: Experiment,
    target: str,
    pulse_type: Literal["pi", "hpi"],
    *,
    drag_coeff: float,
    use_stored_beta: bool,
) -> float:
    param = _drag_param(exp, target, pulse_type)
    if param is not None and use_stored_beta:
        return float(param["beta"])
    return float(-drag_coeff / exp.ctx.qubits[target].alpha)


def _initial_drag_amplitude(
    exp: Experiment,
    target: str,
    pulse_type: Literal["pi", "hpi"],
    *,
    pulse: Waveform,
    sampling_period_ns: float,
    use_stored_amplitude: bool,
) -> float:
    param = _drag_param(exp, target, pulse_type)
    if param is not None and use_stored_amplitude:
        return float(param["amplitude"])
    rabi_rate = _drag_rotation_rate(
        pulse=pulse,
        pulse_type=pulse_type,
        sampling_period_ns=sampling_period_ns,
    )
    return float(exp.calc_control_amplitude(target, rabi_rate))


def _update_drag_param(
    exp: Experiment,
    target: str,
    pulse_type: Literal["pi", "hpi"],
    *,
    duration: float,
    amplitude: float,
    beta: float,
) -> None:
    value = {
        "target": target,
        "duration": duration,
        "amplitude": amplitude,
        "beta": beta,
    }
    if pulse_type == "hpi":
        exp.calib_note.update_drag_hpi_param(target, value)
    elif pulse_type == "pi":
        exp.calib_note.update_drag_pi_param(target, value)
    else:
        raise ValueError("Invalid pulse type.")


def _amplitude_sweep_range(
    amplitude: float,
    *,
    n_points: int,
    n_rotations: int,
) -> NDArray[np.float64]:
    delta = 0.5 / n_rotations
    ampl_min = np.clip(amplitude * (1 - delta), 0.0, 1.0)
    ampl_max = np.clip(amplitude * (1 + delta), 0.0, 1.0)
    if ampl_min == ampl_max:
        ampl_min = 0.0
        ampl_max = 1.0
    return np.linspace(ampl_min, ampl_max, n_points)


def _resolve_x90_map(
    exp: Experiment,
    targets: Sequence[str],
    x90: TargetMap[Waveform] | None,
) -> dict[str, Waveform]:
    if x90 is None:
        return {target: exp.pulse.x90(target) for target in targets}
    return {target: x90[target] for target in targets}


def _apply_final_state(
    ps: PulseSchedule,
    targets: Sequence[str],
    x90: Mapping[str, Waveform],
    final_state: str,
) -> None:
    if len(final_state) != len(targets):
        raise ValueError("final_state length must match the number of targets.")
    for target, state in zip(targets, final_state):
        if state == "0":
            continue
        if state == "1":
            ps.add(target, x90[target].repeated(2))
            continue
        raise ValueError("Only computational final states containing 0/1 are supported.")


def _complete_probabilities(result: Any, targets: Sequence[str]) -> dict[str, float]:
    probabilities = result.get_probabilities(targets)
    return {
        label: float(probabilities.get(label, 0.0))
        for label in result.get_basis_labels(targets)
    }


def _leakage_probability(probabilities: Mapping[str, float]) -> float:
    return float(sum(value for label, value in probabilities.items() if "2" in label))


def _per_target_leakage(
    probabilities: Mapping[str, float],
    targets: Sequence[str],
) -> dict[str, float]:
    return {
        target: float(
            sum(
                value
                for label, value in probabilities.items()
                if len(label) > index and label[index] == "2"
            )
        )
        for index, target in enumerate(targets)
    }


def fit_2d_detuning_map(
    *,
    targets: Sequence[str],
    x_detuning_range: NDArray,
    y_detuning_range: NDArray,
    data: NDArray,
    p0=None,
    plot: bool = True,
    title: str = "Detuning map",
    xlabel: str = "Detuning range",
    ylabel: str = "Detuning range",
    zlabel: str = "Probability",
) -> FitResult:
    """
    Fit a 2D detuning map with a 2D Gaussian and extract the optimal detunings.

    Parameters
    ----------
    targets : Sequence[str]
        List of two qubit labels [q0_label, q1_label].
    x_detuning_range : NDArray[np.float64]
        1D array of detuning values for q0 (axis-0).
    y_detuning_range : NDArray[np.float64]
        1D array of detuning values for q1 (axis-1).
    data : NDArray[np.float64]
        2D array of measured values with shape.
        The first index corresponds to q0, the second index to q1.
    p0 : optional
        Initial guess for the Gaussian parameters
        (A, x0, y0, sx, sy, C). If None, a heuristic guess is used.
    plot : bool, optional
        If True, show a 3D Plotly surface of the data and the fitted peak.
    title : str, optional
        Title prefix for the figure.
    xlabel : str, optional
        Label for the x-axis (q0 detuning). If None, a label is generated
        from targets[0].
    ylabel : str, optional
        Label for the y-axis (q1 detuning). If None, a label is generated
        from targets[1].
    zlabel : str, optional
        Label for the z-axis (metric).

    Returns
    -------
    FitResult
        Result object with the following fields in data:
            - "amplitude": dict mapping each target to its optimal detuning,
            - "metric": fitted metric value at the optimal detuning
            - "r2": coefficient of determination of the Gaussian fit
            - "popt": optimal Gaussian parameters
            - "pcov": covariance matrix of the fit
            - "fig": Plotly Figure object with the 3D surface and peak marker
    """
    # --- input normalization and checks ---
    x_detuning_range = np.asarray(x_detuning_range, dtype=np.float64)
    y_detuning_range = np.asarray(y_detuning_range, dtype=np.float64)
    Z = np.asarray(data, dtype=np.float64)

    if Z.shape != (x_detuning_range.size, y_detuning_range.size):
        return FitResult(
            status=FitStatus.ERROR,
            message=(
                "Shape mismatch: data must have shape "
                "(len(x_detuning_range), len(y_detuning_range))."
            ),
            data={},
        )

    if len(targets) != 2:
        return FitResult(
            status=FitStatus.ERROR,
            message="targets must be a sequence of two labels [q0, q1].",
            data={},
        )

    q0_label, q1_label = targets

    if xlabel is None:
        xlabel = f"{q0_label} detuning"
    if ylabel is None:
        ylabel = f"{q1_label} detuning"

    # --- 2D Gaussian model ---
    def gauss2d(xy_tuple, A, x0, y0, sx, sy, C):
        """
        2D Gaussian:
          f(x, y) = A * exp(-(((x - x0)^2)/(2*sx^2) + ((y - y0)^2)/(2*sy^2))) + C
        """
        x_, y_ = xy_tuple
        return A * np.exp(
            -(((x_ - x0) ** 2) / (2 * sx**2) + ((y_ - y0) ** 2) / (2 * sy**2))
        ) + C

    # --- meshgrid ---
    X, Y = np.meshgrid(x_detuning_range, y_detuning_range, indexing="ij")

    Xv = X.ravel()
    Yv = Y.ravel()
    Zv = Z.ravel()

    # --- initial parameter guess ---
    A_init = Zv.max() - Zv.min()
    idx_peak = np.argmax(Zv)
    x0_init = Xv[idx_peak]
    y0_init = Yv[idx_peak]

    span_x = x_detuning_range.max() - x_detuning_range.min()
    span_y = y_detuning_range.max() - y_detuning_range.min()
    sx_init = span_x / 4 if span_x > 0 else 0.001
    sy_init = span_y / 4 if span_y > 0 else 0.001
    C_init = Zv.min()

    if p0 is None:
        p0 = (A_init, x0_init, y0_init, sx_init, sy_init, C_init)

    # --- Gaussian fit ---
    try:
        popt, pcov = curve_fit(gauss2d, (Xv, Yv), Zv, p0=p0)
    except RuntimeError:
        print(f"Failed to fit 2D map for {q0_label}-{q1_label}.")
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit 2D detuning map.",
            data={
                "amplitude": {q0_label: np.nan, q1_label: np.nan},
                "r2": np.nan,
            },
        )

    A_fit, x0_fit, y0_fit, sx_fit, sy_fit, C_fit = popt

    # --- R² computation ---
    Z_fit_vec = gauss2d((Xv, Yv), *popt)
    ss_res = np.sum((Zv - Z_fit_vec) ** 2)
    ss_tot = np.sum((Zv - Zv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # --- metric at the optimal detuning ---
    z_opt = gauss2d(
        (np.array([x0_fit], dtype=np.float64),
         np.array([y0_fit], dtype=np.float64)),
        *popt,
    )[0]

    # --- 3D Plotly surface ---
    fig = go.Figure()
    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorbar=dict(title=zlabel),
        name="Data",
    )
    fig.add_scatter3d(
        x=[x0_fit],
        y=[y0_fit],
        z=[z_opt],
        mode="markers",
        marker=dict(size=5),
        name="Peak (fit)",
    )
    fig.update_layout(
        title=f"{title} : {q0_label}-{q1_label}",
        scene=dict(
            xaxis_title=f"{xlabel} {q0_label}",
            yaxis_title=f"{ylabel} {q1_label}",
            zaxis_title=zlabel,
        ),
        width=600,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if plot:
        fig.show(config=fitting._plotly_config(f"detuning_map_{q0_label}_{q1_label}"))

    return FitResult(
        status=FitStatus.SUCCESS,
        message="2D detuning map fitting successful.",
        data={
            "detuning": {
                q0_label: x0_fit,
                q1_label: y0_fit,
            },
            "metric": z_opt,
            "r2": r2,
            "popt": popt,
            "pcov": pcov,
            "fig": fig,
        },
        figure=fig,
    )


def fit_2d_ampl_map(
    *,
    targets: Sequence[str],
    x_amplitude_range: NDArray,
    y_amplitude_range: NDArray,
    data: NDArray,
    p0=None,
    plot: bool = True,
    title: str = "Amplitude map",
    xlabel: str = "Amplitude range",
    ylabel: str = "Amplitude range",
    zlabel: str = "Probability",
) -> FitResult:
    """
    Fit a 2D amplitude map with a 2D Gaussian and extract the optimal amplitudes.

    Parameters
    ----------
    targets : Sequence[str]
        List of two qubit labels [q0_label, q1_label].
    x_amplitude_range : NDArray[np.float64]
        1D array of amplitudes for q0 (axis-0).
    y_amplitude_range : NDArray[np.float64]
        1D array of amplitudes for q1 (axis-1).
    data : NDArray[np.float64]
        2D array of measured values with shape.
        The first index corresponds to q0, the second index to q1.
    p0 : optional
        Initial guess for the Gaussian parameters
        (A, x0, y0, sx, sy, C). If None, a heuristic guess is used.
    plot : bool, optional
        If True, show a 3D Plotly surface of the data and the fitted peak.
    title : str, optional
        Title prefix for the figure.
    xlabel : str, optional
        Label for the x-axis (q0 amplitude). If None, a label is generated
        from targets[0].
    ylabel : str, optional
        Label for the y-axis (q1 amplitude). If None, a label is generated
        from targets[1].
    zlabel : str, optional
        Label for the z-axis (metric).

    Returns
    -------
    FitResult
        Result object with the following fields in data:
            - "amplitude": dict mapping each target to its optimal amplitude
            - "metric": fitted metric value at the optimal amplitudes
            - "r2": coefficient of determination of the Gaussian fit
            - "popt": optimal Gaussian parameters
            - "pcov": covariance matrix of the fit
            - "fig": Plotly Figure object with the 3D surface and peak marker
    """
    # --- input normalization and checks ---
    x_amplitude_range = np.asarray(x_amplitude_range, dtype=np.float64)
    y_amplitude_range = np.asarray(y_amplitude_range, dtype=np.float64)
    Z = np.asarray(data, dtype=np.float64)

    if Z.shape != (x_amplitude_range.size, y_amplitude_range.size):
        return FitResult(
            status=FitStatus.ERROR,
            message=(
                "Shape mismatch: data must have shape "
                "(len(x_amplitude_range), len(y_amplitude_range))."
            ),
            data={},
        )

    if len(targets) != 2:
        return FitResult(
            status=FitStatus.ERROR,
            message="targets must be a sequence of two labels [q0, q1].",
            data={},
        )

    q0_label, q1_label = targets

    if xlabel is None:
        xlabel = f"{q0_label} amplitude"
    if ylabel is None:
        ylabel = f"{q1_label} amplitude"

    # --- 2D Gaussian model ---
    def gauss2d(xy_tuple, A, x0, y0, sx, sy, C):
        """
        2D Gaussian:
          f(x, y) = A * exp(-(((x - x0)^2)/(2*sx^2) + ((y - y0)^2)/(2*sy^2))) + C
        """
        x_, y_ = xy_tuple
        return A * np.exp(
            -(((x_ - x0) ** 2) / (2 * sx**2) + ((y_ - y0) ** 2) / (2 * sy**2))
        ) + C

    # --- meshgrid ---
    X, Y = np.meshgrid(x_amplitude_range, y_amplitude_range, indexing="ij")

    Xv = X.ravel()
    Yv = Y.ravel()
    Zv = Z.ravel()

    # --- initial parameter guess ---
    A_init = Zv.max() - Zv.min()
    idx_peak = np.argmax(Zv)
    x0_init = Xv[idx_peak]
    y0_init = Yv[idx_peak]

    span_x = x_amplitude_range.max() - x_amplitude_range.min()
    span_y = y_amplitude_range.max() - y_amplitude_range.min()
    sx_init = span_x / 4 if span_x > 0 else 0.001
    sy_init = span_y / 4 if span_y > 0 else 0.001
    C_init = Zv.min()

    if p0 is None:
        p0 = (A_init, x0_init, y0_init, sx_init, sy_init, C_init)

    # --- Gaussian fit ---
    try:
        popt, pcov = curve_fit(gauss2d, (Xv, Yv), Zv, p0=p0)
    except RuntimeError:
        print(f"Failed to fit 2D amplitude map for {q0_label}-{q1_label}.")
        return FitResult(
            status=FitStatus.ERROR,
            message="Failed to fit 2D amplitude map.",
            data={
                "amplitude": {q0_label: np.nan, q1_label: np.nan},
                "r2": np.nan,
            },
        )

    A_fit, x0_fit, y0_fit, sx_fit, sy_fit, C_fit = popt

    # --- R² computation ---
    Z_fit_vec = gauss2d((Xv, Yv), *popt)
    ss_res = np.sum((Zv - Z_fit_vec) ** 2)
    ss_tot = np.sum((Zv - Zv.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # --- metric at the optimal amplitudes ---
    z_opt = gauss2d(
        (np.array([x0_fit], dtype=np.float64),
         np.array([y0_fit], dtype=np.float64)),
        *popt,
    )[0]

    # --- 3D Plotly surface ---
    fig = go.Figure()
    fig.add_surface(
        x=X,
        y=Y,
        z=Z,
        colorbar=dict(title=zlabel),
        name="Data",
    )
    fig.add_scatter3d(
        x=[x0_fit],
        y=[y0_fit],
        z=[z_opt],
        mode="markers",
        marker=dict(size=5),
        name="Peak (fit)",
    )
    fig.update_layout(
        title=f"{title} : {q0_label}-{q1_label}",
        scene=dict(
            xaxis_title=f"{xlabel} {q0_label}",
            yaxis_title=f"{ylabel} {q1_label}",
            zaxis_title=zlabel,
        ),
        width=600,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if plot:
        fig.show(config=fitting._plotly_config(f"ampl_map_{q0_label}_{q1_label}"))

    return FitResult(
        status=FitStatus.SUCCESS,
        message="2D amplitude map fitting successful.",
        data={
            "amplitude": {
                q0_label: x0_fit,
                q1_label: y0_fit,
            },
            "metric": z_opt,
            "r2": r2,
            "popt": popt,
            "pcov": pcov,
            "fig": fig,
        },
        figure=fig,
    )


def _poly2d_terms(degree: int) -> list[tuple[int, int]]:
    return [
        (x_power, total_power - x_power)
        for total_power in range(degree + 1)
        for x_power in range(total_power + 1)
    ]


def _poly2d_design(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    terms: Sequence[tuple[int, int]],
) -> NDArray[np.float64]:
    return np.column_stack([x**x_power * y**y_power for x_power, y_power in terms])


def _poly2d_eval(
    coeffs: NDArray[np.float64],
    x: NDArray[np.float64] | float,
    y: NDArray[np.float64] | float,
    terms: Sequence[tuple[int, int]],
) -> NDArray[np.float64] | float:
    total = 0.0
    for coeff, (x_power, y_power) in zip(coeffs, terms, strict=True):
        total = total + coeff * np.asarray(x) ** x_power * np.asarray(y) ** y_power
    return total


def fit_2d_drag_beta_map(
    *,
    targets: Sequence[str],
    x_beta_range: NDArray,
    y_beta_range: NDArray,
    data: NDArray,
    degree: int = 3,
    plot: bool = True,
    title: str = "DRAG beta map",
    xlabel: str = "Beta",
    ylabel: str = "Beta",
    zlabel: str = "Residual",
) -> FitResult:
    """
    Fit simultaneous DRAG beta sweeps and extract the joint root.

    `data[0]` and `data[1]` are the normalized DRAG beta calibration signals
    for `targets[0]` and `targets[1]`.  The optimum is the bounded least-squares
    point where both fitted polynomial surfaces are closest to zero, matching
    the single-qubit DRAG beta calibration criterion.
    """
    if degree < 1:
        return FitResult(
            status=FitStatus.ERROR,
            message="degree must be at least 1.",
            data={},
        )
    if len(targets) != 2:
        return FitResult(
            status=FitStatus.ERROR,
            message="targets must be a sequence of two labels [q0, q1].",
            data={},
        )

    x_beta_range = np.asarray(x_beta_range, dtype=np.float64)
    y_beta_range = np.asarray(y_beta_range, dtype=np.float64)
    values = np.asarray(data, dtype=np.float64)

    expected_shape = (2, x_beta_range.size, y_beta_range.size)
    if values.shape != expected_shape:
        return FitResult(
            status=FitStatus.ERROR,
            message=(
                "Shape mismatch: data must have shape "
                "(2, len(x_beta_range), len(y_beta_range))."
            ),
            data={},
        )

    q0_label, q1_label = targets
    X, Y = np.meshgrid(x_beta_range, y_beta_range, indexing="ij")
    Xv = X.ravel()
    Yv = Y.ravel()

    terms = _poly2d_terms(degree)
    design = _poly2d_design(Xv, Yv, terms)
    coeffs: dict[str, NDArray[np.float64]] = {}
    surfaces: dict[str, NDArray[np.float64]] = {}
    for index, target in enumerate(targets):
        coeff, *_ = np.linalg.lstsq(design, values[index].ravel(), rcond=None)
        coeffs[target] = coeff
        surfaces[target] = np.asarray(
            _poly2d_eval(coeff, X, Y, terms),
            dtype=np.float64,
        )

    residual_grid = np.sqrt(surfaces[q0_label] ** 2 + surfaces[q1_label] ** 2)
    idx_min = np.unravel_index(np.argmin(residual_grid), residual_grid.shape)
    initial = np.array([X[idx_min], Y[idx_min]], dtype=np.float64)

    def residual(beta_pair: NDArray[np.float64]) -> NDArray[np.float64]:
        x, y = beta_pair
        return np.array(
            [
                _poly2d_eval(coeffs[q0_label], x, y, terms),
                _poly2d_eval(coeffs[q1_label], x, y, terms),
            ],
            dtype=np.float64,
        )

    lower = np.array([x_beta_range.min(), y_beta_range.min()], dtype=np.float64)
    upper = np.array([x_beta_range.max(), y_beta_range.max()], dtype=np.float64)
    try:
        opt = least_squares(residual, initial, bounds=(lower, upper))
        beta0, beta1 = opt.x
        metric = float(np.linalg.norm(opt.fun))
    except ValueError as exc:
        return FitResult(
            status=FitStatus.ERROR,
            message=f"Failed to fit 2D DRAG beta map: {exc}",
            data={},
        )

    fig = go.Figure()
    fig.add_surface(
        x=X,
        y=Y,
        z=residual_grid,
        colorbar=dict(title=zlabel),
        name="Residual",
    )
    fig.add_scatter3d(
        x=[beta0],
        y=[beta1],
        z=[metric],
        mode="markers",
        marker=dict(size=5),
        name="Joint root",
    )
    fig.update_layout(
        title=f"{title} : {q0_label}-{q1_label}",
        scene=dict(
            xaxis_title=f"{xlabel} {q0_label}",
            yaxis_title=f"{ylabel} {q1_label}",
            zaxis_title=zlabel,
        ),
        width=600,
        height=500,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if plot:
        fig.show(config=fitting._plotly_config(f"drag_beta_map_{q0_label}_{q1_label}"))

    return FitResult(
        status=FitStatus.SUCCESS,
        message="2D DRAG beta map fitting successful.",
        data={
            "beta": {
                q0_label: beta0,
                q1_label: beta1,
            },
            "metric": metric,
            "coefficients": coeffs,
            "terms": terms,
            "surfaces": surfaces,
            "fig": fig,
        },
        figure=fig,
    )


def calibrate_freq_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"],
    duration: float | None = None,
    ramptime: float | None = None,
    detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
    n_rotations: int = 1,
    r2_threshold: float = 0.5,
    use_stored_amplitude: bool = False,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_freq_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    
    exp.validate_rabi_params(targets)

    def calibrate(targets: list):
        if pulse_type == "hpi":
            pulse = FlatTop(
                duration=duration if duration is not None else HPI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else HPI_RAMPTIME,
            )
            area = pulse.real.sum() * pulse.SAMPLING_PERIOD
            rabi_rate = 0.25 / area
        elif pulse_type == "pi":
            pulse = FlatTop(
                duration=duration if duration is not None else PI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else PI_RAMPTIME,
            )
            area = pulse.real.sum() * pulse.SAMPLING_PERIOD
            rabi_rate = 0.5 / area
        else:
            raise ValueError("Invalid pulse type.")
        
        n_per_rotation = 2 if pulse_type == "pi" else 4

        if not use_stored_amplitude:
            result = exp.calibrate_default_pulse(
                targets=targets,
                pulse_type=pulse_type,
                duration=duration,
                ramptime=ramptime,
                plot=False,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
            rough_cal_amplitude = {target: data.calib_value for target, data in result.data.items()}
        else:
            rough_cal_amplitude = {target: exp.calc_control_amplitude(target, rabi_rate) for target in targets}

        sequence={target: pulse.scaled(rough_cal_amplitude[target]).repeated(n_per_rotation * n_rotations) for target in targets}

        P = np.zeros((len(detuning_range), len(detuning_range)))

        for i, dq0 in enumerate(detuning_range):
            for j, dq1 in enumerate(detuning_range):
                frequencies = {
                    target: exp.targets[target].frequency + detuning 
                    for target, detuning in zip(targets, (dq0, dq1))
                }

                result = exp.measure(
                    sequence=sequence,
                    frequencies=frequencies,
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )

                P[i, j] = result.get_probabilities(targets).get("00", 0.0)

        fit_result = fit_2d_detuning_map(
            targets=targets,
            x_detuning_range=detuning_range,
            y_detuning_range=detuning_range,
            data=P,
            plot=plot,
        )

        r2 = fit_result["r2"]
        if r2 > r2_threshold:
            for target, data in fit_result.data["detuning"].items():
                print(f"{target}:{data}")
            if pulse_type == "hpi":
                for target in targets:
                    exp.calib_note.update_hpi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": rough_cal_amplitude[target],
                            "tau": pulse.tau,
                        },
                    )
            elif pulse_type == "pi":
                for target in targets:
                    exp.calib_note.update_pi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": rough_cal_amplitude[target],
                            "tau": pulse.tau,
                        },
                    )
        else:
            print(f"Error: R² value is too low ({r2:.3f})")
            print(f"Calibration data not stored for {targets}.")

        return fit_result
    
    result = calibrate(targets)

    return Result(data=result.data, figure=result.figure)

def calibrate_ampl_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"],
    fit_freq: dict[str, float] | None = None,
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int = 20,
    n_rotations: int = 1,
    r2_threshold: float = 0.5,
    use_stored_amplitude: bool = False,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_ampl_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    
    exp.validate_rabi_params(targets)

    if fit_freq is None:
        fit_freq = {target : 0 for target in targets}

    if not use_stored_amplitude:
        result = exp.calibrate_default_pulse(
            targets=targets,
            pulse_type=pulse_type,
            duration=duration,
            ramptime=ramptime,
            plot=False,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        rough_cal_amplitude = {target: data.calib_value for target, data in result.data.items()}

    def calibrate(targets: list):
        if pulse_type == "hpi":
            pulse = FlatTop(
                duration=duration if duration is not None else HPI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else HPI_RAMPTIME,
            )
            area = pulse.real.sum() * pulse.SAMPLING_PERIOD
            rabi_rate = 0.25 / area
        elif pulse_type == "pi":
            pulse = FlatTop(
                duration=duration if duration is not None else PI_DURATION,
                amplitude=1,
                tau=ramptime if ramptime is not None else PI_RAMPTIME,
            )
            area = pulse.real.sum() * pulse.SAMPLING_PERIOD
            rabi_rate = 0.5 / area    
        else:
            raise ValueError("Invalid pulse type.")
        
        if use_stored_amplitude:
            ampl = {target: exp.calc_control_amplitude(target, rabi_rate) for target in targets}
        else:
            ampl = rough_cal_amplitude

        ampl_min: dict[str, float] = {}
        ampl_max: dict[str, float] = {}
        ampl_range: dict[str, np.ndarray] = {}

        for target in targets:
            base_ampl = ampl[target]
            delta = 0.5 / n_rotations

            a_min = base_ampl * (1 - delta)
            a_max = base_ampl * (1 + delta)

            a_min = np.clip(a_min, 0.0, 1.0)
            a_max = np.clip(a_max, 0.0, 1.0)

            if a_min == a_max:
                a_min, a_max = 0.0, 1.0

            ampl_min[target] = a_min
            ampl_max[target] = a_max
            ampl_range[target] = np.linspace(a_min, a_max, n_points)

        n_per_rotation = 2 if pulse_type == "pi" else 4

        q0, q1 = targets
        P = np.zeros((len(ampl_range[q0]), len(ampl_range[q1])))

        frequencies = {target: exp.targets[target].frequency + fit_freq[target] for target in targets}
        for i, dq0 in enumerate(ampl_range[targets[0]]):
            for j, dq1 in enumerate(ampl_range[targets[1]]):
                sequence={
                    q0: pulse.scaled(dq0).repeated(n_per_rotation * n_rotations),
                    q1: pulse.scaled(dq1).repeated(n_per_rotation * n_rotations),
                }
                result = exp.measure(
                    sequence=sequence,
                    frequencies=frequencies,
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    plot=False,
                )

                P[i, j] = result.get_probabilities(targets).get("00", 0.0)

        fit_result = fit_2d_ampl_map(
            targets=targets,
            x_amplitude_range=ampl_range[q0],
            y_amplitude_range=ampl_range[q1],
            data=P,
            plot=plot,
        )

        r2 = fit_result["r2"]
        if r2 > r2_threshold:
            if pulse_type == "hpi":
                for target in targets:
                    exp.calib_note.update_hpi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"][target],
                            "tau": pulse.tau,
                        },
                    )
            elif pulse_type == "pi":
                for target in targets:
                    exp.calib_note.update_pi_param(
                        target,
                        {
                            "target": target,
                            "duration": pulse.duration,
                            "amplitude": fit_result["amplitude"][target],
                            "tau": pulse.tau,
                        },
                    )
        else:
            print(f"Error: R² value is too low ({r2:.3f})")
            print(f"Calibration data not stored for {targets}.")

        return fit_result
    
    result = calibrate(targets)

    return Result(data=result.data, figure=result.figure)


def calibrate_drag_ampl_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"],
    duration: float | None = None,
    n_points: int = 20,
    n_rotations: int = 4,
    r2_threshold: float = 0.5,
    drag_coeff: float = DRAG_COEFF,
    use_stored_amplitude: bool = False,
    use_stored_beta: bool = False,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    """Calibrate simultaneous DRAG pulse amplitudes for two muxed targets."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_drag_ampl_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    exp.validate_rabi_params(targets)
    sampling_period_ns = _measurement_sampling_period_ns(exp)
    repetitions = (2 if pulse_type == "pi" else 4) * n_rotations

    pulses: dict[str, Drag] = {}
    betas: dict[str, float] = {}
    amplitudes: dict[str, float] = {}
    amplitude_ranges: dict[str, NDArray[np.float64]] = {}
    for target in targets:
        beta = _initial_drag_beta(
            exp,
            target,
            pulse_type,
            drag_coeff=drag_coeff,
            use_stored_beta=use_stored_beta,
        )
        pulse = Drag(
            duration=_drag_duration(pulse_type, duration),
            amplitude=1,
            beta=beta,
        )
        amplitude = _initial_drag_amplitude(
            exp,
            target,
            pulse_type,
            pulse=pulse,
            sampling_period_ns=sampling_period_ns,
            use_stored_amplitude=use_stored_amplitude,
        )
        pulses[target] = pulse
        betas[target] = beta
        amplitudes[target] = amplitude
        amplitude_ranges[target] = _amplitude_sweep_range(
            amplitude,
            n_points=n_points,
            n_rotations=n_rotations,
        )

    q0, q1 = targets
    P = np.zeros((len(amplitude_ranges[q0]), len(amplitude_ranges[q1])))
    for i, a0 in enumerate(amplitude_ranges[q0]):
        for j, a1 in enumerate(amplitude_ranges[q1]):
            sequence = {
                q0: pulses[q0].scaled(a0).repeated(repetitions),
                q1: pulses[q1].scaled(a1).repeated(repetitions),
            }
            result = exp.measure(
                sequence=sequence,
                mode="single",
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )
            P[i, j] = result.get_probabilities(targets).get("00", 0.0)

    fit_result = fit_2d_ampl_map(
        targets=targets,
        x_amplitude_range=amplitude_ranges[q0],
        y_amplitude_range=amplitude_ranges[q1],
        data=P,
        plot=plot,
        title=f"DRAG {pulse_type} amplitude map",
    )

    data = dict(fit_result.data)
    data["initial_amplitude"] = amplitudes
    data["beta"] = betas
    data["amplitude_range"] = amplitude_ranges
    data["raw_probability"] = P

    r2 = float(fit_result.data.get("r2", np.nan))
    if np.isfinite(r2) and r2 > r2_threshold:
        for target in targets:
            _update_drag_param(
                exp,
                target,
                pulse_type,
                duration=float(pulses[target].duration),
                amplitude=float(fit_result["amplitude"][target]),
                beta=betas[target],
            )
    else:
        print(f"Error: R² value is too low ({r2:.3f})")
        print(f"Calibration data not stored for {targets}.")

    return Result(data=data, figure=fit_result.figure)


def calibrate_drag_beta_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"] = "hpi",
    beta_range: ArrayLike | None = None,
    duration: float | None = None,
    n_turns: int = 1,
    degree: int = 3,
    drag_coeff: float = DRAG_COEFF,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    """Calibrate simultaneous DRAG beta values for two muxed targets."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_drag_beta_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    exp.validate_rabi_params(targets)
    if beta_range is None:
        beta_range = np.linspace(-2.0, 2.0, 20)

    current_params: dict[str, Mapping[str, Any]] = {}
    beta_ranges: dict[str, NDArray[np.float64]] = {}
    for target in targets:
        param = _drag_param(exp, target, pulse_type)
        if param is None:
            raise ValueError("DRAG parameters are not stored.")
        current_params[target] = param
        beta_center = float(
            param.get(
                "beta",
                -drag_coeff / exp.ctx.qubits[target].alpha,
            )
        )
        beta_ranges[target] = np.asarray(beta_range, dtype=np.float64) + beta_center

    q0, q1 = targets
    signals = np.zeros((2, len(beta_ranges[q0]), len(beta_ranges[q1])))

    def add_beta_sequence(ps: PulseSchedule, target: str, beta: float) -> None:
        param = current_params[target]
        drag_duration = float(duration if duration is not None else param["duration"])
        drag_amplitude = float(param["amplitude"])
        y90m = exp.pulse.get_hpi_pulse(target).shifted(-np.pi / 2)
        if pulse_type == "hpi":
            x90p = Drag(
                duration=drag_duration,
                amplitude=drag_amplitude,
                beta=beta,
            )
            x90m = x90p.scaled(-1)
            ps.add(target, x90p)
            for _ in range(n_turns):
                ps.add(target, x90m)
                ps.add(target, x90p)
            ps.add(target, y90m)
        elif pulse_type == "pi":
            x180p = Drag(
                duration=drag_duration,
                amplitude=drag_amplitude,
                beta=beta,
            )
            x180m = x180p.scaled(-1)
            for _ in range(n_turns):
                ps.add(target, x180p)
                ps.add(target, x180m)
            ps.add(target, y90m)
        else:
            raise ValueError("Invalid pulse type.")

    for i, beta0 in enumerate(beta_ranges[q0]):
        for j, beta1 in enumerate(beta_ranges[q1]):
            with PulseSchedule(targets) as sequence:
                add_beta_sequence(sequence, q0, float(beta0))
                add_beta_sequence(sequence, q1, float(beta1))
            result = exp.measure(
                sequence=sequence,
                mode="avg",
                n_shots=n_shots,
                shot_interval=shot_interval,
                plot=False,
            )
            for index, target in enumerate(targets):
                iq = result.data[target].kerneled
                signals[index, i, j] = exp.pulse.rabi_params[target].normalize(iq)

    fit_result = fit_2d_drag_beta_map(
        targets=targets,
        x_beta_range=beta_ranges[q0],
        y_beta_range=beta_ranges[q1],
        data=signals,
        degree=degree,
        plot=plot,
        title=f"DRAG {pulse_type} beta map",
    )

    data = dict(fit_result.data)
    data["beta_range"] = beta_ranges
    data["raw_signal"] = signals

    if fit_result.status == FitStatus.SUCCESS:
        for target in targets:
            param = current_params[target]
            _update_drag_param(
                exp,
                target,
                pulse_type,
                duration=float(duration if duration is not None else param["duration"]),
                amplitude=float(param["amplitude"]),
                beta=float(fit_result["beta"][target]),
            )

    return Result(data=data, figure=fit_result.figure)


def calibrate_drag_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"] = "hpi",
    n_points: int = 20,
    n_rotations: int = 4,
    n_turns: int = 1,
    n_iterations: int = 2,
    degree: int = 3,
    r2_threshold: float = 0.5,
    calibrate_beta: bool = True,
    beta_range: ArrayLike | None = None,
    duration: float | None = None,
    drag_coeff: float = DRAG_COEFF,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    """Calibrate simultaneous DRAG pulses for two muxed targets."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_drag_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")

    amplitude: Result | None = None
    beta: Result | dict[str, float] | None = None
    for i in range(n_iterations):
        print(f"\nIteration {i + 1}/{n_iterations}")
        amplitude = calibrate_drag_ampl_mux_pulse(
            exp,
            targets=targets,
            pulse_type=pulse_type,
            duration=duration,
            n_points=n_points,
            n_rotations=1 if i == 0 else n_rotations,
            r2_threshold=r2_threshold,
            drag_coeff=drag_coeff,
            use_stored_amplitude=i != 0,
            use_stored_beta=i != 0,
            plot=plot,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        if calibrate_beta:
            beta = calibrate_drag_beta_mux_pulse(
                exp,
                targets=targets,
                pulse_type=pulse_type,
                beta_range=beta_range,
                duration=duration,
                n_turns=n_turns,
                degree=degree,
                drag_coeff=drag_coeff,
                plot=plot,
                n_shots=n_shots,
                shot_interval=shot_interval,
            )
        else:
            beta = {
                target: _initial_drag_beta(
                    exp,
                    target,
                    pulse_type,
                    drag_coeff=drag_coeff,
                    use_stored_beta=True,
                )
                for target in targets
            }

    return Result(
        data={
            "amplitude": amplitude["amplitude"] if amplitude is not None else {},
            "beta": beta["beta"] if isinstance(beta, Result) else beta,
            "amplitude_result": amplitude,
            "beta_result": beta,
        }
    )


def calibrate_drag_hpi_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    **kwargs: object,
) -> Result:
    """Calibrate simultaneous DRAG half-pi pulses for two muxed targets."""
    return calibrate_drag_mux_pulse(
        exp,
        targets=targets,
        pulse_type="hpi",
        **kwargs,
    )


def calibrate_drag_pi_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    **kwargs: object,
) -> Result:
    """Calibrate simultaneous DRAG pi pulses for two muxed targets."""
    return calibrate_drag_mux_pulse(
        exp,
        targets=targets,
        pulse_type="pi",
        **kwargs,
    )


def calibrate_mux_pulse(
    exp: Experiment,
    targets: Collection[str] | str | None = None,
    *,
    pulse_type: Literal["pi", "hpi"],
    detuning_range: ArrayLike = np.linspace(-0.01, 0.01, 21),
    duration: float | None = None,
    ramptime: float | None = None,
    n_points: int = 20,
    n_rotations: int = 1,
    n_iterations: int = 2,
    r2_threshold: float = 0.5,
    plot: bool = True,
    n_shots: int | None = CALIBRATION_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="calibrate_mux_pulse",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    targets = _require_mux_targets(exp, targets)
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")
    
    fit_freq: dict[str, float] = {target: 0.0 for target in targets}
    fit_ampl: dict[str, float] = {target: 0.0 for target in targets}

    for i in range(n_iterations):
        print(f"\nIteration {i + 1}/{n_iterations}")

        use_stored_amp_for_freq = (i != 0)

        freq_result = calibrate_freq_mux_pulse(
            exp,
            targets=targets,
            pulse_type=pulse_type,
            detuning_range=detuning_range,
            duration=duration,
            ramptime=ramptime,
            n_rotations=n_rotations,
            r2_threshold=r2_threshold,
            use_stored_amplitude=use_stored_amp_for_freq,
            plot=plot,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        fit_freq = freq_result.data["detuning"]

        ampl_result = calibrate_ampl_mux_pulse(
            exp,
            targets=targets,
            pulse_type=pulse_type,
            fit_freq=fit_freq,
            duration=duration,
            ramptime=ramptime,
            n_rotations=n_rotations,
            n_points=n_points,
            r2_threshold=r2_threshold,
            use_stored_amplitude=True,
            plot=plot,
            n_shots=n_shots,
            shot_interval=shot_interval,
        )
        fit_ampl = ampl_result.data["amplitude"]

    return Result(
        data = {
            "amplitude": fit_ampl,
            "detuning": fit_freq,
        }
    )


def ncopy_rb_sequence(
    exp: Experiment,
    targets: Collection[str] | str | None,
    *,
    n: int,
    x90: TargetMap[Waveform] | None = None,
    seed: int | None = None,
    sequence_type: Literal["even", "odd"] = "even",
    odd_target: str | int | None = None,
    final_state: str | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
) -> PulseSchedule:
    """Build one ncopy RB sequence for muxed single-qubit targets."""
    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("At least one target is required.")
    if final_state is None:
        final_state = "0" * len(target_list)

    x90_map = _resolve_x90_map(exp, target_list, x90)
    odd_label: str | None = None
    if sequence_type == "odd":
        if odd_target is None:
            raise ValueError("odd_target is required for odd ncopy RB sequences.")
        odd_label = target_list[odd_target] if isinstance(odd_target, int) else odd_target
        if odd_label not in target_list:
            raise ValueError(f"Invalid odd_target: {odd_target}")
    elif sequence_type != "even":
        raise ValueError("sequence_type must be 'even' or 'odd'.")

    with PulseSchedule(target_list) as ps:
        if odd_label is not None:
            ps.add(odd_label, x90_map[odd_label].repeated(2))
            ps.barrier(target_list)

        for target in target_list:
            rb_sequence = exp.benchmarking_service.rb_sequence_1q(
                target=target,
                n=n,
                x90=x90_map[target],
                interleaved_waveform=interleaved_waveform.get(target)
                if interleaved_waveform
                else None,
                interleaved_clifford=interleaved_clifford,
                seed=seed,
            )
            ps.add(target, rb_sequence)

        ps.barrier(target_list)
        if odd_label is not None:
            ps.add(odd_label, x90_map[odd_label].repeated(2))
            ps.barrier(target_list)

        _apply_final_state(ps, target_list, x90_map, final_state)
        ps.barrier(target_list)

    return ps


def _ncopy_sequence_keys(targets: Sequence[str]) -> list[str]:
    return ["even", *[f"odd_{index}" for index in range(len(targets))]]


def _arrayify_probability_data(
    probabilities: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
) -> dict[str, dict[str, dict[str, NDArray[np.float64]]]]:
    return {
        final_state: {
            seq_key: {
                label: np.asarray(values, dtype=np.float64)
                for label, values in state_probs.items()
            }
            for seq_key, state_probs in seq_probs.items()
        }
        for final_state, seq_probs in probabilities.items()
    }


def _arrayify_curve_data(
    curves: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, dict[str, NDArray[np.float64]]]:
    return {
        final_state: {
            seq_key: np.asarray(values, dtype=np.float64)
            for seq_key, values in seq_curves.items()
        }
        for final_state, seq_curves in curves.items()
    }


def _arrayify_per_target_curve_data(
    curves: Mapping[str, Mapping[str, Mapping[str, list[float]]]],
) -> dict[str, dict[str, dict[str, NDArray[np.float64]]]]:
    return {
        final_state: {
            seq_key: {
                target: np.asarray(values, dtype=np.float64)
                for target, values in target_curves.items()
            }
            for seq_key, target_curves in seq_curves.items()
        }
        for final_state, seq_curves in curves.items()
    }


def ncopy_rb_experiment(
    exp: Experiment,
    targets: Collection[str] | str | None,
    *,
    n_cliffords_range: ArrayLike,
    x90: TargetMap[Waveform] | None = None,
    seed: int | None = None,
    final_states: Collection[str] | str | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    return_raw: bool = False,
    reset_awg_and_capunits: bool = True,
    n_shots: int | None = DEFAULT_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    """Run one-seed ncopy randomized benchmarking and keep all state probabilities."""
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ncopy_rb_experiment",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL

    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("At least one target is required.")
    for target in target_list:
        target_object = exp.ctx.experiment_system.get_target(target)
        if target_object.is_cr:
            raise ValueError(f"`{target}` is not a 1Q target.")

    n_cliffords = np.asarray(n_cliffords_range, dtype=int)
    if final_states is None:
        final_state_list = ["0" * len(target_list)]
    elif isinstance(final_states, str):
        final_state_list = [final_states]
    else:
        final_state_list = list(final_states)
    for final_state in final_state_list:
        if len(final_state) != len(target_list):
            raise ValueError("Each final state length must match targets.")

    sequence_keys = _ncopy_sequence_keys(target_list)
    probabilities: dict[str, dict[str, dict[str, list[float]]]] = {
        final_state: {seq_key: defaultdict(list) for seq_key in sequence_keys}
        for final_state in final_state_list
    }
    leakage: dict[str, dict[str, list[float]]] = {
        final_state: {seq_key: [] for seq_key in sequence_keys}
        for final_state in final_state_list
    }
    per_target_leakage: dict[str, dict[str, dict[str, list[float]]]] = {
        final_state: {
            seq_key: {target: [] for target in target_list}
            for seq_key in sequence_keys
        }
        for final_state in final_state_list
    }
    raw_result: dict[str, dict[str, list[Any]]] | None = None
    if return_raw:
        raw_result = {
            final_state: {seq_key: [] for seq_key in sequence_keys}
            for final_state in final_state_list
        }

    for n_clifford in n_cliffords:
        for final_state in final_state_list:
            for seq_key in sequence_keys:
                if seq_key == "even":
                    sequence = ncopy_rb_sequence(
                        exp,
                        target_list,
                        n=int(n_clifford),
                        x90=x90,
                        seed=seed,
                        sequence_type="even",
                        final_state=final_state,
                        interleaved_clifford=interleaved_clifford,
                        interleaved_waveform=interleaved_waveform,
                    )
                else:
                    odd_index = int(seq_key.removeprefix("odd_"))
                    sequence = ncopy_rb_sequence(
                        exp,
                        target_list,
                        n=int(n_clifford),
                        x90=x90,
                        seed=seed,
                        sequence_type="odd",
                        odd_target=odd_index,
                        final_state=final_state,
                        interleaved_clifford=interleaved_clifford,
                        interleaved_waveform=interleaved_waveform,
                    )
                result = exp.measure(
                    sequence=sequence,
                    mode="single",
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    reset_awg_and_capunits=reset_awg_and_capunits,
                    plot=False,
                )
                probs = _complete_probabilities(result, target_list)
                for label, value in probs.items():
                    probabilities[final_state][seq_key][label].append(value)
                leakage[final_state][seq_key].append(_leakage_probability(probs))
                target_leakage = _per_target_leakage(probs, target_list)
                for target, value in target_leakage.items():
                    per_target_leakage[final_state][seq_key][target].append(value)
                if raw_result is not None:
                    raw_result[final_state][seq_key].append(result)

    data: dict[str, object] = {
        "n_cliffords": n_cliffords,
        "seed": seed,
        "targets": target_list,
        "final_states": final_state_list,
        "probabilities": _arrayify_probability_data(probabilities),
        "leakage": _arrayify_curve_data(leakage),
        "per_target_leakage": _arrayify_per_target_curve_data(per_target_leakage),
    }
    if raw_result is not None:
        data["raw_result"] = raw_result

    return Result(data=data)


def _aggregate_probability_trials(
    trial_results: Sequence[Result],
    final_states: Sequence[str],
    sequence_keys: Sequence[str],
    n_cliffords: NDArray[np.int_],
) -> tuple[
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
]:
    mean: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    std: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    zeros = np.zeros_like(n_cliffords, dtype=np.float64)
    for final_state in final_states:
        mean[final_state] = {}
        std[final_state] = {}
        for seq_key in sequence_keys:
            labels: set[str] = set()
            for result in trial_results:
                labels.update(result["probabilities"][final_state][seq_key])
            mean[final_state][seq_key] = {}
            std[final_state][seq_key] = {}
            for label in sorted(labels):
                curves = [
                    result["probabilities"][final_state][seq_key].get(label, zeros)
                    for result in trial_results
                ]
                values = np.asarray(curves, dtype=np.float64)
                mean[final_state][seq_key][label] = np.mean(values, axis=0)
                std[final_state][seq_key][label] = np.std(values, axis=0)
    return mean, std


def _aggregate_curve_trials(
    trial_results: Sequence[Result],
    key: str,
    final_states: Sequence[str],
    sequence_keys: Sequence[str],
) -> tuple[
    dict[str, dict[str, NDArray[np.float64]]],
    dict[str, dict[str, NDArray[np.float64]]],
]:
    mean: dict[str, dict[str, NDArray[np.float64]]] = {}
    std: dict[str, dict[str, NDArray[np.float64]]] = {}
    for final_state in final_states:
        mean[final_state] = {}
        std[final_state] = {}
        for seq_key in sequence_keys:
            values = np.asarray(
                [result[key][final_state][seq_key] for result in trial_results],
                dtype=np.float64,
            )
            mean[final_state][seq_key] = np.mean(values, axis=0)
            std[final_state][seq_key] = np.std(values, axis=0)
    return mean, std


def _aggregate_per_target_leakage_trials(
    trial_results: Sequence[Result],
    targets: Sequence[str],
    final_states: Sequence[str],
    sequence_keys: Sequence[str],
) -> tuple[
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
    dict[str, dict[str, dict[str, NDArray[np.float64]]]],
]:
    mean: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    std: dict[str, dict[str, dict[str, NDArray[np.float64]]]] = {}
    for final_state in final_states:
        mean[final_state] = {}
        std[final_state] = {}
        for seq_key in sequence_keys:
            mean[final_state][seq_key] = {}
            std[final_state][seq_key] = {}
            for target in targets:
                values = np.asarray(
                    [
                        result["per_target_leakage"][final_state][seq_key][target]
                        for result in trial_results
                    ],
                    dtype=np.float64,
                )
                mean[final_state][seq_key][target] = np.mean(values, axis=0)
                std[final_state][seq_key][target] = np.std(values, axis=0)
    return mean, std


def _plot_ncopy_results(
    *,
    targets: Sequence[str],
    n_cliffords: NDArray[np.int_],
    final_states: Sequence[str],
    sequence_keys: Sequence[str],
    leakage_mean: Mapping[str, Mapping[str, NDArray[np.float64]]],
    leakage_std: Mapping[str, Mapping[str, NDArray[np.float64]]],
    probability_mean: Mapping[str, Mapping[str, Mapping[str, NDArray[np.float64]]]],
    probability_std: Mapping[str, Mapping[str, Mapping[str, NDArray[np.float64]]]],
    plot_metric: Literal["leakage", "ground_probability"],
    xaxis_type: Literal["linear", "log"],
    plot: bool,
) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}
    ground_label = "0" * len(targets)
    for final_state in final_states:
        for target_index, target in enumerate(targets):
            odd_key = f"odd_{target_index}"
            if odd_key not in sequence_keys:
                continue
            fig = go.Figure()
            for seq_key, name in (("even", "even"), (odd_key, "odd")):
                if plot_metric == "leakage":
                    y = leakage_mean[final_state][seq_key]
                    yerr = leakage_std[final_state][seq_key]
                    ylabel = "Leakage probability"
                else:
                    y = probability_mean[final_state][seq_key].get(
                        ground_label,
                        np.zeros_like(n_cliffords, dtype=np.float64),
                    )
                    yerr = probability_std[final_state][seq_key].get(
                        ground_label,
                        np.zeros_like(n_cliffords, dtype=np.float64),
                    )
                    ylabel = f"P({ground_label})"
                fig.add_trace(
                    go.Scatter(
                        x=n_cliffords,
                        y=y,
                        error_y=dict(type="data", array=yerr),
                        mode="markers",
                        name=name,
                    )
                )
            fig.update_layout(
                title=f"ncopy RB : {target} final={final_state}",
                xaxis_title="Number of Cliffords",
                yaxis_title=ylabel,
                xaxis_type=xaxis_type,
                yaxis_type="linear",
            )
            if plot:
                fig.show(
                    config=fitting._plotly_config(
                        f"ncopy_rb_{target}_{final_state}_{plot_metric}"
                    )
                )
            figures[f"{target}_{final_state}_{plot_metric}"] = fig
    return figures


def _default_n_cliffords_range(max_n_cliffords: int) -> NDArray[np.int_]:
    values: list[int] = []
    idx = 0
    while True:
        n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
        if n_clifford > max_n_cliffords:
            break
        values.append(n_clifford)
        idx += 1
    return np.asarray(values, dtype=int)


def _ncopy_metric_curve(
    *,
    targets: Sequence[str],
    target_index: int,
    final_state: str,
    leakage_values: Mapping[str, Mapping[str, NDArray[np.float64]]],
    probability_values: Mapping[
        str,
        Mapping[str, Mapping[str, NDArray[np.float64]]],
    ],
    plot_metric: Literal["leakage", "ground_probability"],
) -> NDArray[np.float64]:
    seq_key = f"odd_{target_index}"
    if plot_metric == "leakage":
        return leakage_values[final_state][seq_key]
    ground_label = "0" * len(targets)
    fallback = np.zeros_like(leakage_values[final_state][seq_key], dtype=np.float64)
    return probability_values[final_state][seq_key].get(
        ground_label,
        fallback,
    )


def ncopy_randomized_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    in_parallel: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    final_states: Collection[str] | str | None = None,
    interleaved_clifford: Clifford | None = None,
    interleaved_waveform: TargetMap[Waveform] | None = None,
    return_raw: bool = False,
    reset_awg_and_capunits: bool = True,
    plot_metric: Literal["leakage", "ground_probability"] = "leakage",
    plot: bool | None = None,
    save_image: bool | None = None,
    n_shots: int | None = DEFAULT_SHOTS,
    shot_interval: float | None = DEFAULT_INTERVAL,
    **deprecated_options: object,
) -> Result:
    """
    Run ncopy randomized benchmarking for muxed single-qubit targets.

    With three-state classifiers, leave `final_states` unset and use the returned
    `leakage` / `per_target_leakage` curves.  For two-state compatibility with
    older notebooks, pass computational `final_states` such as
    `("00", "01", "10", "11")` and inspect `probabilities`.
    """
    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="ncopy_randomized_benchmarking",
    )
    if deprecated_options:
        unexpected = ", ".join(sorted(deprecated_options))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS
    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q
    if in_parallel is None:
        in_parallel = True
    if xaxis_type is None:
        xaxis_type = "linear"
    if plot is None:
        plot = True
    if save_image is None:
        save_image = True

    target_list = _normalize_targets(exp, targets)
    if not target_list:
        raise ValueError("At least one target is required.")
    if n_cliffords_range is None:
        n_cliffords = _default_n_cliffords_range(max_n_cliffords)
    else:
        n_cliffords = np.asarray(n_cliffords_range, dtype=int)
    if seeds is None:
        seed_list = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seed_list = np.asarray(seeds, dtype=int)
        if len(seed_list) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    target_groups = (
        [target_list] if in_parallel else [[target] for target in target_list]
    )

    return_data: dict[str, dict[str, object]] = {}
    figures: dict[str, go.Figure] = {}
    for target_group in target_groups:
        if final_states is None:
            final_state_list = ["0" * len(target_group)]
        elif isinstance(final_states, str):
            final_state_list = [final_states]
        else:
            final_state_list = list(final_states)
        sequence_keys = _ncopy_sequence_keys(target_group)

        trial_results: list[Result] = []
        for seed in seed_list:
            trial_results.append(
                ncopy_rb_experiment(
                    exp,
                    target_group,
                    n_cliffords_range=n_cliffords,
                    x90=x90,
                    seed=int(seed),
                    final_states=final_state_list,
                    interleaved_clifford=interleaved_clifford,
                    interleaved_waveform=interleaved_waveform,
                    return_raw=return_raw,
                    reset_awg_and_capunits=reset_awg_and_capunits,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                )
            )

        probability_mean, probability_std = _aggregate_probability_trials(
            trial_results,
            final_state_list,
            sequence_keys,
            n_cliffords,
        )
        leakage_mean, leakage_std = _aggregate_curve_trials(
            trial_results,
            "leakage",
            final_state_list,
            sequence_keys,
        )
        per_target_leakage_mean, per_target_leakage_std = (
            _aggregate_per_target_leakage_trials(
                trial_results,
                target_group,
                final_state_list,
                sequence_keys,
            )
        )
        group_figures = _plot_ncopy_results(
            targets=target_group,
            n_cliffords=n_cliffords,
            final_states=final_state_list,
            sequence_keys=sequence_keys,
            leakage_mean=leakage_mean,
            leakage_std=leakage_std,
            probability_mean=probability_mean,
            probability_std=probability_std,
            plot_metric=plot_metric,
            xaxis_type=xaxis_type,
            plot=plot,
        )
        figures.update(group_figures)

        default_final_state = final_state_list[0]
        ground_label = "0" * len(target_group)
        if plot_metric == "leakage":
            even_mean = leakage_mean[default_final_state]["even"]
            even_std = leakage_std[default_final_state]["even"]
        else:
            even_mean = probability_mean[default_final_state]["even"].get(
                ground_label,
                np.zeros_like(n_cliffords, dtype=np.float64),
            )
            even_std = probability_std[default_final_state]["even"].get(
                ground_label,
                np.zeros_like(n_cliffords, dtype=np.float64),
            )
        for target_index, target in enumerate(target_group):
            mean = _ncopy_metric_curve(
                targets=target_group,
                target_index=target_index,
                final_state=default_final_state,
                leakage_values=leakage_mean,
                probability_values=probability_mean,
                plot_metric=plot_metric,
            )
            std = _ncopy_metric_curve(
                targets=target_group,
                target_index=target_index,
                final_state=default_final_state,
                leakage_values=leakage_std,
                probability_values=probability_std,
                plot_metric=plot_metric,
            )
            return_data[target] = {
                "n_cliffords": n_cliffords,
                "mean": mean,
                "std": std if n_trials > 1 else None,
                "even_mean": even_mean,
                "even_std": even_std if n_trials > 1 else None,
                "seeds": np.asarray(seed_list, dtype=np.int64),
                "targets": target_group,
                "final_states": final_state_list,
                "plot_metric": plot_metric,
                "probability_mean": probability_mean,
                "probability_std": probability_std,
                "leakage_mean": leakage_mean,
                "leakage_std": leakage_std,
                "per_target_leakage_mean": per_target_leakage_mean,
                "per_target_leakage_std": per_target_leakage_std,
            }
            if return_raw:
                return_data[target]["raw_result"] = trial_results

    if save_image:
        for name, fig in figures.items():
            viz.save_figure(fig, name=f"ncopy_randomized_benchmarking_{name}")

    return Result(data=return_data, figures=figures or None)


def n_copy_rb_experiment(*args: Any, **kwargs: Any) -> Result:
    """Compatibility alias for `ncopy_rb_experiment`."""
    return ncopy_rb_experiment(*args, **kwargs)


def n_copy_randomized_benchmarking(*args: Any, **kwargs: Any) -> Result:
    """Compatibility alias for `ncopy_randomized_benchmarking`."""
    return ncopy_randomized_benchmarking(*args, **kwargs)

"""Contributed mux pulse calibration helpers."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from qubex.analysis import fitting
from qubex.analysis.fit_result import FitResult, FitStatus
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    CALIBRATION_SHOTS,
    DEFAULT_INTERVAL,
    HPI_DURATION,
    HPI_RAMPTIME,
    PI_DURATION,
    PI_RAMPTIME,
)
from qubex.experiment.models import Result
from qubex.pulse import FlatTop

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

    if targets is None:
        targets = _normalize_targets(exp, targets)

    if len(targets) != 2:
        raise ValueError("Mux pulse needs just 2 targets")
    
    rabi_params = exp.rabi_params
    exp.validate_rabi_params(rabi_params)

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
            rabi_rate = 0.25 / area
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

                P[i, j] = result.get_probabilities(targets).get("00")

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

    if targets is None:
        targets = _normalize_targets(exp, targets)

    if len(targets) != 2:
        raise ValueError("Mux pulse needs just 2 targets")
    
    rabi_params = exp.rabi_params
    exp.validate_rabi_params(rabi_params)

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

                P[i, j] = result.get_probabilities(targets).get("00")

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

    if targets is None:
        targets = _normalize_targets(exp, targets)

    if len(targets) != 2:
        raise ValueError("Mux pulse needs just 2 targets")
    
    fit_freq = dict[str, float]
    fit_ampl = dict[str, float]

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
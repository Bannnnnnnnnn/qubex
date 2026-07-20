"""Contributed JPA calibration workflow."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from qubex.backend.dc_voltage_controller import dc_voltage
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import PulseSchedule
from qubex.system import Mux

DEFAULT_COARSE_POINTS = 9
DEFAULT_DC_VOLTAGE_HALF_WIDTH = 1.0
DEFAULT_PUMP_FREQUENCY_HALF_WIDTH_GHZ = 0.25
DEFAULT_PUMP_AMPLITUDE_HALF_WIDTH = 0.5
DEFAULT_DC_VOLTAGE_LIMITS = (0.0, 4.0)
DEFAULT_N_SHOTS = 128
DEFAULT_FLATNESS_THRESHOLD: float | None = None
DEFAULT_MINIMUM_SCORE_GAIN = 1.05
DEFAULT_MAXIMUM_FLATNESS_RATIO = 1.1
DEFAULT_BASELINE_DC_VOLTAGE = 0.0
DEFAULT_FINE_POINTS = 9
DEFAULT_DC_VOLTAGE_TOLERANCE = 1e-3
DEFAULT_DC_SETTLE_TIME = 0.05
DEFAULT_DC_MAX_ATTEMPTS = 5
_RANK_TOLERANCE = 64.0 * np.finfo(np.float64).eps


class _DCVoltageSupply(Protocol):
    """Describe the ONS61797 operations used during calibration."""

    def set_voltage(self, *, channel: int, voltage: float) -> None:
        """Set one output voltage."""
        ...

    def get_voltage(self, *, channel: int) -> float:
        """Return one output voltage."""
        ...

    def on(self, *, channel: int) -> None:
        """Enable one output channel."""
        ...

    def get_output_state(self, *, channel: int) -> int:
        """Return one output state."""
        ...


class JPAConstraintError(ValueError):
    """Report JPA calibration data that cannot yield a safe operating point."""

    def __init__(
        self,
        message: str,
        *,
        stage: Literal["baseline", "coarse", "fine"] | None = None,
        diagnostics: Result,
    ) -> None:
        """Initialize the error and retain the complete scan diagnostics."""
        super().__init__(message)
        self.stage = stage
        self.diagnostics = diagnostics


class _NoValidPoint(ValueError):
    """Carry selection maps when no grid point satisfies every peer."""

    def __init__(
        self,
        *,
        score_gains: NDArray[np.float64],
        flatness_ratios: NDArray[np.float64],
        aggregate_score_map: NDArray[np.float64],
        aggregate_score_gain_map: NDArray[np.float64],
        measurement_valid_mask: NDArray[np.bool_],
        gain_valid_mask: NDArray[np.bool_],
        flatness_valid_mask: NDArray[np.bool_],
        valid_mask: NDArray[np.bool_],
    ) -> None:
        """Initialize the internal selection failure."""
        super().__init__("No grid point satisfies all peer constraints.")
        self.score_gains = score_gains
        self.flatness_ratios = flatness_ratios
        self.aggregate_score_map = aggregate_score_map
        self.aggregate_score_gain_map = aggregate_score_gain_map
        self.measurement_valid_mask = measurement_valid_mask
        self.gain_valid_mask = gain_valid_mask
        self.flatness_valid_mask = flatness_valid_mask
        self.valid_mask = valid_mask


class _ScanConstraintError(ValueError):
    """Carry one completed raw scan to the public workflow boundary."""

    def __init__(self, *, scan_payload: dict[str, object]) -> None:
        """Initialize the internal scan failure."""
        super().__init__("No grid point satisfies all peer constraints.")
        self.scan_payload = scan_payload


@dataclass(frozen=True)
class _MuxOptimum:
    """Store one selected point and its aggregate constraint maps."""

    index: tuple[int, ...]
    aggregate_score: float
    aggregate_score_gain: float
    peer_scores: NDArray[np.float64]
    peer_score_gains: NDArray[np.float64]
    peer_flatness: NDArray[np.float64]
    peer_flatness_ratios: NDArray[np.float64]
    aggregate_score_map: NDArray[np.float64]
    aggregate_score_gain_map: NDArray[np.float64]
    score_gains: NDArray[np.float64]
    flatness_ratios: NDArray[np.float64]
    measurement_valid_mask: NDArray[np.bool_]
    gain_valid_mask: NDArray[np.bool_]
    flatness_valid_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    on_boundary: bool


@dataclass(frozen=True)
class _JPAScan:
    """Store one three-dimensional JPA parameter scan."""

    dc_voltages: NDArray[np.float64]
    pump_frequencies: NDArray[np.float64]
    pump_amplitudes: NDArray[np.float64]
    scores: NDArray[np.float64]
    flatness: NDArray[np.float64]
    optimum: _MuxOptimum


def _normalize_iq_samples(samples: ArrayLike, *, name: str) -> NDArray[np.complex128]:
    """Return one finite, one-dimensional single-shot IQ array."""
    array = np.asarray(samples, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be a 1D single-shot IQ array.")
    if array.size < 2:
        raise ValueError(f"`{name}` must contain at least two shots.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"`{name}` must contain only finite IQ samples.")
    return array


def _state_separation_score(
    ground_samples: ArrayLike,
    excited_samples: ArrayLike,
) -> float:
    """Return noise-normalized g/e separation along the mean-state axis."""
    ground = _normalize_iq_samples(ground_samples, name="ground_samples")
    excited = _normalize_iq_samples(excited_samples, name="excited_samples")
    if ground.shape != excited.shape:
        raise ValueError("Ground and excited IQ arrays must have the same shape.")

    separation = np.mean(excited) - np.mean(ground)
    if abs(separation) == 0.0:
        return 0.0

    axis = separation / abs(separation)
    projected_ground = np.real(ground * np.conj(axis))
    projected_excited = np.real(excited * np.conj(axis))
    noise = math.sqrt(
        0.5 * (float(np.var(projected_ground)) + float(np.var(projected_excited)))
    )
    scale = max(
        abs(separation),
        float(np.max(np.abs(ground - np.mean(ground)))),
        float(np.max(np.abs(excited - np.mean(excited)))),
        np.finfo(np.float64).tiny,
    )
    noise_floor = np.finfo(np.float64).eps * scale
    return float(abs(separation) / max(noise, noise_floor))


def _covariance_flatness(samples: ArrayLike) -> float:
    """Return the rotation-invariant principal-axis ratio of one IQ cloud."""
    array = _normalize_iq_samples(samples, name="samples")
    centered = array - np.mean(array)
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        return math.nan

    real = np.real(centered) / scale
    imag = np.imag(centered) / scale
    covariance_xx = float(np.mean(real * real))
    covariance_yy = float(np.mean(imag * imag))
    covariance_xy = float(np.mean(real * imag))
    trace = covariance_xx + covariance_yy
    delta = math.hypot(covariance_xx - covariance_yy, 2.0 * covariance_xy)
    lambda_max = 0.5 * (trace + delta)
    lambda_min = max(0.0, 0.5 * (trace - delta))
    if lambda_max <= np.finfo(np.float64).eps:
        return math.nan
    if lambda_min <= _RANK_TOLERANCE * lambda_max:
        return math.inf
    return math.sqrt(lambda_max / lambda_min)


def _select_mux_optimum(
    scores: ArrayLike,
    flatness: ArrayLike,
    *,
    baseline_scores: ArrayLike,
    baseline_flatness: ArrayLike,
    minimum_score_gain: float,
    maximum_flatness_ratio: float,
    flatness_threshold: float | None,
) -> _MuxOptimum:
    """Select the best worst-peer gain relative to the JPA-off baseline."""
    score_array = np.asarray(scores, dtype=np.float64)
    flatness_array = np.asarray(flatness, dtype=np.float64)
    if score_array.shape != flatness_array.shape:
        raise ValueError("`scores` and `flatness` must have the same shape.")
    if score_array.ndim < 2 or score_array.shape[0] == 0:
        raise ValueError("Score arrays must contain a non-empty peer axis and grid.")
    if any(size == 0 for size in score_array.shape[1:]):
        raise ValueError("Score grid axes must not be empty.")

    baseline_score_array = np.asarray(baseline_scores, dtype=np.float64)
    baseline_flatness_array = np.asarray(baseline_flatness, dtype=np.float64)
    expected_baseline_shape = (score_array.shape[0],)
    if baseline_score_array.shape != expected_baseline_shape:
        raise ValueError("`baseline_scores` must contain one value per peer.")
    if baseline_flatness_array.shape != expected_baseline_shape:
        raise ValueError("`baseline_flatness` must contain one value per peer.")
    if not np.all(np.isfinite(baseline_score_array)) or np.any(
        baseline_score_array <= 0.0
    ):
        raise ValueError("JPA-off baseline scores must be finite and positive.")
    if not np.all(np.isfinite(baseline_flatness_array)) or np.any(
        baseline_flatness_array < 1.0
    ):
        raise ValueError("JPA-off baseline flatness must be finite and at least 1.")
    if not math.isfinite(minimum_score_gain) or minimum_score_gain < 1.0:
        raise ValueError("`minimum_score_gain` must be finite and at least 1.")
    if not math.isfinite(maximum_flatness_ratio) or maximum_flatness_ratio < 1.0:
        raise ValueError("`maximum_flatness_ratio` must be finite and at least 1.")
    if flatness_threshold is not None and (
        not math.isfinite(flatness_threshold) or flatness_threshold < 1.0
    ):
        raise ValueError(
            "`flatness_threshold` must be `None`, or finite and at least 1."
        )

    baseline_shape = (score_array.shape[0],) + (1,) * (score_array.ndim - 1)
    score_gains = score_array / baseline_score_array.reshape(baseline_shape)
    flatness_ratios = flatness_array / baseline_flatness_array.reshape(baseline_shape)

    peer_measurement_valid = (
        np.isfinite(score_array)
        & (score_array >= 0.0)
        & np.isfinite(score_gains)
        & np.isfinite(flatness_array)
        & (flatness_array >= 1.0)
        & np.isfinite(flatness_ratios)
    )
    peer_gain_valid = peer_measurement_valid & (score_gains >= minimum_score_gain)
    peer_flatness_valid = peer_measurement_valid & (
        flatness_ratios <= maximum_flatness_ratio
    )
    if flatness_threshold is not None:
        peer_flatness_valid &= flatness_array <= flatness_threshold
    measurement_valid_mask = np.all(peer_measurement_valid, axis=0)
    gain_valid_mask = np.all(peer_gain_valid, axis=0)
    flatness_valid_mask = np.all(peer_flatness_valid, axis=0)
    valid_mask = gain_valid_mask & flatness_valid_mask
    aggregate_score_map = np.min(score_array, axis=0)
    aggregate_score_gain_map = np.min(score_gains, axis=0)
    if not np.any(valid_mask):
        raise _NoValidPoint(
            score_gains=np.asarray(score_gains, dtype=np.float64),
            flatness_ratios=np.asarray(flatness_ratios, dtype=np.float64),
            aggregate_score_map=np.asarray(aggregate_score_map, dtype=np.float64),
            aggregate_score_gain_map=np.asarray(
                aggregate_score_gain_map,
                dtype=np.float64,
            ),
            measurement_valid_mask=np.asarray(
                measurement_valid_mask,
                dtype=np.bool_,
            ),
            gain_valid_mask=np.asarray(gain_valid_mask, dtype=np.bool_),
            flatness_valid_mask=np.asarray(flatness_valid_mask, dtype=np.bool_),
            valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        )

    ranking = np.where(valid_mask, aggregate_score_gain_map, -math.inf)
    flat_index = int(np.argmax(ranking))
    index = tuple(int(value) for value in np.unravel_index(flat_index, ranking.shape))
    on_boundary = any(
        axis_size > 1 and axis_index in (0, axis_size - 1)
        for axis_index, axis_size in zip(index, ranking.shape, strict=True)
    )
    peer_index = (slice(None), *index)
    return _MuxOptimum(
        index=index,
        aggregate_score=float(aggregate_score_map[index]),
        aggregate_score_gain=float(aggregate_score_gain_map[index]),
        peer_scores=np.asarray(score_array[peer_index], dtype=np.float64),
        peer_score_gains=np.asarray(score_gains[peer_index], dtype=np.float64),
        peer_flatness=np.asarray(flatness_array[peer_index], dtype=np.float64),
        peer_flatness_ratios=np.asarray(
            flatness_ratios[peer_index],
            dtype=np.float64,
        ),
        aggregate_score_map=np.asarray(aggregate_score_map, dtype=np.float64),
        aggregate_score_gain_map=np.asarray(
            aggregate_score_gain_map,
            dtype=np.float64,
        ),
        score_gains=np.asarray(score_gains, dtype=np.float64),
        flatness_ratios=np.asarray(flatness_ratios, dtype=np.float64),
        measurement_valid_mask=np.asarray(
            measurement_valid_mask,
            dtype=np.bool_,
        ),
        gain_valid_mask=np.asarray(gain_valid_mask, dtype=np.bool_),
        flatness_valid_mask=np.asarray(flatness_valid_mask, dtype=np.bool_),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        on_boundary=on_boundary,
    )


def _normalize_axis(
    values: ArrayLike,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.float64]:
    """Return one finite, strictly increasing calibration axis."""
    axis = np.asarray(values, dtype=np.float64)
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"`{name}` must be a non-empty 1D array.")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"`{name}` must contain only finite values.")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"`{name}` must be strictly increasing.")
    if minimum is not None and np.any(axis < minimum):
        raise ValueError(f"`{name}` must be at least {minimum}.")
    if maximum is not None and np.any(axis > maximum):
        raise ValueError(f"`{name}` must be at most {maximum}.")
    return axis


def _normalize_dc_voltage_limits(
    limits: tuple[float, float],
) -> tuple[float, float]:
    """Return one finite, increasing DC safety interval."""
    lower, upper = (float(value) for value in limits)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(
            "`dc_voltage_limits` must contain two finite, increasing values."
        )
    return lower, upper


def _centered_axis(
    center: float,
    *,
    half_width: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NDArray[np.float64]:
    """Build one clipped default sweep axis containing its configured value."""
    bounded_center = center
    if minimum is not None:
        bounded_center = max(bounded_center, minimum)
    if maximum is not None:
        bounded_center = min(bounded_center, maximum)
    lower = bounded_center - half_width
    upper = bounded_center + half_width
    if minimum is not None:
        lower = max(lower, minimum)
    if maximum is not None:
        upper = min(upper, maximum)
    if lower == upper:
        return np.asarray([lower], dtype=np.float64)
    if bounded_center <= lower or bounded_center >= upper:
        return np.linspace(lower, upper, DEFAULT_COARSE_POINTS, dtype=np.float64)
    left_points = DEFAULT_COARSE_POINTS // 2 + 1
    right_points = DEFAULT_COARSE_POINTS - left_points + 1
    return np.concatenate(
        [
            np.linspace(lower, bounded_center, left_points, dtype=np.float64),
            np.linspace(bounded_center, upper, right_points, dtype=np.float64)[1:],
        ]
    )


def _default_pump_frequency_axis(
    configured_frequency: float,
    *,
    configured_amplitude: float,
) -> NDArray[np.float64]:
    """Build a first-calibration or recentering pump-frequency axis."""
    if math.isclose(configured_amplitude, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return np.linspace(
            configured_frequency,
            configured_frequency + 2.0 * DEFAULT_PUMP_FREQUENCY_HALF_WIDTH_GHZ,
            DEFAULT_COARSE_POINTS,
            dtype=np.float64,
        )
    return _centered_axis(
        configured_frequency,
        half_width=DEFAULT_PUMP_FREQUENCY_HALF_WIDTH_GHZ,
        minimum=float(np.finfo(np.float64).tiny),
    )


def _refined_axis(
    axis: NDArray[np.float64],
    optimum_index: int,
    *,
    points: int,
) -> NDArray[np.float64]:
    """Build one fine axis bounded by the neighboring coarse points."""
    if axis.size == 1:
        return axis.copy()
    center = float(axis[optimum_index])
    lower = float(axis[max(0, optimum_index - 1)])
    upper = float(axis[min(axis.size - 1, optimum_index + 1)])
    if optimum_index == 0 or optimum_index == axis.size - 1:
        return np.linspace(lower, upper, points, dtype=np.float64)
    left_points = points // 2 + 1
    right_points = points - left_points + 1
    return np.concatenate(
        [
            np.linspace(lower, center, left_points, dtype=np.float64),
            np.linspace(center, upper, right_points, dtype=np.float64)[1:],
        ]
    )


def _resolve_evaluated_qubits(
    exp: Experiment, qubit: str
) -> tuple[str, Mux, tuple[str, ...]]:
    """Resolve one anchor qubit, its MUX, and all active valid peers."""
    anchor_qubit = exp.ctx.resolve_qubit_label(qubit)
    mux = exp.ctx.experiment_system.get_mux_by_qubit(anchor_qubit)
    active_qubits = set(exp.ctx.qubit_labels)
    evaluated_qubits = tuple(
        resonator.qubit
        for resonator in mux.resonators
        if resonator.is_valid and resonator.qubit in active_qubits
    )
    if anchor_qubit not in evaluated_qubits:
        raise ValueError(
            f"Anchor qubit `{anchor_qubit}` is not an active, valid qubit on `{mux.label}`."
        )
    return anchor_qubit, mux, evaluated_qubits


def _reset_jpa_hardware(
    exp: Experiment,
    evaluated_qubits: tuple[str, ...],
    *,
    mux: Mux,
) -> None:
    """Reset every box used by peer control, readout, capture, or JPA pump."""
    experiment_system = exp.ctx.experiment_system
    box_ids = {
        box.id for box in experiment_system.get_boxes_for_qubits(evaluated_qubits)
    }
    pump_target = experiment_system.get_target(mux.label)
    box_ids.add(pump_target.channel.port.box_id)
    exp.ctx.reset_awg_and_capunits(box_ids=box_ids)


def _acquire_state_samples(
    exp: Experiment,
    evaluated_qubits: tuple[str, ...],
    *,
    mux: Mux,
    pump_amplitude: float,
    excited: bool,
    n_shots: int,
    shot_interval: float | None,
    readout_amplitude: float | None,
    readout_duration: float | None,
) -> dict[str, NDArray[np.complex128]]:
    """Acquire one simultaneous ground- or excited-state IQ distribution."""
    read_labels = {
        qubit: exp.ctx.resolve_read_label(qubit) for qubit in evaluated_qubits
    }
    labels = [*read_labels.values(), mux.label]
    if excited:
        labels = [*evaluated_qubits, *labels]
    readout_pulses = {
        qubit: exp.pulse.readout(
            qubit,
            amplitude=readout_amplitude,
            duration=readout_duration,
        )
        for qubit in evaluated_qubits
    }
    pump_duration = max(float(pulse.duration) for pulse in readout_pulses.values())

    with PulseSchedule(labels) as schedule:
        if excited:
            for qubit in evaluated_qubits:
                schedule.add(qubit, exp.pulse.x180(qubit))
            schedule.barrier()
        for qubit, readout_pulse in readout_pulses.items():
            schedule.add(read_labels[qubit], readout_pulse)
        schedule.add(
            mux.label,
            exp.measurement.pulse_factory.pump_pulse(
                mux_index=mux.index,
                duration=pump_duration,
                amplitude=pump_amplitude,
            ),
        )

    result = exp.measurement.execute(
        schedule,
        n_shots=n_shots,
        shot_interval=shot_interval,
        shot_averaging=False,
        time_integration=True,
        state_classification=False,
        readout_amplification=False,
        final_measurement=False,
        plot=False,
    )
    samples: dict[str, NDArray[np.complex128]] = {}
    for qubit in evaluated_qubits:
        try:
            captures = result.data[qubit]
        except KeyError:
            raise RuntimeError(
                f"Measurement result does not contain `{qubit}`."
            ) from None
        if len(captures) != 1:
            raise RuntimeError(
                f"Expected one readout capture for `{qubit}`, got {len(captures)}."
            )
        samples[qubit] = _normalize_iq_samples(
            captures[0].kerneled,
            name=f"{qubit}_samples",
        )
    return samples


def _measure_jpa_point(
    exp: Experiment,
    evaluated_qubits: tuple[str, ...],
    *,
    mux: Mux,
    pump_amplitude: float,
    n_shots: int,
    shot_interval: float | None,
    readout_amplitude: float | None,
    readout_duration: float | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Measure g/e separation and worst-state IQ flatness at one JPA point."""
    ground_samples = _acquire_state_samples(
        exp,
        evaluated_qubits,
        mux=mux,
        pump_amplitude=pump_amplitude,
        excited=False,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
    )
    excited_samples = _acquire_state_samples(
        exp,
        evaluated_qubits,
        mux=mux,
        pump_amplitude=pump_amplitude,
        excited=True,
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
    )
    scores = {
        qubit: _state_separation_score(
            ground_samples[qubit],
            excited_samples[qubit],
        )
        for qubit in evaluated_qubits
    }
    flatness: dict[str, float] = {}
    for qubit in evaluated_qubits:
        ground_flatness = _covariance_flatness(ground_samples[qubit])
        excited_flatness = _covariance_flatness(excited_samples[qubit])
        flatness[qubit] = (
            math.nan
            if math.isnan(ground_flatness) or math.isnan(excited_flatness)
            else max(ground_flatness, excited_flatness)
        )
    return scores, flatness


def _set_dc_voltage(
    supply: _DCVoltageSupply,
    *,
    channel: int,
    voltage: float,
    tolerance: float,
    settle_time: float,
    max_attempts: int,
) -> None:
    """Apply and verify one DC voltage using a bounded retry loop."""
    measured_voltage = math.nan
    output_state = 0
    for _ in range(max_attempts):
        supply.set_voltage(channel=channel, voltage=voltage)
        supply.on(channel=channel)
        if settle_time > 0.0:
            time.sleep(settle_time)
        measured_voltage = float(supply.get_voltage(channel=channel))
        output_state = int(supply.get_output_state(channel=channel))
        if output_state == 1 and math.isclose(
            measured_voltage,
            voltage,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            return
    raise RuntimeError(
        "Failed to apply DC voltage "
        f"{voltage:g} V on channel {channel}: measured {measured_voltage:g} V, "
        f"output state {output_state}."
    )


def _scan_jpa_grid(
    exp: Experiment,
    evaluated_qubits: tuple[str, ...],
    *,
    mux: Mux,
    supply: _DCVoltageSupply,
    dc_voltages: NDArray[np.float64],
    pump_frequencies: NDArray[np.float64],
    pump_amplitudes: NDArray[np.float64],
    n_shots: int,
    shot_interval: float | None,
    readout_amplitude: float | None,
    readout_duration: float | None,
    baseline_scores: NDArray[np.float64],
    baseline_flatness: NDArray[np.float64],
    minimum_score_gain: float,
    maximum_flatness_ratio: float,
    flatness_threshold: float | None,
    dc_voltage_tolerance: float,
    dc_settle_time: float,
    dc_max_attempts: int,
    enable_tqdm: bool,
    description: str,
) -> _JPAScan:
    """Acquire and analyze one rectangular JPA parameter grid."""
    grid_shape = (
        len(evaluated_qubits),
        dc_voltages.size,
        pump_frequencies.size,
        pump_amplitudes.size,
    )
    scores = np.full(grid_shape, np.nan, dtype=np.float64)
    flatness = np.full(grid_shape, np.nan, dtype=np.float64)
    channel = mux.index + 1
    total = int(np.prod(grid_shape[1:]))
    with tqdm(total=total, desc=description, disable=not enable_tqdm) as progress:
        for dc_index, voltage in enumerate(dc_voltages):
            _set_dc_voltage(
                supply,
                channel=channel,
                voltage=float(voltage),
                tolerance=dc_voltage_tolerance,
                settle_time=dc_settle_time,
                max_attempts=dc_max_attempts,
            )
            for frequency_index, frequency in enumerate(pump_frequencies):
                with exp.ctx.modified_frequencies({mux.label: float(frequency)}):
                    for amplitude_index, amplitude in enumerate(pump_amplitudes):
                        point_scores, point_flatness = _measure_jpa_point(
                            exp,
                            evaluated_qubits,
                            mux=mux,
                            pump_amplitude=float(amplitude),
                            n_shots=n_shots,
                            shot_interval=shot_interval,
                            readout_amplitude=readout_amplitude,
                            readout_duration=readout_duration,
                        )
                        for peer_index, peer in enumerate(evaluated_qubits):
                            scores[
                                peer_index,
                                dc_index,
                                frequency_index,
                                amplitude_index,
                            ] = point_scores[peer]
                            flatness[
                                peer_index,
                                dc_index,
                                frequency_index,
                                amplitude_index,
                            ] = point_flatness[peer]
                        progress.update()

    try:
        optimum = _select_mux_optimum(
            scores,
            flatness,
            baseline_scores=baseline_scores,
            baseline_flatness=baseline_flatness,
            minimum_score_gain=minimum_score_gain,
            maximum_flatness_ratio=maximum_flatness_ratio,
            flatness_threshold=flatness_threshold,
        )
    except _NoValidPoint as exc:
        raise _ScanConstraintError(
            scan_payload=_failed_scan_payload(
                dc_voltages=dc_voltages,
                pump_frequencies=pump_frequencies,
                pump_amplitudes=pump_amplitudes,
                scores=scores,
                flatness=flatness,
                failure=exc,
                evaluated_qubits=evaluated_qubits,
            )
        ) from exc
    return _JPAScan(
        dc_voltages=dc_voltages,
        pump_frequencies=pump_frequencies,
        pump_amplitudes=pump_amplitudes,
        scores=scores,
        flatness=flatness,
        optimum=optimum,
    )


def _scan_payload(
    scan: _JPAScan,
    *,
    evaluated_qubits: Sequence[str],
) -> dict[str, object]:
    """Convert one internal scan to a stable Result payload."""
    return {
        "dc_voltages": scan.dc_voltages,
        "pump_frequencies": scan.pump_frequencies,
        "pump_amplitudes": scan.pump_amplitudes,
        "aggregate_score": scan.optimum.aggregate_score_map,
        "aggregate_score_gain": scan.optimum.aggregate_score_gain_map,
        "measurement_valid_mask": scan.optimum.measurement_valid_mask,
        "gain_valid_mask": scan.optimum.gain_valid_mask,
        "flatness_valid_mask": scan.optimum.flatness_valid_mask,
        "valid_mask": scan.optimum.valid_mask,
        "scores": {
            qubit: scan.scores[index] for index, qubit in enumerate(evaluated_qubits)
        },
        "score_gains": {
            qubit: scan.optimum.score_gains[index]
            for index, qubit in enumerate(evaluated_qubits)
        },
        "flatness": {
            qubit: scan.flatness[index] for index, qubit in enumerate(evaluated_qubits)
        },
        "flatness_ratios": {
            qubit: scan.optimum.flatness_ratios[index]
            for index, qubit in enumerate(evaluated_qubits)
        },
        "optimal_index": scan.optimum.index,
        "on_boundary": scan.optimum.on_boundary,
        "candidate_index": scan.optimum.index,
        "candidate_on_boundary": scan.optimum.on_boundary,
    }


def _failed_scan_payload(
    *,
    dc_voltages: NDArray[np.float64],
    pump_frequencies: NDArray[np.float64],
    pump_amplitudes: NDArray[np.float64],
    scores: NDArray[np.float64],
    flatness: NDArray[np.float64],
    failure: _NoValidPoint,
    evaluated_qubits: Sequence[str],
) -> dict[str, object]:
    """Convert a completed scan without an acceptable point to diagnostics."""
    candidate_mask = failure.flatness_valid_mask
    if not np.any(candidate_mask):
        candidate_mask = failure.measurement_valid_mask
    candidate_index: tuple[int, ...] | None = None
    candidate_on_boundary: bool | None = None
    if np.any(candidate_mask):
        ranking = np.where(
            candidate_mask,
            failure.aggregate_score_gain_map,
            -math.inf,
        )
        flat_index = int(np.argmax(ranking))
        candidate_index = tuple(
            int(value) for value in np.unravel_index(flat_index, ranking.shape)
        )
        candidate_on_boundary = any(
            axis_size > 1 and axis_index in (0, axis_size - 1)
            for axis_index, axis_size in zip(
                candidate_index,
                ranking.shape,
                strict=True,
            )
        )
    return {
        "dc_voltages": dc_voltages,
        "pump_frequencies": pump_frequencies,
        "pump_amplitudes": pump_amplitudes,
        "aggregate_score": failure.aggregate_score_map,
        "aggregate_score_gain": failure.aggregate_score_gain_map,
        "measurement_valid_mask": failure.measurement_valid_mask,
        "gain_valid_mask": failure.gain_valid_mask,
        "flatness_valid_mask": failure.flatness_valid_mask,
        "valid_mask": failure.valid_mask,
        "scores": {
            qubit: scores[index] for index, qubit in enumerate(evaluated_qubits)
        },
        "score_gains": {
            qubit: failure.score_gains[index]
            for index, qubit in enumerate(evaluated_qubits)
        },
        "flatness": {
            qubit: flatness[index] for index, qubit in enumerate(evaluated_qubits)
        },
        "flatness_ratios": {
            qubit: failure.flatness_ratios[index]
            for index, qubit in enumerate(evaluated_qubits)
        },
        "optimal_index": None,
        "on_boundary": None,
        "candidate_index": candidate_index,
        "candidate_on_boundary": candidate_on_boundary,
    }


def _baseline_payload(
    *,
    dc_voltage: float,
    scores: NDArray[np.float64],
    flatness: NDArray[np.float64],
    evaluated_qubits: Sequence[str],
) -> dict[str, object]:
    """Convert the common JPA-off measurement to a Result payload."""
    return {
        "parameters": {
            "dc_voltage": dc_voltage,
            "pump_amplitude": 0.0,
        },
        "score": float(np.min(scores)),
        "scores_by_qubit": dict(zip(evaluated_qubits, scores.tolist(), strict=True)),
        "flatness_by_qubit": dict(
            zip(evaluated_qubits, flatness.tolist(), strict=True)
        ),
    }


def _constraint_failure_status(scan_payload: dict[str, object]) -> str:
    """Classify why a completed grid has no acceptable point."""
    measurement_valid = np.asarray(
        scan_payload["measurement_valid_mask"],
        dtype=np.bool_,
    )
    flatness_valid = np.asarray(
        scan_payload["flatness_valid_mask"],
        dtype=np.bool_,
    )
    gain_valid = np.asarray(scan_payload["gain_valid_mask"], dtype=np.bool_)
    if not np.any(measurement_valid):
        return "invalid_measurement"
    if not np.any(flatness_valid):
        return "no_flatness_compliant_point"
    if not np.any(gain_valid):
        return "no_improvement"
    return "no_jointly_valid_point"


def _failure_message(*, stage: str, status: str) -> str:
    """Return one actionable public constraint failure message."""
    explanations = {
        "invalid_baseline": "the JPA-off baseline is not finite and usable",
        "invalid_measurement": "all grid measurements are invalid",
        "no_flatness_compliant_point": (
            "no point satisfies the IQ-flatness constraints"
        ),
        "no_improvement": "no point improves every MUX peer over JPA-off",
        "no_jointly_valid_point": (
            "no point satisfies the gain and flatness constraints together"
        ),
    }
    detail = explanations.get(status, "no acceptable point was found")
    return f"JPA calibration failed during the {stage} stage: {detail}."


def _validate_options(
    *,
    n_shots: int,
    shot_interval: float | None,
    readout_amplitude: float | None,
    readout_duration: float | None,
    minimum_score_gain: float,
    maximum_flatness_ratio: float,
    flatness_threshold: float | None,
    fine_points: int | None,
    dc_voltage_tolerance: float,
    dc_settle_time: float,
    dc_max_attempts: int,
) -> None:
    """Validate scalar calibration options before touching hardware."""
    if n_shots < 3:
        raise ValueError("`n_shots` must be at least 3 for IQ flatness estimation.")
    if shot_interval is not None and (
        not math.isfinite(shot_interval) or shot_interval < 0.0
    ):
        raise ValueError("`shot_interval` must be finite and non-negative.")
    if readout_amplitude is not None and (
        not math.isfinite(readout_amplitude) or not 0.0 <= readout_amplitude <= 1.0
    ):
        raise ValueError("`readout_amplitude` must be finite and within [0, 1].")
    if readout_duration is not None and (
        not math.isfinite(readout_duration) or readout_duration <= 0.0
    ):
        raise ValueError("`readout_duration` must be finite and positive.")
    if not math.isfinite(minimum_score_gain) or minimum_score_gain < 1.0:
        raise ValueError("`minimum_score_gain` must be finite and at least 1.")
    if not math.isfinite(maximum_flatness_ratio) or maximum_flatness_ratio < 1.0:
        raise ValueError("`maximum_flatness_ratio` must be finite and at least 1.")
    if flatness_threshold is not None and (
        not math.isfinite(flatness_threshold) or flatness_threshold < 1.0
    ):
        raise ValueError(
            "`flatness_threshold` must be `None`, or finite and at least 1."
        )
    if fine_points is not None and fine_points < 3:
        raise ValueError("`fine_points` must be `None` or at least 3.")
    if not math.isfinite(dc_voltage_tolerance) or dc_voltage_tolerance <= 0.0:
        raise ValueError("`dc_voltage_tolerance` must be finite and positive.")
    if not math.isfinite(dc_settle_time) or dc_settle_time < 0.0:
        raise ValueError("`dc_settle_time` must be finite and non-negative.")
    if dc_max_attempts < 1:
        raise ValueError("`dc_max_attempts` must be at least 1.")


def calibrate_jpa(
    exp: Experiment,
    qubit: str,
    *,
    dc_voltage_range: ArrayLike | None = None,
    pump_frequency_range: ArrayLike | None = None,
    pump_amplitude_range: ArrayLike | None = None,
    dc_voltage_limits: tuple[float, float] = DEFAULT_DC_VOLTAGE_LIMITS,
    n_shots: int = DEFAULT_N_SHOTS,
    shot_interval: float | None = None,
    readout_amplitude: float | None = None,
    readout_duration: float | None = None,
    minimum_score_gain: float = DEFAULT_MINIMUM_SCORE_GAIN,
    maximum_flatness_ratio: float = DEFAULT_MAXIMUM_FLATNESS_RATIO,
    flatness_threshold: float | None = DEFAULT_FLATNESS_THRESHOLD,
    baseline_dc_voltage: float = DEFAULT_BASELINE_DC_VOLTAGE,
    fine_points: int | None = DEFAULT_FINE_POINTS,
    dc_voltage_tolerance: float = DEFAULT_DC_VOLTAGE_TOLERANCE,
    dc_settle_time: float = DEFAULT_DC_SETTLE_TIME,
    dc_max_attempts: int = DEFAULT_DC_MAX_ATTEMPTS,
    reset_awg_and_capunits: bool = True,
    enable_tqdm: bool = True,
) -> Result:
    """
    Calibrate MUX-shared JPA parameters from one anchor qubit.

    The anchor resolves the MUX automatically. Every active, valid qubit on
    that MUX is measured simultaneously. A common completely-off measurement
    (zero pump amplitude and `baseline_dc_voltage`) is acquired first with the
    same readout settings. The selected point maximizes the weakest peer's
    noise-normalized g/e separation gain over that baseline while limiting IQ
    cloud deformation relative to the same baseline.

    Parameters
    ----------
    exp : Experiment
        Connected and configured Qubex experiment.
    qubit : str
        Qubit or readout label used to resolve the shared MUX.
    dc_voltage_range : ArrayLike, optional
        Strictly increasing DC voltages in V. Defaults to nine points around
        the configured MUX voltage, clipped to `dc_voltage_limits`.
    pump_frequency_range : ArrayLike, optional
        Strictly increasing pump frequencies in GHz. Defaults to nine points
        from the configured frequency to +0.5 GHz when the configured pump is
        off, or within ±0.25 GHz when recalibrating an active pump.
    pump_amplitude_range : ArrayLike, optional
        Strictly increasing dimensionless amplitudes in [0, 1]. Defaults to
        nine points around the configured amplitude.
    dc_voltage_limits : tuple[float, float], optional
        Inclusive safety limits in V applied to both default and explicit DC
        voltage ranges. Defaults to `(0, 4)`.
    n_shots : int, optional
        Single-shot samples acquired for each of the g and e states. Must be
        at least three for the two-dimensional covariance estimate.
    shot_interval : float, optional
        Shot interval in ns.
    readout_amplitude : float, optional
        Shared readout amplitude override in [0, 1]. When omitted, each qubit
        uses its configured value.
    readout_duration : float, optional
        Readout pulse duration in ns. When omitted, uses the configured default.
    minimum_score_gain : float, optional
        Minimum separation-score ratio to JPA-off required independently for
        every evaluated qubit. Defaults to `1.05`, requiring at least 5 percent
        improvement for the weakest MUX peer.
    maximum_flatness_ratio : float, optional
        Maximum ratio of IQ-cloud flatness to its JPA-off value, independently
        for every evaluated qubit. Defaults to `1.1`.
    flatness_threshold : float, optional
        Optional absolute IQ-cloud flatness cap applied in addition to the
        relative constraint. The default `None` disables the absolute cap.
    baseline_dc_voltage : float, optional
        DC voltage in V used for the zero-pump reference measurement. Defaults
        to `0.0` and must lie within `dc_voltage_limits`.
    fine_points : int, optional
        Points per non-singleton axis in the automatic fine scan. Set to
        `None` to run only the coarse scan; otherwise must be at least three.
    dc_voltage_tolerance : float, optional
        Absolute DC voltage verification tolerance in V.
    dc_settle_time : float, optional
        Delay after each bounded DC voltage update in seconds.
    dc_max_attempts : int, optional
        Maximum attempts used to set and verify each DC voltage.
    reset_awg_and_capunits : bool, optional
        Whether to reset every control/readout/capture/pump box once before
        the sweep.
    enable_tqdm : bool, optional
        Whether to display coarse and fine scan progress bars.

    Returns
    -------
    Result
        Optimal `jpa_params`, the JPA-off baseline, and complete raw and
        baseline-relative coarse/fine maps. `score` remains the raw weakest-peer
        separation; `score_gain` is the weakest-peer ratio to JPA-off.

    Raises
    ------
    JPAConstraintError
        Raised after safe hardware cleanup when the baseline is unusable or no
        scanned point satisfies every peer's gain and flatness constraints.
        Complete data acquired before the failure is available from
        `error.diagnostics`, and the failed stage from `error.stage`.
    ValueError
        Raised for invalid ranges, invalid options, or missing topology.
    RuntimeError
        Raised when DC verification or measurement result validation fails.

    Notes
    -----
    This workflow controls the ONS61797 DC supply on the backend's configured
    port (currently `/dev/ttyACM0`) and prepares the e state, so valid π pulses
    and pump wiring are required. DC voltage is restored and the touched output
    is disabled when the context exits. The returned optimum is neither applied
    permanently nor written to `jpa_params.yaml`.

    Examples
    --------
    >>> from qubex import contrib
    >>> result = contrib.calibrate_jpa(exp, "Q24")
    >>> result["optimal_parameters"]
    {'dc_voltage': ..., 'pump_frequency': ..., 'pump_amplitude': ...}
    >>> result["score_gain"]
    1.05...

    Inspect a completed scan that found no acceptable point:

    >>> try:
    ...     result = contrib.calibrate_jpa(exp, "Q24")
    ... except contrib.JPAConstraintError as error:
    ...     print(error.stage, error.diagnostics["status"])
    ...     coarse = error.diagnostics["coarse_scan"]
    """
    _validate_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        readout_amplitude=readout_amplitude,
        readout_duration=readout_duration,
        minimum_score_gain=minimum_score_gain,
        maximum_flatness_ratio=maximum_flatness_ratio,
        flatness_threshold=flatness_threshold,
        fine_points=fine_points,
        dc_voltage_tolerance=dc_voltage_tolerance,
        dc_settle_time=dc_settle_time,
        dc_max_attempts=dc_max_attempts,
    )
    anchor_qubit, mux, evaluated_qubits = _resolve_evaluated_qubits(exp, qubit)
    control_params = exp.ctx.experiment_system.control_params
    configured_dc_voltage = float(control_params.get_dc_voltage(mux.index))
    configured_pump_frequency = float(control_params.get_pump_frequency(mux.index))
    configured_pump_amplitude = float(control_params.get_pump_amplitude(mux.index))
    minimum_dc_voltage, maximum_dc_voltage = _normalize_dc_voltage_limits(
        dc_voltage_limits
    )
    if not math.isfinite(baseline_dc_voltage) or not (
        minimum_dc_voltage <= baseline_dc_voltage <= maximum_dc_voltage
    ):
        raise ValueError(
            "`baseline_dc_voltage` must be finite and within `dc_voltage_limits`."
        )

    dc_voltages = _normalize_axis(
        (
            dc_voltage_range
            if dc_voltage_range is not None
            else _centered_axis(
                configured_dc_voltage,
                half_width=DEFAULT_DC_VOLTAGE_HALF_WIDTH,
                minimum=minimum_dc_voltage,
                maximum=maximum_dc_voltage,
            )
        ),
        name="dc_voltage_range",
        minimum=minimum_dc_voltage,
        maximum=maximum_dc_voltage,
    )
    pump_frequencies = _normalize_axis(
        (
            pump_frequency_range
            if pump_frequency_range is not None
            else _default_pump_frequency_axis(
                configured_pump_frequency,
                configured_amplitude=configured_pump_amplitude,
            )
        ),
        name="pump_frequency_range",
        minimum=float(np.finfo(np.float64).tiny),
    )
    pump_amplitudes = _normalize_axis(
        (
            pump_amplitude_range
            if pump_amplitude_range is not None
            else _centered_axis(
                configured_pump_amplitude,
                half_width=DEFAULT_PUMP_AMPLITUDE_HALF_WIDTH,
                minimum=0.0,
                maximum=1.0,
            )
        ),
        name="pump_amplitude_range",
        minimum=0.0,
        maximum=1.0,
    )

    if reset_awg_and_capunits:
        _reset_jpa_hardware(exp, evaluated_qubits, mux=mux)

    channel = mux.index + 1
    baseline_scores = np.full(len(evaluated_qubits), np.nan, dtype=np.float64)
    baseline_flatness = np.full(len(evaluated_qubits), np.nan, dtype=np.float64)
    coarse_scan: _JPAScan | None = None
    fine_scan: _JPAScan | None = None
    coarse_payload: dict[str, object] | None = None
    fine_payload: dict[str, object] | None = None
    failure_stage: Literal["baseline", "coarse", "fine"] | None = None
    failure_status: str | None = None

    with dc_voltage({channel: float(baseline_dc_voltage)}) as supply:
        _set_dc_voltage(
            supply,
            channel=channel,
            voltage=float(baseline_dc_voltage),
            tolerance=dc_voltage_tolerance,
            settle_time=dc_settle_time,
            max_attempts=dc_max_attempts,
        )
        baseline_score_values, baseline_flatness_values = _measure_jpa_point(
            exp,
            evaluated_qubits,
            mux=mux,
            pump_amplitude=0.0,
            n_shots=n_shots,
            shot_interval=shot_interval,
            readout_amplitude=readout_amplitude,
            readout_duration=readout_duration,
        )
        baseline_scores = np.asarray(
            [baseline_score_values[peer] for peer in evaluated_qubits],
            dtype=np.float64,
        )
        baseline_flatness = np.asarray(
            [baseline_flatness_values[peer] for peer in evaluated_qubits],
            dtype=np.float64,
        )

        if (
            not np.all(np.isfinite(baseline_scores))
            or np.any(baseline_scores <= 0.0)
            or not np.all(np.isfinite(baseline_flatness))
            or np.any(baseline_flatness < 1.0)
        ):
            failure_stage = "baseline"
            failure_status = "invalid_baseline"
        else:
            coarse_constraint_status: str | None = None
            try:
                coarse_scan = _scan_jpa_grid(
                    exp,
                    evaluated_qubits,
                    mux=mux,
                    supply=supply,
                    dc_voltages=dc_voltages,
                    pump_frequencies=pump_frequencies,
                    pump_amplitudes=pump_amplitudes,
                    n_shots=n_shots,
                    shot_interval=shot_interval,
                    readout_amplitude=readout_amplitude,
                    readout_duration=readout_duration,
                    baseline_scores=baseline_scores,
                    baseline_flatness=baseline_flatness,
                    minimum_score_gain=minimum_score_gain,
                    maximum_flatness_ratio=maximum_flatness_ratio,
                    flatness_threshold=flatness_threshold,
                    dc_voltage_tolerance=dc_voltage_tolerance,
                    dc_settle_time=dc_settle_time,
                    dc_max_attempts=dc_max_attempts,
                    enable_tqdm=enable_tqdm,
                    description=f"JPA coarse scan {mux.label}",
                )
            except _ScanConstraintError as error:
                coarse_payload = error.scan_payload
                coarse_constraint_status = _constraint_failure_status(coarse_payload)
            else:
                coarse_payload = _scan_payload(
                    coarse_scan,
                    evaluated_qubits=evaluated_qubits,
                )

            refinement_index: tuple[int, ...] | None = None
            if coarse_scan is not None:
                refinement_index = coarse_scan.optimum.index
            elif coarse_payload is not None:
                candidate_index = coarse_payload["candidate_index"]
                if isinstance(candidate_index, tuple):
                    refinement_index = tuple(int(value) for value in candidate_index)
            should_refine = (
                fine_points is not None
                and refinement_index is not None
                and any(
                    axis.size > 1
                    for axis in (
                        dc_voltages,
                        pump_frequencies,
                        pump_amplitudes,
                    )
                )
            )
            if should_refine:
                if fine_points is None or refinement_index is None:
                    raise RuntimeError("JPA fine-scan inputs are unexpectedly missing.")
                fine_dc_voltages = _refined_axis(
                    dc_voltages,
                    refinement_index[0],
                    points=fine_points,
                )
                fine_pump_frequencies = _refined_axis(
                    pump_frequencies,
                    refinement_index[1],
                    points=fine_points,
                )
                fine_pump_amplitudes = _refined_axis(
                    pump_amplitudes,
                    refinement_index[2],
                    points=fine_points,
                )
                try:
                    fine_scan = _scan_jpa_grid(
                        exp,
                        evaluated_qubits,
                        mux=mux,
                        supply=supply,
                        dc_voltages=fine_dc_voltages,
                        pump_frequencies=fine_pump_frequencies,
                        pump_amplitudes=fine_pump_amplitudes,
                        n_shots=n_shots,
                        shot_interval=shot_interval,
                        readout_amplitude=readout_amplitude,
                        readout_duration=readout_duration,
                        baseline_scores=baseline_scores,
                        baseline_flatness=baseline_flatness,
                        minimum_score_gain=minimum_score_gain,
                        maximum_flatness_ratio=maximum_flatness_ratio,
                        flatness_threshold=flatness_threshold,
                        dc_voltage_tolerance=dc_voltage_tolerance,
                        dc_settle_time=dc_settle_time,
                        dc_max_attempts=dc_max_attempts,
                        enable_tqdm=enable_tqdm,
                        description=f"JPA fine scan {mux.label}",
                    )
                except _ScanConstraintError as error:
                    fine_payload = error.scan_payload
                    failure_stage = "fine"
                    failure_status = _constraint_failure_status(fine_payload)
                else:
                    fine_payload = _scan_payload(
                        fine_scan,
                        evaluated_qubits=evaluated_qubits,
                    )
            elif coarse_constraint_status is not None:
                failure_stage = "coarse"
                failure_status = coarse_constraint_status

    baseline = _baseline_payload(
        dc_voltage=float(baseline_dc_voltage),
        scores=baseline_scores,
        flatness=baseline_flatness,
        evaluated_qubits=evaluated_qubits,
    )
    common_payload: dict[str, object] = {
        "anchor_qubit": anchor_qubit,
        "evaluated_qubits": evaluated_qubits,
        "mux_label": mux.label,
        "mux_index": mux.index,
        "objective": "worst_peer_state_separation_gain",
        "minimum_score_gain": float(minimum_score_gain),
        "maximum_flatness_ratio": float(maximum_flatness_ratio),
        "flatness_threshold": (
            None if flatness_threshold is None else float(flatness_threshold)
        ),
        "baseline": baseline,
        "coarse_scan": coarse_payload,
        "fine_scan": fine_payload,
    }
    if failure_stage is not None:
        if failure_status is None:
            failure_status = "no_jointly_valid_point"
        message = _failure_message(stage=failure_stage, status=failure_status)
        diagnostics = Result(
            data={
                **common_payload,
                "success": False,
                "status": failure_status,
                "message": message,
                "failure_stage": failure_stage,
                "optimal_parameters": None,
                "score": None,
                "score_gain": None,
                "scores_by_qubit": None,
                "score_gains_by_qubit": None,
                "flatness_by_qubit": None,
                "flatness_ratios_by_qubit": None,
                "on_boundary": None,
            }
        )
        raise JPAConstraintError(
            message,
            stage=failure_stage,
            diagnostics=diagnostics,
        )

    selected_scan = fine_scan if fine_scan is not None else coarse_scan
    if selected_scan is None:
        raise RuntimeError("JPA calibration result is unexpectedly missing.")
    dc_index, frequency_index, amplitude_index = selected_scan.optimum.index
    optimal_parameters = {
        "dc_voltage": float(selected_scan.dc_voltages[dc_index]),
        "pump_frequency": float(selected_scan.pump_frequencies[frequency_index]),
        "pump_amplitude": float(selected_scan.pump_amplitudes[amplitude_index]),
    }
    return Result(
        data={
            **common_payload,
            "success": True,
            "status": "success",
            "message": "JPA calibration found an acceptable operating point.",
            "failure_stage": None,
            "optimal_parameters": optimal_parameters,
            "score": selected_scan.optimum.aggregate_score,
            "score_gain": selected_scan.optimum.aggregate_score_gain,
            "scores_by_qubit": dict(
                zip(
                    evaluated_qubits,
                    selected_scan.optimum.peer_scores.tolist(),
                    strict=True,
                )
            ),
            "score_gains_by_qubit": dict(
                zip(
                    evaluated_qubits,
                    selected_scan.optimum.peer_score_gains.tolist(),
                    strict=True,
                )
            ),
            "flatness_by_qubit": dict(
                zip(
                    evaluated_qubits,
                    selected_scan.optimum.peer_flatness.tolist(),
                    strict=True,
                )
            ),
            "flatness_ratios_by_qubit": dict(
                zip(
                    evaluated_qubits,
                    selected_scan.optimum.peer_flatness_ratios.tolist(),
                    strict=True,
                )
            ),
            "on_boundary": selected_scan.optimum.on_boundary,
        }
    )


__all__ = ["JPAConstraintError", "calibrate_jpa"]

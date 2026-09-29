"""Contributed simultaneous randomized benchmarking helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

import qubex.visualization as viz
from qubex.analysis import fitting
from qubex.clifford import Clifford, CliffordSequence
from qubex.experiment import Experiment
from qubex.experiment.experiment_constants import (
    DEFAULT_INTERVAL,
    DEFAULT_MAX_N_CLIFFORDS_1Q,
    DEFAULT_RB_N_TRIALS,
    DEFAULT_SHOTS,
)
from qubex.experiment.models import Result
from qubex.pulse import PulseArray, PulseSchedule, Waveform
from qubex.typing import TargetMap

from ._deprecated_options import resolve_shot_options

_XY_GATE_NAMES = frozenset({"X90", "Y90"})
_N_1Q_CLIFFORDS = 24


def generate_1q_xy_cliffords(
    *,
    max_gates: int = 7,
) -> dict[Clifford, CliffordSequence]:
    """
    Generate 1Q Clifford representatives using only physical `X90`/`Y90` gates.

    Parameters
    ----------
    max_gates
        Maximum number of generator gates to search.

    Returns
    -------
    dict[Clifford, CliffordSequence]
        Clifford map to shortest known `X90`/`Y90` representative sequence.

    Raises
    ------
    ValueError
        If `max_gates` is negative or the full 1Q Clifford group is not found.
    """
    if max_gates < 0:
        raise ValueError("max_gates must be non-negative.")

    identity = CliffordSequence.I()
    found_cliffords: dict[Clifford, CliffordSequence] = {
        identity.clifford: identity
    }
    generators = (Clifford.X90(), Clifford.Y90())

    def sequence_cost(sequence: CliffordSequence) -> tuple[int, tuple[str, ...]]:
        return sequence.length, tuple(sequence.gate_sequence)

    def visit(sequence: CliffordSequence, remaining_gates: int) -> None:
        if remaining_gates == 0:
            return

        for generator in generators:
            new_sequence = sequence.compose(generator)
            existing = found_cliffords.get(new_sequence.clifford)
            if existing is None or sequence_cost(new_sequence) < sequence_cost(
                existing
            ):
                found_cliffords[new_sequence.clifford] = new_sequence
            visit(new_sequence, remaining_gates - 1)

    visit(identity, max_gates)

    if len(found_cliffords) != _N_1Q_CLIFFORDS:
        raise ValueError(
            "X90/Y90 search did not generate all 24 1Q Cliffords; "
            "increase max_gates."
        )

    return found_cliffords


def create_xy_rb_sequences(
    n: int,
    *,
    seed: int | None = None,
    max_gates: int = 7,
    cliffords: Mapping[Clifford, CliffordSequence] | None = None,
) -> tuple[list[list[str]], list[str]]:
    """
    Create randomized benchmarking Clifford sequences using `X90`/`Y90`.

    Parameters
    ----------
    n
        Number of random Cliffords.
    seed
        Random seed used to choose Clifford representatives.
    max_gates
        Maximum generator search depth when `cliffords` is not provided.
    cliffords
        Optional precomputed `X90`/`Y90` Clifford representatives.

    Returns
    -------
    tuple[list[list[str]], list[str]]
        Random Clifford gate sequences and the final inverse gate sequence.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")

    clifford_table = (
        dict(cliffords)
        if cliffords is not None
        else generate_1q_xy_cliffords(max_gates=max_gates)
    )
    for clifford_sequence in clifford_table.values():
        _assert_xy_gate_sequence(clifford_sequence.gate_sequence)

    rng = random.Random(seed)
    random_sequences = rng.choices(list(clifford_table.values()), k=n)

    composed = CliffordSequence.I()
    for clifford_sequence in random_sequences:
        composed = composed.compose(clifford_sequence.clifford)

    inverse = clifford_table.get(composed.clifford.inverse)
    if inverse is None:
        raise ValueError("Inverse Clifford was not found in the X90/Y90 table.")

    return [
        clifford_sequence.gate_sequence for clifford_sequence in random_sequences
    ], inverse.gate_sequence


def xy_rb_sequence_1q(
    exp: Experiment,
    target: str,
    *,
    n: int,
    x90: Waveform | None = None,
    y90: Waveform | None = None,
    seed: int | None = None,
    max_gates: int = 7,
) -> PulseArray:
    """
    Build a single-qubit RB sequence using only `X90` and `Y90` pulses.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse calibrations.
    target
        Target qubit label.
    n
        Number of random Cliffords.
    x90
        Optional `X90` waveform override.
    y90
        Optional `Y90` waveform override.
    seed
        Random seed for Clifford selection.
    max_gates
        Maximum generator search depth for the Clifford table.

    Returns
    -------
    PulseArray
        Pulse sequence ending with the total inverse Clifford.
    """
    x90_waveform, y90_waveform = _resolve_xy_waveforms(
        exp,
        target,
        x90_waveform=x90,
        y90_waveform=y90,
    )
    cliffords, inverse = create_xy_rb_sequences(
        n,
        seed=seed,
        max_gates=max_gates,
    )
    sequence: list[Waveform] = []

    def add_gate(gate: str) -> None:
        if gate == "X90":
            sequence.append(x90_waveform)
        elif gate == "Y90":
            sequence.append(y90_waveform)
        else:
            raise ValueError(f"Invalid XY Clifford gate: {gate}.")

    for clifford in cliffords:
        for gate in clifford:
            add_gate(gate)
    for gate in inverse:
        add_gate(gate)

    return PulseArray(sequence)


def simultaneous_xy_rb_sequence(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n: int,
    x90: TargetMap[Waveform] | None = None,
    y90: TargetMap[Waveform] | None = None,
    seed: int | None = None,
    max_gates: int = 7,
) -> PulseSchedule:
    """
    Build a simultaneous two-qubit RB schedule from independent XY Cliffords.

    Each target receives an independently sampled 1Q Clifford at each RB layer.
    A barrier is inserted after every layer so shorter physical decompositions
    are padded before the next Clifford starts.

    Parameters
    ----------
    exp
        Experiment instance that provides pulse calibrations.
    targets
        Exactly two 1Q target labels.
    n
        Number of random Cliffords.
    x90
        Optional `X90` waveform overrides by target.
    y90
        Optional `Y90` waveform overrides by target.
    seed
        Base random seed. Per-target seeds are derived from it.
    max_gates
        Maximum generator search depth for the Clifford table.

    Returns
    -------
    PulseSchedule
        Simultaneous RB pulse schedule.
    """
    target_list = _normalize_two_1q_targets(exp, targets)
    target_seeds = _target_seeds(target_list, seed)
    clifford_table = generate_1q_xy_cliffords(max_gates=max_gates)
    xy_sequences: dict[str, tuple[list[list[str]], list[str]]] = {}

    for target in target_list:
        xy_sequences[target] = create_xy_rb_sequences(
            n,
            seed=target_seeds[target],
            max_gates=max_gates,
            cliffords=clifford_table,
        )

    x90_waveforms: dict[str, Waveform] = {}
    y90_waveforms: dict[str, Waveform] = {}
    for target in target_list:
        x90_waveforms[target], y90_waveforms[target] = _resolve_xy_waveforms(
            exp,
            target,
            x90_waveform=x90.get(target) if x90 is not None else None,
            y90_waveform=y90.get(target) if y90 is not None else None,
        )

    with PulseSchedule(target_list) as schedule:

        def add_gate(target: str, gate: str) -> None:
            if gate == "X90":
                schedule.add(target, x90_waveforms[target])
            elif gate == "Y90":
                schedule.add(target, y90_waveforms[target])
            else:
                raise ValueError(f"Invalid XY Clifford gate: {gate}.")

        for clifford_index in range(n):
            for target in target_list:
                cliffords, _inverse = xy_sequences[target]
                for gate in cliffords[clifford_index]:
                    add_gate(target, gate)
            schedule.barrier(target_list)

        for target in target_list:
            _cliffords, inverse = xy_sequences[target]
            for gate in inverse:
                add_gate(target, gate)

    return schedule


def simultaneous_randomized_benchmarking(
    exp: Experiment,
    targets: Collection[str] | str,
    *,
    n_cliffords_range: ArrayLike | None = None,
    n_trials: int | None = None,
    seeds: ArrayLike | None = None,
    max_n_cliffords: int | None = None,
    x90: TargetMap[Waveform] | None = None,
    y90: TargetMap[Waveform] | None = None,
    n_shots: int | None = None,
    shot_interval: float | None = None,
    time_integration: bool | None = None,
    xaxis_type: Literal["linear", "log"] | None = None,
    plot: bool | None = None,
    save_image: bool | None = None,
    reset_awg_and_capunits: bool | None = None,
    max_gates: int = 7,
    **deprecated_options: Any,
) -> Result:
    """
    Run simultaneous randomized benchmarking for two 1Q targets.

    The experiment follows the existing 1Q randomized benchmarking workflow but
    replaces virtual-Z Clifford representatives with physical `X90`/`Y90`
    representatives. The two targets are measured in one schedule per
    `(n_cliffords, seed)` point.

    Parameters
    ----------
    exp
        Experiment instance.
    targets
        Exactly two 1Q target labels.
    n_cliffords_range
        Clifford count sweep range.
    n_trials
        Number of random trials per sweep point.
    seeds
        Base seeds for each trial.
    max_n_cliffords
        Maximum Clifford count when `n_cliffords_range` is omitted.
    x90
        Optional `X90` waveform overrides by target.
    y90
        Optional `Y90` waveform overrides by target.
    n_shots
        Number of shots per measurement.
    shot_interval
        Shot interval.
    time_integration
        Whether to enable time integration in measurement.
    xaxis_type
        Fit x-axis scale.
    plot
        Whether to plot fit results.
    save_image
        Whether to save fit figures.
    reset_awg_and_capunits
        Whether to reset AWGs/capture units before each measurement.
    max_gates
        Maximum generator search depth for the XY Clifford table.

    Returns
    -------
    Result
        RB fit results keyed by target.
    """
    target_list = _normalize_two_1q_targets(exp, targets)

    if plot is None:
        plot = True
    if save_image is None:
        save_image = True
    if reset_awg_and_capunits is None:
        reset_awg_and_capunits = True
    if time_integration is None:
        time_integration = True

    if n_cliffords_range is not None:
        n_cliffords_range = np.asarray(n_cliffords_range, dtype=int)

    if n_trials is None:
        n_trials = DEFAULT_RB_N_TRIALS

    if seeds is None:
        seeds = np.random.default_rng().integers(0, 2**32, n_trials)
    else:
        seeds = np.asarray(seeds, dtype=int)
        if len(seeds) != n_trials:
            raise ValueError(
                "The number of seeds must be equal to the number of trials."
            )

    if max_n_cliffords is None:
        max_n_cliffords = DEFAULT_MAX_N_CLIFFORDS_1Q

    n_shots, shot_interval = resolve_shot_options(
        n_shots=n_shots,
        shot_interval=shot_interval,
        deprecated_options=deprecated_options,
        function_name="simultaneous_randomized_benchmarking",
    )
    if n_shots is None:
        n_shots = DEFAULT_SHOTS
    if shot_interval is None:
        shot_interval = DEFAULT_INTERVAL
    if xaxis_type is None:
        xaxis_type = "linear"

    sweep_range: list[int] = []
    mean_data: dict[str, list[float]] = defaultdict(list)
    std_data: dict[str, list[float]] = defaultdict(list)
    trial_matrix_data: dict[str, list[np.ndarray]] = defaultdict(list)

    idx = 0
    while True:
        if n_cliffords_range is None:
            n_clifford = 0 if idx == 0 else 2 ** (idx - 1)
            if n_clifford > max_n_cliffords:
                break
        else:
            if idx >= len(n_cliffords_range):
                break
            n_clifford = int(n_cliffords_range[idx])

        idx += 1
        sweep_range.append(n_clifford)

        trial_data: dict[str, list[float]] = defaultdict(list)
        for seed in seeds:
            result = exp.measurement_service.measure(
                sequence=simultaneous_xy_rb_sequence(
                    exp,
                    target_list,
                    n=n_clifford,
                    x90=x90,
                    y90=y90,
                    seed=int(seed),
                    max_gates=max_gates,
                ),
                mode="avg",
                n_shots=n_shots,
                shot_interval=shot_interval,
                time_integration=time_integration,
                reset_awg_and_capunits=reset_awg_and_capunits,
                plot=False,
            )
            for target in target_list:
                iq = result.data[target].kerneled
                z = exp.pulse.rabi_params[target].normalize(iq)
                trial_data[target].append(float((z + 1) / 2))

        check_vals = {}
        for target in target_list:
            trial_values = np.asarray(trial_data[target], dtype=float)
            mean = float(np.mean(trial_values))
            std = float(np.std(trial_values))
            trial_matrix_data[target].append(trial_values)
            mean_data[target].append(mean)
            std_data[target].append(std)
            check_vals[target] = mean - std * 0.5

        if n_cliffords_range is None and max(check_vals.values()) < 0.5:
            break

    sweep_array = np.asarray(sweep_range, dtype=int)
    return_data = {}
    figures = {}

    for target in target_list:
        mean = np.asarray(mean_data[target], dtype=float)
        std = np.asarray(std_data[target], dtype=float) if n_trials > 1 else None
        fit_result = fitting.fit_rb(
            target=target,
            x=sweep_array,
            y=mean,
            error_y=std,
            bounds=((0, 0, 0), (0.5, 1, 1)),
            title="Simultaneous randomized benchmarking",
            xlabel="Number of Cliffords",
            ylabel="Normalized signal",
            xaxis_type=xaxis_type,
            yaxis_type="linear",
            plot=plot,
        )

        if save_image:
            fig = fit_result.get_figure()
            viz.save_figure(
                fig,
                name=f"simultaneous_randomized_benchmarking_{target}",
            )

        return_data[target] = {
            "n_cliffords": sweep_array,
            "mean": mean,
            "std": std,
            "trials": np.vstack(trial_matrix_data[target]),
            "seeds": np.asarray(seeds, dtype=int),
            **fit_result,
        }
        figures[target] = return_data[target]["fig"]

    return Result(data=return_data, figures=figures)


def _normalize_two_1q_targets(
    exp: Experiment,
    targets: Collection[str] | str,
) -> list[str]:
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = list(targets)

    if len(target_list) != 2:
        raise ValueError("simultaneous RB requires exactly two 1Q targets.")

    for target in target_list:
        target_object = exp.ctx.experiment_system.get_target(target)
        if target_object.is_cr:
            raise ValueError(f"`{target}` is not a 1Q target.")

    return target_list


def _target_seeds(targets: list[str], seed: int | None) -> dict[str, int | None]:
    if seed is None:
        return dict.fromkeys(targets, None)

    rng = random.Random(seed)
    return {target: rng.randrange(0, 2**32) for target in targets}


def _resolve_xy_waveforms(
    exp: Experiment,
    target: str,
    *,
    x90_waveform: Waveform | None,
    y90_waveform: Waveform | None,
) -> tuple[Waveform, Waveform]:
    resolved_x90 = x90_waveform or exp.pulse.x90(target)
    if y90_waveform is not None:
        return resolved_x90, y90_waveform
    if x90_waveform is not None:
        return resolved_x90, resolved_x90.shifted(np.pi / 2)
    return resolved_x90, exp.pulse.y90(target)


def _assert_xy_gate_sequence(gate_sequence: Collection[str]) -> None:
    invalid_gates = sorted(set(gate_sequence) - _XY_GATE_NAMES)
    if invalid_gates:
        raise ValueError(f"Invalid non-XY Clifford gates: {invalid_gates}.")

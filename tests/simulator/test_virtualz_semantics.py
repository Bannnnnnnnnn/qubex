"""Regression tests for simulator VirtualZ semantics."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt
from qxpulse import Arbitrary, PulseArray, PulseChannel, PulseSchedule, VirtualZ
from qxsimulator import Control, QuantumSimulator, QuantumSystem, Transmon


def test_schedule_final_state_and_density_matrix_apply_terminal_virtualz() -> None:
    """PulseSchedule results should expose terminal VirtualZ in user-facing state accessors."""
    simulator, system, qubit = _single_qubit_simulator()
    schedule = _make_schedule(
        qubit,
        PulseArray([Arbitrary([0 + 0j]), VirtualZ(np.pi / 2)]),
    )

    result = simulator.simulate(
        schedule,
        initial_state=system.state({qubit.label: "+"}),
        dt=2.0,
    )
    direct_result = simulator.simulate(
        [
            Control(
                target=qubit.label,
                frequency=qubit.frequency,
                waveform=schedule.get_sampled_sequences()[qubit.label],
            )
        ],
        initial_state=system.state({qubit.label: "+"}),
        dt=2.0,
    )

    expected = _rotate_density_matrix(
        system,
        qubit.label,
        result.states[-1],
        np.pi / 2,
    )

    assert np.allclose(result.states[-1].full(), direct_result.states[-1].full())
    assert np.allclose(result.final_state.full(), expected.full())
    assert np.allclose(result.get_final_substate(qubit.label).full(), expected.full())
    assert np.allclose(result.get_density_matrices(qubit.label)[-1], expected.full())
    assert np.allclose(
        direct_result.final_state.full(), direct_result.states[-1].full()
    )


def test_schedule_bloch_vectors_reflect_mid_sequence_virtualz_jump() -> None:
    """PulseSchedule Bloch vectors should include the logical-frame jump from VirtualZ."""
    simulator, system, qubit = _single_qubit_simulator()
    schedule = _make_schedule(
        qubit,
        PulseArray(
            [
                Arbitrary([0 + 0j]),
                VirtualZ(np.pi / 2),
                Arbitrary([0 + 0j]),
            ]
        ),
    )

    result = simulator.simulate(
        schedule,
        initial_state=system.state({qubit.label: "+"}),
        dt=2.0,
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    assert np.allclose(result.get_bloch_vectors(qubit.label), expected, atol=1e-9)


def test_schedule_frame_metadata_does_not_change_raw_solver_evolution() -> None:
    """PulseSchedule metadata should change user-facing accessors without changing raw solver states."""
    simulator, system, qubit = _single_qubit_simulator()
    schedule = _make_schedule(
        qubit,
        PulseArray(
            [
                Arbitrary([0.05 + 0j]),
                VirtualZ(np.pi / 2),
                Arbitrary([0.05 + 0j]),
            ]
        ),
    )
    direct_control = Control(
        target=qubit.label,
        frequency=qubit.frequency,
        waveform=schedule.get_sampled_sequences()[qubit.label],
    )

    schedule_result = simulator.simulate(
        schedule,
        initial_state=system.state({qubit.label: "0"}),
        dt=2.0,
    )
    direct_result = simulator.simulate(
        [direct_control],
        initial_state=system.state({qubit.label: "0"}),
        dt=2.0,
    )
    expected = _rotate_density_matrix(
        system,
        qubit.label,
        direct_result.final_state,
        np.pi / 2,
    )

    assert np.allclose(
        schedule_result.states[-1].full(), direct_result.states[-1].full()
    )
    assert np.allclose(schedule_result.final_state.full(), expected.full())
    assert np.allclose(
        direct_result.final_state.full(), direct_result.states[-1].full()
    )


def test_direct_controls_without_frame_metadata_keep_raw_visualization() -> None:
    """Direct controls without frame metadata should keep the previous raw-state visualization semantics."""
    simulator, system, qubit = _single_qubit_simulator()
    schedule = _make_schedule(
        qubit,
        PulseArray(
            [
                Arbitrary([0 + 0j]),
                VirtualZ(np.pi / 2),
                Arbitrary([0 + 0j]),
            ]
        ),
    )
    control = Control(
        target=qubit.label,
        frequency=qubit.frequency,
        waveform=schedule.get_sampled_sequences()[qubit.label],
    )

    result = simulator.simulate(
        [control],
        initial_state=system.state({qubit.label: "+"}),
        dt=2.0,
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert np.allclose(result.get_bloch_vectors(qubit.label), expected, atol=1e-9)


def test_propagator_and_gate_fidelity_keep_terminal_virtualz_semantics() -> None:
    """Propagator and gate fidelity should continue to treat terminal VirtualZ as a final rotation."""
    simulator, system, qubit = _single_qubit_simulator()
    schedule = _make_schedule(
        qubit,
        PulseArray([Arbitrary([0 + 0j]), VirtualZ(np.pi / 2)]),
    )
    target_unitary = system.get_rotation_matrix({qubit.label: np.pi / 2})

    propagator = simulator.propagator(schedule, dt=2.0)

    assert np.allclose(propagator.full(), qt.to_super(target_unitary).full())
    assert simulator.gate_fidelity(schedule, target_unitary, dt=2.0) == pytest.approx(
        1.0
    )


def _single_qubit_simulator() -> tuple[QuantumSimulator, QuantumSystem, Transmon]:
    qubit = Transmon(
        label="Q00",
        dimension=2,
        frequency=5.0,
        anharmonicity=-0.2,
    )
    system = QuantumSystem(objects=[qubit])
    simulator = QuantumSimulator(system)
    return simulator, system, qubit


def _make_schedule(qubit: Transmon, waveform: PulseArray) -> PulseSchedule:
    schedule = PulseSchedule(
        [
            PulseChannel(
                label=qubit.label,
                frequency=qubit.frequency,
                target=qubit.label,
            )
        ]
    )
    with schedule as pulse_schedule:
        pulse_schedule.add(qubit.label, waveform)
    return schedule


def _rotate_density_matrix(
    system: QuantumSystem,
    label: str,
    density_matrix: qt.Qobj,
    angle: float,
) -> qt.Qobj:
    rotation = system.get_rotation_matrix({label: angle})
    return rotation @ density_matrix @ rotation.dag()

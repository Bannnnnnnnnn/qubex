"""Tests for Stark-driven CR tomography schedules."""

from __future__ import annotations

from typing import cast

import pytest

from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.pulse import FlatTop


class _ContextStub:
    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return {
            "C": "C",
            "T": "T",
            "C_insitu": "C",
            "T_insitu": "T",
        }[target]


class _PulseStub:
    def __init__(self) -> None:
        self.x90_targets: list[str] = []
        self.x180_targets: list[str] = []

    @staticmethod
    def calc_control_amplitude(*, target: str, rabi_rate: float) -> float:
        assert target in {"C", "T"}
        assert rabi_rate == 0.01
        return 0.8

    def x90(self, target: str) -> FlatTop:
        self.x90_targets.append(target)
        return FlatTop(
            duration=4,
            amplitude={
                "C": 0.1,
                "T": 0.2,
                "C_insitu": 0.3,
                "T_insitu": 0.4,
            }[target],
            tau=0,
        )

    def x180(self, target: str) -> FlatTop:
        self.x180_targets.append(target)
        return FlatTop(duration=6, amplitude=0.9, tau=0)


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = _ContextStub()
        self.pulse = _PulseStub()


@pytest.mark.parametrize(
    (
        "stark_drive_qubit",
        "stark_label",
        "control_pulse_label",
        "cr_label",
        "cancel_pulse_label",
        "expected_x90_targets",
    ),
    [
        (
            "control",
            "C_stark",
            "C_insitu",
            "C_insitu-T",
            "T",
            ["C_insitu", "T"],
        ),
        (
            "target",
            "T_stark",
            "C",
            "C-T_insitu",
            "T_insitu",
            ["C", "T_insitu"],
        ),
    ],
)
def test_stark_cr_tomography_keeps_all_control_pulses_inside_stark(
    stark_drive_qubit: sc.StarkDriveQubit,
    stark_label: str,
    control_pulse_label: str,
    cr_label: str,
    cancel_pulse_label: str,
    expected_x90_targets: list[str],
) -> None:
    """Stark CR tomography should wrap preparation, CR, and both basis rotations."""
    stub = _ExperimentStub()

    schedule = sc.stark_cr_tomography_sequence(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        stark_amplitude=0.01,
        stark_drive_qubit=stark_drive_qubit,
        basis="X",
        control_state="1",
        cr_duration=8,
        ramptime=0,
        stark_ramptime=4,
        cr_amplitude=0.6,
        cancel_amplitude=0.7,
        plot=False,
    )

    pulse_ranges = schedule.get_pulse_ranges()
    stark_range = pulse_ranges[stark_label][0]
    x180_range, control_basis_range = pulse_ranges[control_pulse_label]
    cancel_range, target_basis_range = pulse_ranges[cancel_pulse_label]
    cr_range = pulse_ranges[cr_label][0]

    assert stub.pulse.x180_targets == [
        "C_insitu" if stark_drive_qubit == "control" else "C"
    ]
    assert stub.pulse.x90_targets == expected_x90_targets
    assert stark_range.start < x180_range.start
    assert x180_range.stop <= cr_range.start
    assert cr_range == cancel_range
    assert cr_range.stop <= control_basis_range.start
    assert control_basis_range.start == target_basis_range.start
    assert stark_range.stop > max(control_basis_range.stop, target_basis_range.stop)


@pytest.mark.parametrize(
    ("stark_drive_qubit", "stark_label", "cr_label"),
    [
        ("control", "C_stark", "C_insitu-T"),
        ("target", "T_stark", "C-T_insitu"),
    ],
)
def test_stark_cr_tomography_omits_x180_for_zero_control_state(
    stark_drive_qubit: sc.StarkDriveQubit,
    stark_label: str,
    cr_label: str,
) -> None:
    """Stark CR tomography should omit control preparation for the zero state."""
    stub = _ExperimentStub()

    schedule = sc.stark_cr_tomography_sequence(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        stark_amplitude=0.01,
        stark_drive_qubit=stark_drive_qubit,
        basis="Z",
        control_state="0",
        cr_duration=8,
        ramptime=0,
        stark_ramptime=4,
        plot=False,
    )

    pulse_ranges = schedule.get_pulse_ranges()
    stark_range = pulse_ranges[stark_label][0]
    cr_range = pulse_ranges[cr_label][0]

    assert stub.pulse.x180_targets == []
    assert stark_range.start < cr_range.start
    assert stark_range.stop > cr_range.stop

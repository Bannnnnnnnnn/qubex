"""Tests for control-and-spectator Stark CR calibration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

import qubex.contrib as ctb
from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import FlatTop


class _CalibrationNoteStub:
    def __init__(self) -> None:
        self.updated: list[tuple[str, dict[str, Any]]] = []

    def get_cr_param(self, _target: str) -> None:
        return None

    def update_cr_param(self, target: str, params: dict[str, Any]) -> None:
        self.updated.append((target, params))


class _ContextStub:
    def __init__(self) -> None:
        self.calib_note = _CalibrationNoteStub()
        self.reset_qubits: list[str] = []

    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return {"C": "C", "T": "T", "S": "S", "C_insitu": "C"}[target]

    def reset_awg_and_capunits(self, *, qubits: list[str]) -> None:
        self.reset_qubits = qubits


class _PulseStub:
    def __init__(self) -> None:
        self.x90_targets: list[str] = []
        self.x180_targets: list[str] = []
        self.rabi_params = {
            "C": SimpleNamespace(normalize=lambda value: value),
            "T": SimpleNamespace(normalize=lambda value: value),
        }

    @staticmethod
    def calc_control_amplitude(*, target: str, rabi_rate: float) -> float:
        assert target in {"C", "S"}
        return rabi_rate

    @staticmethod
    def calc_rabi_rate(_target: str, control_amplitude: float) -> float:
        return control_amplitude

    def x90(self, target: str) -> FlatTop:
        self.x90_targets.append(target)
        return FlatTop(duration=4, amplitude=0.2, tau=0)

    def x180(self, target: str) -> FlatTop:
        self.x180_targets.append(target)
        return FlatTop(duration=6, amplitude=0.9, tau=0)


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = _ContextStub()
        self.pulse = _PulseStub()
        self.targets = {
            "C": SimpleNamespace(frequency=5.0),
            "T": SimpleNamespace(frequency=5.2),
            "C_insitu": SimpleNamespace(frequency=4.95),
            "C_insitu-T": SimpleNamespace(frequency=5.2),
        }
        self.measurement_service: Any = None


def test_control_spectator_stark_api_is_exported() -> None:
    """Control-and-spectator Stark acquisition should be public from contrib."""
    assert callable(ctb.obtain_cr_params_under_control_and_spectator_stark)


def test_control_spectator_stark_tomography_uses_nested_ramps() -> None:
    """Spectator ramp should enclose the control ramp and dressed tomography body."""
    stub = _ExperimentStub()

    sequence = sc.control_spectator_stark_cr_tomography_sequence(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        control_stark_amplitude=0.02,
        spectator_stark_amplitude=0.01,
        control_stark_ramptime=4,
        spectator_stark_ramptime=6,
        control_state="1",
        basis="X",
        cr_duration=8,
        ramptime=0,
        plot=False,
    )

    ranges = sequence.get_pulse_ranges()
    spectator = ranges["S_stark"][0]
    control = ranges["C_stark"][0]
    body_ranges = [
        pulse_range
        for label in ("C_insitu", "C_insitu-T", "T")
        for pulse_range in ranges[label]
        if pulse_range.stop > pulse_range.start
    ]

    assert set(sequence.labels) == {
        "S_stark",
        "C_stark",
        "C_insitu",
        "C_insitu-T",
        "T",
    }
    assert spectator.start == 0
    assert spectator.stop == sequence.length
    assert control.start == 3
    assert control.stop == sequence.length - 3
    assert min(pulse_range.start for pulse_range in body_ranges) >= 5
    assert max(pulse_range.stop for pulse_range in body_ranges) <= sequence.length - 5
    assert stub.pulse.x180_targets == ["C_insitu"]
    assert stub.pulse.x90_targets == ["C_insitu", "T"]


def test_control_spectator_stark_hamiltonian_uses_both_control_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual-Stark Hamiltonian tomography should combine control states zero and one."""
    stub = _ExperimentStub()
    measured_states: list[str] = []

    class _FitResult(dict[str, Any]):
        def __init__(self, state: str) -> None:
            omega = np.array([1.0, 2.0, 3.0])
            super().__init__(Omega=omega if state == "0" else -omega)

        @staticmethod
        def get_figure(_key: str | None = None) -> go.Figure:
            return go.Figure(data=[go.Scatter(x=[0.0], y=[0.0])])

    def measure(*_args: Any, **kwargs: Any) -> Result:
        state = kwargs["control_state"]
        measured_states.append(state)
        return Result(
            data={
                "effective_drive_range": np.array([0.0]),
                "control_states": np.array([[float(state), 0.0, 1.0]]),
                "fit_result": _FitResult(state),
            }
        )

    monkeypatch.setattr(sc, "_control_spectator_stark_measure_cr_dynamics", measure)
    monkeypatch.setattr(
        sc.viz,
        "make_bloch_vectors_figure",
        lambda *_args: go.Figure(data=[go.Scatter(x=[0.0], y=[0.0])]),
    )
    monkeypatch.setattr(sc, "_bare_control_amplitude", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(sc, "_bare_rabi_rate", lambda *_args, **_kwargs: 0.3)

    result = sc.control_spectator_stark_cr_hamiltonian_tomography(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        spectator_qubit="S",
        control_stark_amplitude=0.02,
        spectator_stark_amplitude=0.01,
        cr_amplitude=0.5,
        reset_awg_and_capunits=False,
        plot=False,
    )

    assert measured_states == ["0", "1"]
    assert np.array_equal(
        result["Omega"],
        np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]),
    )
    assert len(result["fig_c"].data) == 2
    assert len(result["fig_t"].data) == 2


def test_obtain_control_spectator_stark_cr_params_matches_standard_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dual-Stark CR acquisition should return the standard history payload."""
    stub = _ExperimentStub()
    monkeypatch.setattr(sc, "_measurement_sampling_period", lambda _exp: 2.0)
    monkeypatch.setattr(sc, "_bare_control_amplitude", lambda *_args, **_kwargs: 0.4)
    monkeypatch.setattr(
        sc,
        "_control_spectator_stark_update_cr_params",
        lambda *_args, **_kwargs: Result(
            data={
                "cr_param": {
                    "cr_phase": 0.1,
                    "cancel_amplitude": 0.2,
                    "cancel_phase": 0.3,
                },
                "zx90_duration": 16.0,
                "fig_c": object(),
                "fig_t": object(),
                "coeffs": {"IX": 0.01, "IY": 0.02, "ZX": 0.03},
            }
        ),
    )

    result = sc.obtain_cr_params_under_control_and_spectator_stark(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        control_stark_amplitude=0.02,
        spectator_stark_amplitude=0.01,
        n_iterations=1,
        auto_register_cr_channel=False,
        plot=False,
    )

    assert set(result.data) == {
        "params_history",
        "coeffs_history",
        "figs_history",
    }
    assert len(result["params_history"]) == 2
    assert result["params_history"][0]["cr_phase"] == 0.0
    assert result["params_history"][1]["cr_phase"] == 0.1
    assert result.figure is None
    assert result.figures is None

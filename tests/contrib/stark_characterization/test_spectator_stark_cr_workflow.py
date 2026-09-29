"""Tests for spectator-Stark two-qubit workflows."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import pytest

from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.pulse import FlatTop


class _CalibrationNoteStub:
    def __init__(self) -> None:
        self.requested_targets: list[str] = []
        self.update_cr_param: Any = lambda _target, _params: None

    def get_cr_param(
        self,
        target: str,
        *,
        valid_days: int | None = None,
    ) -> dict[str, Any]:
        self.requested_targets.append(target)
        return {
            "target": target,
            "duration": 8.0,
            "ramptime": 0.0,
            "cr_amplitude": 0.6,
            "cr_phase": 0.0,
            "cr_beta": 0.0,
            "cancel_amplitude": 0.7,
            "cancel_phase": 0.0,
            "cancel_beta": 0.0,
            "rotary_amplitude": 0.0,
            "zx_rotation_rate": 0.01,
        }


class _ContextStub:
    calibration_valid_days = 30

    def __init__(self) -> None:
        self.calib_note = _CalibrationNoteStub()

    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return {"C": "C", "T": "T", "S": "S"}[target]


class _PulseStub:
    def __init__(self) -> None:
        self.x90_targets: list[str] = []
        self.x180_targets: list[str] = []
        self.get_pulse_for_state: Any = lambda _target, _state: FlatTop(
            duration=6,
            amplitude=0.5,
            tau=0,
        )

    @staticmethod
    def calc_control_amplitude(*, target: str, rabi_rate: float) -> float:
        assert target == "S"
        assert rabi_rate == 0.01
        return 0.8

    def x90(self, target: str) -> FlatTop:
        self.x90_targets.append(target)
        return FlatTop(duration=4, amplitude=0.2, tau=0)

    def x180(self, target: str) -> FlatTop:
        self.x180_targets.append(target)
        return FlatTop(duration=6, amplitude=0.9, tau=0)


class _CliffordGeneratorStub:
    @staticmethod
    def create_rb_sequences(**_: Any) -> tuple[list[list[str]], list[str]]:
        return [["XI90", "IX90", "ZX90"]], ["XI90"]


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = _ContextStub()
        self.pulse = _PulseStub()
        self.targets: dict[str, Any] = {}
        self.measurement_service: Any = None
        self.calibration_service = SimpleNamespace(
            calc_zx90_coherence_limit=lambda *_: (_ for _ in ()).throw(KeyError())
        )
        self.clifford: dict[str, Any] = {}
        self.benchmarking_service = SimpleNamespace(
            clifford_generator=_CliffordGeneratorStub()
        )


def _assert_inside_stark(
    schedule: Any,
    *,
    stark_label: str = "S_stark",
    ramp_samples: int = 2,
) -> None:
    pulse_ranges = schedule.get_pulse_ranges()
    inner_ranges = [
        pulse_range
        for label, ranges in pulse_ranges.items()
        if label != stark_label
        for pulse_range in ranges
    ]

    assert pulse_ranges[stark_label][0].start == 0
    assert min(pulse_range.start for pulse_range in inner_ranges) >= ramp_samples
    assert max(pulse_range.stop for pulse_range in inner_ranges) <= (
        schedule.length - ramp_samples
    )


def test_spectator_stark_cr_target_keeps_calibration_separate() -> None:
    """Spectator-Stark CR parameters should use an environment-specific key."""
    exp = cast(Experiment, _ExperimentStub())

    assert sc.spectator_stark_cr_target(exp, "C", "T", "S") == ("C-T_under_S_stark")


def test_spectator_stark_cr_tomography_uses_only_bare_pair_gates() -> None:
    """Spectator-Stark tomography should keep all bare pair gates inside Stark."""
    stub = _ExperimentStub()

    schedule = sc.spectator_stark_cr_tomography_sequence(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        spectator_qubit="S",
        stark_amplitude=0.01,
        stark_ramptime=4,
        basis="X",
        control_state="1",
        cr_duration=8,
        ramptime=0,
        cr_amplitude=0.6,
        cancel_amplitude=0.7,
        plot=False,
    )

    assert set(schedule.labels) == {"S_stark", "C", "C-T", "T"}
    assert stub.pulse.x180_targets == ["C"]
    assert stub.pulse.x90_targets == ["C", "T"]
    assert all("insitu" not in label for label in schedule.labels)
    _assert_inside_stark(schedule)


@pytest.mark.parametrize(
    ("builder", "expected_x90_targets"),
    [
        ("spectator_stark_zx90", []),
        ("spectator_stark_bell_state_sequence", ["C", "T", "T"]),
        ("spectator_stark_rb_sequence_2q", ["C", "T"]),
    ],
)
def test_spectator_stark_two_qubit_builders_wrap_bare_gate_bodies(
    builder: str,
    expected_x90_targets: list[str],
) -> None:
    """Spectator-Stark two-qubit builders should wrap only bare gate labels."""
    stub = _ExperimentStub()
    kwargs: dict[str, Any] = {}
    if builder == "spectator_stark_bell_state_sequence":
        kwargs.update(control_basis="X", target_basis="Y")
    elif builder == "spectator_stark_rb_sequence_2q":
        kwargs.update(n=1, seed=1)

    schedule = getattr(sc, builder)(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        spectator_qubit="S",
        stark_amplitude=0.01,
        stark_ramptime=4,
        plot=False,
        **kwargs,
    )

    assert "S" not in schedule.labels
    assert all("insitu" not in label for label in schedule.labels)
    assert stub.pulse.x90_targets == expected_x90_targets
    assert stub.ctx.calib_note.requested_targets == ["C-T_under_S_stark"]
    _assert_inside_stark(schedule)


def test_spectator_stark_workflow_entrypoints_are_available() -> None:
    """Spectator-Stark CR calibration and evaluation entrypoints should exist."""
    assert callable(sc.obtain_cr_params_under_spectator_stark)
    assert callable(sc.calibrate_spectator_stark_zx90)
    assert callable(sc.spectator_stark_bell_state_tomography)
    assert callable(sc.spectator_stark_interleaved_randomized_benchmarking_2q)


def test_obtain_cr_params_under_spectator_stark_matches_standard_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectator-Stark CR acquisition should return the standard history payload."""
    stub = _ExperimentStub()
    stub.targets = {
        "C": SimpleNamespace(frequency=5.0),
        "T": SimpleNamespace(frequency=5.2),
    }
    monkeypatch.setattr(sc, "_measurement_sampling_period", lambda _exp: 2.0)
    monkeypatch.setattr(sc, "_bare_control_amplitude", lambda *_args, **_kwargs: 0.4)
    monkeypatch.setattr(
        sc,
        "spectator_stark_update_cr_params",
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

    result = sc.obtain_cr_params_under_spectator_stark(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        stark_amplitude=0.01,
        n_iterations=1,
        plot=False,
    )

    assert set(result.data) == {
        "params_history",
        "coeffs_history",
        "figs_history",
    }
    assert result.figure is None
    assert result.figures is None
    assert len(result["params_history"]) == 2
    assert result["params_history"][0]["cr_phase"] == 0.0
    assert result["params_history"][1]["cr_phase"] == 0.1


def test_spectator_stark_cr_hamiltonian_tomography_combines_both_control_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hamiltonian tomography should measure and plot control states zero and one."""
    stub = _ExperimentStub()
    stub.targets = {
        "C": SimpleNamespace(frequency=5.0),
        "T": SimpleNamespace(frequency=5.2),
    }
    measured_states: list[str] = []
    bloch_states: list[np.ndarray[Any, Any]] = []

    class _FitResult(dict[str, Any]):
        def __init__(self, state: str) -> None:
            omega = np.array([1.0, 2.0, 3.0])
            super().__init__(Omega=omega if state == "0" else -omega)
            self.state = state

        def get_figure(self, key: str | None = None) -> go.Figure:
            assert key is None
            return go.Figure(
                data=[go.Scatter(x=[0.0], y=[float(self.state)], name=self.state)]
            )

    def measure(*_args: Any, **kwargs: Any) -> Result:
        state = kwargs["control_state"]
        measured_states.append(state)
        state_value = float(state)
        return Result(
            data={
                "effective_drive_range": np.array([0.0]),
                "control_states": np.array([[state_value, 0.0, 1.0]]),
                "fit_result": _FitResult(state),
            }
        )

    def make_bloch_figure(_time: Any, states: Any) -> go.Figure:
        states_array = np.asarray(states)
        bloch_states.append(states_array)
        return go.Figure(
            data=[go.Scatter(x=[0.0], y=[states_array[0, 0]], name="control")]
        )

    monkeypatch.setattr(sc, "spectator_stark_measure_cr_dynamics", measure)
    monkeypatch.setattr(sc.viz, "make_bloch_vectors_figure", make_bloch_figure)
    monkeypatch.setattr(sc, "_bare_control_amplitude", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(sc, "_bare_rabi_rate", lambda *_args, **_kwargs: 0.3)

    result = sc.spectator_stark_cr_hamiltonian_tomography(
        cast(Experiment, stub),
        control_qubit="C",
        target_qubit="T",
        spectator_qubit="S",
        stark_amplitude=0.01,
        cr_amplitude=0.5,
        reset_awg_and_capunits=False,
        plot=False,
    )

    assert measured_states == ["0", "1"]
    assert [states[0, 0] for states in bloch_states] == [0.0, 1.0]
    assert len(result["fig_c"].data) == 2
    assert len(result["fig_t"].data) == 2
    assert result["result_0"]["fit_result"].state == "0"
    assert result["result_1"]["fit_result"].state == "1"


def test_calibrate_spectator_stark_zx90_stores_environment_specific_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectator-Stark ZX90 calibration should store only its contextual CR key."""
    stub = _ExperimentStub()
    stub.targets = {
        "C": SimpleNamespace(frequency=5.0),
        "T": SimpleNamespace(frequency=5.2),
    }
    stored: list[tuple[str, dict[str, Any]]] = []
    stub.ctx.calib_note.update_cr_param = lambda target, params: stored.append(
        (target, params)
    )
    stub.pulse.calc_control_amplitude = lambda **_: 0.2
    stub.pulse.get_pulse_for_state = lambda _target, _state: FlatTop(
        duration=6,
        amplitude=0.5,
        tau=0,
    )
    schedules: list[Any] = []

    def sweep_parameter(sequence: Any, sweep_range: Any, **_: Any) -> Any:
        values = np.asarray(sweep_range)
        schedules.append(sequence(float(values[0])))
        signal = np.linspace(-1.0, 1.0, len(values))
        return SimpleNamespace(
            data={"T": SimpleNamespace(normalized=signal, zvalues=signal)}
        )

    stub.measurement_service = SimpleNamespace(sweep_parameter=sweep_parameter)
    monkeypatch.setattr(sc.fitting, "fit_polynomial", lambda **_: {"root": 0.55})

    result = sc.calibrate_spectator_stark_zx90(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        stark_amplitude=0.01,
        stark_ramptime=4,
        ramptime=0,
        duration=16,
        amplitude_range=np.array([0.4, 0.5, 0.6]),
        use_drag=False,
        plot=False,
    )

    assert set(result.data) == {
        "amplitude_range",
        "signal",
        "root",
        "n1",
        "n3",
        "coherence_limit",
    }
    assert result["coherence_limit"] == {}
    assert [target for target, _ in stored] == ["C-T_under_S_stark"]
    assert len(schedules) == 2
    for schedule in schedules:
        assert set(schedule.labels) == {"S_stark", "C", "C-T", "T"}
        _assert_inside_stark(schedule)


def test_spectator_stark_irb_runs_reference_and_interleaved_with_shared_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectator-Stark IRB should compare runs using the same random seeds."""
    stub = _ExperimentStub()
    clifford = SimpleNamespace(name="ZX90")
    stub.clifford = {"ZX90": clifford}
    calls: list[dict[str, Any]] = []

    def fake_rb(**kwargs: Any) -> Result:
        calls.append(kwargs)
        return Result(data={"C-T_under_S_stark": {"p": 0.99}})

    monkeypatch.setattr(sc, "spectator_stark_rb_experiment_2q", fake_rb)
    fig = object()
    monkeypatch.setattr(
        sc,
        "_interleaved_fit_result",
        lambda **_: Result(
            data={
                "C-T_under_S_stark": {
                    "gate_error": 0.01,
                    "gate_fidelity": 0.99,
                    "gate_fidelity_err": 0.001,
                    "rb_fit_result": {},
                    "irb_fit_result": {},
                }
            },
            figure=fig,  # type: ignore[arg-type]
            figures={"C-T_under_S_stark": fig},  # type: ignore[dict-item]
        ),
    )

    result = sc.spectator_stark_interleaved_randomized_benchmarking_2q(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        stark_amplitude=0.01,
        interleaved_clifford="ZX90",
        n_trials=2,
        seeds=np.array([11, 22]),
        plot=False,
        save_image=False,
    )

    target_result = result["C-T_under_S_stark"]
    assert set(target_result) == {
        "gate_error",
        "gate_fidelity",
        "gate_fidelity_err",
        "rb_fit_result",
        "irb_fit_result",
        "fig",
    }
    assert target_result["fig"] is fig
    assert result.figure is None
    assert result.figures == {"C-T_under_S_stark": fig}
    assert len(calls) == 2
    assert np.array_equal(calls[0]["seeds"], calls[1]["seeds"])
    assert calls[0]["interleaved_clifford"] is None
    assert calls[1]["interleaved_clifford"] is clifford


def test_spectator_stark_bell_tomography_matches_standard_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spectator-Stark Bell tomography should retain the standard figure key."""
    stub = _ExperimentStub()
    probabilities = np.array([0.5, 0.0, 0.0, 0.5])
    rho = np.diag([0.5, 0.0, 0.0, 0.5]).astype(np.complex128)
    fig = object()
    monkeypatch.setattr(
        sc,
        "spectator_stark_measure_bell_state",
        lambda *_args, **_kwargs: Result(
            data={"raw": probabilities, "mitigated": probabilities}
        ),
    )
    monkeypatch.setattr(sc, "mle_fit_density_matrix", lambda _values: rho)
    monkeypatch.setattr(
        sc,
        "plot_ghz_state_tomography",
        lambda **_kwargs: {"figure": fig},
    )

    result = sc.spectator_stark_bell_state_tomography(
        cast(Experiment, stub),
        "C",
        "T",
        "S",
        stark_amplitude=0.01,
        plot=False,
        save_image=False,
    )

    assert set(result.data) == {
        "probabilities",
        "expected_values",
        "density_matrix",
        "fidelity",
        "figure",
    }
    assert dict.__getitem__(result.data, "figure") is fig
    assert result.figure is fig

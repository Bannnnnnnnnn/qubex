"""Tests for functional APIs in `qubex.contrib.experiment.stark_characterization`."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib import (
    ac_stark_shift_spectroscopy,
    ac_stark_shift_spectroscopy_over_time,
    calibrate_spectator_stark_default_pulse,
    calibrate_spectator_stark_drag_amplitude,
    calibrate_spectator_stark_drag_beta,
    calibrate_spectator_stark_drag_hpi_pulse,
    calibrate_spectator_stark_drag_pi_pulse,
    calibrate_spectator_stark_hpi_pulse,
    calibrate_spectator_stark_pi_pulse,
    calibrate_stark_default_pulse,
    calibrate_stark_drag_amplitude,
    calibrate_stark_drag_beta,
    calibrate_stark_drag_hpi_pulse,
    calibrate_stark_drag_pi_pulse,
    calibrate_stark_hpi_pulse,
    calibrate_stark_pi_pulse,
    calibrate_stark_zx90,
    insitu_target,
    make_insitu_channel,
    make_spectator_stark_channel,
    make_stark_channel,
    make_stark_cr_channel,
    obtain_cr_params_under_stark,
    ramsey_experiment_under_stark,
    spectator_stark_chevron_pattern,
    spectator_stark_gate_characterization,
    spectator_stark_rabi_experiment,
    spectator_stark_rabi_sequence,
    spectator_stark_ramsey_experiment,
    spectator_stark_ramsey_sequence,
    spectator_stark_t1_experiment,
    spectator_stark_t1_sequence,
    spectator_stark_t2_experiment,
    spectator_stark_t2_sequence,
    spectator_stark_target,
    stark_bell_state_sequence,
    stark_bell_state_tomography,
    stark_chevron_pattern,
    stark_cnot,
    stark_cr_hamiltonian_tomography,
    stark_cr_target,
    stark_interleaved_purity_benchmarking,
    stark_interleaved_randomized_benchmarking,
    stark_interleaved_randomized_benchmarking_2q,
    stark_ipurity_experiment,
    stark_irb_experiment,
    stark_measure_cr_dynamics,
    stark_obtain_cr_params,
    stark_purity_experiment_1q,
    stark_purity_sequence_1q,
    stark_rabi_experiment,
    stark_rabi_sequence,
    stark_ramsey_experiment,
    stark_ramsey_sequence_under_stark,
    stark_rb_experiment_1q,
    stark_rb_experiment_2q,
    stark_rb_sequence_1q,
    stark_rb_sequence_2q,
    stark_repeat_sequence,
    stark_repeat_sequence_sample,
    stark_t1_experiment,
    stark_t1_sequence_under_stark,
    stark_t2_sequence_under_stark,
    stark_target,
    stark_update_cr_params,
    stark_zx90,
    t1_experiment_under_stark,
    t2_experiment_under_stark,
)
from qubex.contrib.experiment import stark_characterization as sc
from qubex.experiment import Experiment
from qubex.experiment.models import Result
from qubex.experiment.models.rabi_param import RabiParam
from qubex.pulse import Blank, FlatTop, PulseArray, PulseSchedule


class _UtilStub:
    def discretize_time_range(
        self,
        values: np.ndarray,
        sampling_period: float | None = None,
    ) -> np.ndarray:
        return values


class _ExperimentStub:
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            util=_UtilStub(),
        )


class _SpectatorExperimentStub:
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(sampling_period=1.0),
            resolve_qubit_label=lambda target: {"Q16": "Q16", "Q17": "Q17"}[target],
        )


class _SpectatorMeasurementPulseStub:
    @staticmethod
    def calc_control_amplitude(*, target: str, rabi_rate: float) -> float:
        assert target == "Q33"
        assert rabi_rate == 0.05
        return 0.25

    @staticmethod
    def readout(
        target: str,
        *,
        duration: float | None = None,
        amplitude: float | None = None,
        pre_margin: float | None = None,
        post_margin: float | None = None,
        **_: Any,
    ) -> PulseArray:
        assert target == "RQ18"
        assert duration is not None
        assert pre_margin is not None
        assert post_margin is not None
        readout = FlatTop(
            duration=duration,
            amplitude=0.2 if amplitude is None else amplitude,
            tau=0,
            sampling_period=2.0,
        )
        return PulseArray(
            [
                Blank(pre_margin),
                readout.padded(
                    total_duration=duration + post_margin,
                    pad_side="right",
                ),
            ]
        )


class _SpectatorMeasurementExperimentStub:
    def __init__(self) -> None:
        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(
                sampling_period=2.0,
                constraint_profile=SimpleNamespace(
                    enforce_word_alignment=True,
                    word_duration_ns=8.0,
                ),
            ),
            resolve_qubit_label=lambda target: {
                "Q18": "Q18",
                "Q33": "Q33",
            }[target],
            resolve_read_label=lambda target: {"Q18": "RQ18"}[target],
        )
        self.pulse = _SpectatorMeasurementPulseStub()


class _SpectatorSweepExperimentStub:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, Any]] = []
        self.reset_qubits: set[str] | None = None

        def execute(**kwargs: Any) -> SimpleNamespace:
            self.execute_calls.append(kwargs)
            return SimpleNamespace(data={"Q18": [SimpleNamespace(kerneled=1.0 + 0.0j)]})

        self.ctx = SimpleNamespace(
            measurement=SimpleNamespace(execute=execute),
            ordered_qubit_labels=lambda labels: ["Q33", "Q18"],
            state_centers={},
            reset_awg_and_capunits=self._reset_awg_and_capunits,
            get_rabi_param=lambda target: None,
        )

    def _reset_awg_and_capunits(self, *, qubits: set[str]) -> None:
        self.reset_qubits = qubits


class _SpectatorChevronExperimentStub:
    def __init__(self) -> None:
        self.modified_frequency_calls: list[dict[str, float]] = []
        self.targets = {
            "Q18": SimpleNamespace(frequency=4.8),
            "Q18_under_Q33_stark": SimpleNamespace(frequency=5.0),
        }
        self.ctx = SimpleNamespace(
            resolve_qubit_label=lambda target: {
                "Q18": "Q18",
                "Q33": "Q33",
            }[target],
            util=SimpleNamespace(no_output=lambda: nullcontext()),
        )

    @contextmanager
    def modified_frequencies(
        self,
        frequencies: dict[str, float] | None = None,
    ) -> Any:
        point_frequencies = dict(frequencies or {})
        self.modified_frequency_calls.append(point_frequencies)
        original_frequencies = {
            label: self.targets[label].frequency
            for label in point_frequencies
            if label in self.targets
        }
        for label, frequency in point_frequencies.items():
            if label in self.targets:
                self.targets[label].frequency = frequency
        try:
            yield
        finally:
            for label, frequency in original_frequencies.items():
                self.targets[label].frequency = frequency


class _SpectatorGateWorkflowContextStub:
    def __init__(self) -> None:
        self.stored_rabi_params: dict[str, RabiParam] = {}

    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return {"Q18": "Q18", "Q33": "Q33"}[target]

    def store_rabi_params(
        self,
        rabi_params: dict[str, RabiParam],
        r2_threshold: float | None = None,
    ) -> None:
        self.stored_rabi_params.update(rabi_params)


class _SpectatorGateWorkflowExperimentStub:
    def __init__(self) -> None:
        self.ctx = _SpectatorGateWorkflowContextStub()
        self.targets = {"Q18_under_Q33_stark": SimpleNamespace(frequency=5.0)}


def test_all_stark_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all stark helpers are available."""
    assert callable(ac_stark_shift_spectroscopy)
    assert callable(ac_stark_shift_spectroscopy_over_time)
    assert callable(stark_t1_experiment)
    assert callable(stark_ramsey_experiment)
    assert callable(stark_target)
    assert callable(insitu_target)
    assert callable(make_stark_channel)
    assert callable(make_insitu_channel)
    assert callable(spectator_stark_target)
    assert callable(make_spectator_stark_channel)
    assert callable(stark_cr_target)
    assert callable(make_stark_cr_channel)
    assert callable(spectator_stark_rabi_sequence)
    assert callable(spectator_stark_rabi_experiment)
    assert callable(spectator_stark_chevron_pattern)
    assert callable(spectator_stark_t1_sequence)
    assert callable(spectator_stark_t2_sequence)
    assert callable(spectator_stark_ramsey_sequence)
    assert callable(spectator_stark_t1_experiment)
    assert callable(spectator_stark_t2_experiment)
    assert callable(spectator_stark_ramsey_experiment)
    assert callable(spectator_stark_gate_characterization)
    assert callable(calibrate_stark_default_pulse)
    assert callable(calibrate_stark_hpi_pulse)
    assert callable(calibrate_stark_pi_pulse)
    assert callable(calibrate_stark_zx90)
    assert callable(calibrate_stark_drag_amplitude)
    assert callable(calibrate_stark_drag_beta)
    assert callable(calibrate_stark_drag_hpi_pulse)
    assert callable(calibrate_stark_drag_pi_pulse)
    assert callable(calibrate_spectator_stark_default_pulse)
    assert callable(calibrate_spectator_stark_hpi_pulse)
    assert callable(calibrate_spectator_stark_pi_pulse)
    assert callable(calibrate_spectator_stark_drag_amplitude)
    assert callable(calibrate_spectator_stark_drag_beta)
    assert callable(calibrate_spectator_stark_drag_hpi_pulse)
    assert callable(calibrate_spectator_stark_drag_pi_pulse)
    assert callable(t1_experiment_under_stark)
    assert callable(t2_experiment_under_stark)
    assert callable(ramsey_experiment_under_stark)
    assert callable(stark_rabi_experiment)
    assert callable(stark_rabi_sequence)
    assert callable(stark_repeat_sequence)
    assert callable(stark_repeat_sequence_sample)
    assert callable(stark_chevron_pattern)
    assert callable(stark_zx90)
    assert callable(stark_cnot)
    assert callable(stark_bell_state_sequence)
    assert callable(stark_bell_state_tomography)
    assert callable(stark_rb_experiment_1q)
    assert callable(stark_rb_sequence_1q)
    assert callable(stark_rb_experiment_2q)
    assert callable(stark_rb_sequence_2q)
    assert callable(stark_purity_experiment_1q)
    assert callable(stark_purity_sequence_1q)
    assert callable(stark_irb_experiment)
    assert callable(stark_ipurity_experiment)
    assert callable(stark_interleaved_randomized_benchmarking)
    assert callable(stark_interleaved_randomized_benchmarking_2q)
    assert callable(stark_interleaved_purity_benchmarking)
    assert callable(stark_t1_sequence_under_stark)
    assert callable(stark_t2_sequence_under_stark)
    assert callable(stark_ramsey_sequence_under_stark)
    assert callable(stark_measure_cr_dynamics)
    assert callable(stark_cr_hamiltonian_tomography)
    assert callable(stark_update_cr_params)
    assert callable(obtain_cr_params_under_stark)
    assert callable(stark_obtain_cr_params)


def test_spectator_stark_rabi_sequence_splits_stark_and_control_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a spectator Stark Rabi sequence, then Stark and control labels are distinct."""
    wrapped: dict[str, Any] = {}
    drive: dict[str, Any] = {}

    def fake_stark_drive_amplitude(
        exp: Experiment,
        *,
        target: str,
        stark_amplitude: float,
    ) -> float:
        drive.update(target=target, stark_amplitude=stark_amplitude)
        return 0.25

    def fake_stark_wrapped_schedule(**kwargs: Any) -> object:
        wrapped.update(kwargs)
        return object()

    monkeypatch.setattr(sc, "_stark_drive_amplitude", fake_stark_drive_amplitude)
    monkeypatch.setattr(sc, "_stark_wrapped_schedule", fake_stark_wrapped_schedule)
    monkeypatch.setattr(
        sc,
        "_plot_sequence_sample",
        lambda sequence, **kwargs: sequence,
    )

    result = sc.spectator_stark_rabi_sequence(
        cast(Experiment, _SpectatorExperimentStub()),
        target="Q16",
        stark_drive_target="Q17",
        stark_amplitude=0.05,
        amplitude=0.3,
        stark_ramptime=20,
        duration=40,
        plot=False,
    )

    assert result is not None
    assert drive == {"target": "Q17", "stark_amplitude": 0.05}
    assert wrapped["stark_label"] == "Q17_stark"
    assert wrapped["insitu_label"] == "Q16_under_Q17_stark"
    assert wrapped["stark_amplitude"] == 0.25
    assert wrapped["stark_ramptime"] == 20.0


def test_spectator_stark_rabi_measurement_sequence_keeps_readout_inside_stark_flat_top() -> (
    None
):
    """Given a spectator Stark Rabi schedule, then readout happens before Stark fall."""
    sequence = sc._spectator_stark_rabi_measurement_sequence(  # noqa: SLF001
        cast(Experiment, _SpectatorMeasurementExperimentStub()),
        target="Q18",
        stark_drive_target="Q33",
        stark_amplitude=0.05,
        amplitude=0.3,
        stark_ramptime=10,
        duration=40,
        ramptime=4,
        readout_duration=32,
        readout_pre_margin=8,
        readout_post_margin=8,
        plot=False,
    )

    ranges = sequence.get_pulse_ranges(sequence.labels)
    stark_range = ranges["Q33_stark"][0]
    control_range = ranges["Q18_under_Q33_stark"][0]
    readout_range = ranges["RQ18"][0]

    assert sequence.duration == 122.0
    assert (control_range.start, control_range.stop) == (5, 29)
    assert (readout_range.start, readout_range.stop) == (36, 56)
    assert stark_range.start == 0
    assert stark_range.stop == 61
    assert readout_range.start * 2.0 % 8.0 == 0
    assert len(readout_range) * 2.0 % 8.0 == 0
    assert readout_range.stop <= stark_range.stop


def test_spectator_stark_rabi_sweep_executes_explicit_readout_schedule() -> None:
    """Given a spectator Stark Rabi sweep, then final readout is not appended."""
    stub = _SpectatorSweepExperimentStub()

    def sequence(_: float) -> PulseSchedule:
        return PulseSchedule(["Q33_stark", "Q18_under_Q33_stark", "RQ18"])

    result = sc._run_spectator_stark_rabi_sweep(  # noqa: SLF001
        cast(Experiment, stub),
        sequence=sequence,
        sweep_range=np.array([0.0, 8.0]),
        frequencies={"Q18_under_Q33_stark": 5.0},
        n_shots=128,
        shot_interval=1024.0,
        plot=False,
        title="title",
        xlabel="x",
        ylabel="y",
    )

    assert stub.reset_qubits == {"Q33", "Q18"}
    assert len(stub.execute_calls) == 2
    assert all(call["final_measurement"] is False for call in stub.execute_calls)
    assert all(call["shot_averaging"] is True for call in stub.execute_calls)
    assert all(call["time_integration"] is False for call in stub.execute_calls)
    assert all(
        call["schedule"].get_frequency("Q18_under_Q33_stark") == 5.0
        for call in stub.execute_calls
    )
    assert np.array_equal(result.data["Q18"].data, np.array([1.0 + 0.0j] * 2))


def test_spectator_stark_chevron_uses_modified_target_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given spectator chevron detuning, then Rabi runs under modified frequency."""
    stub = _SpectatorChevronExperimentStub()
    rabi_call_frequencies: list[float] = []

    def fake_spectator_stark_rabi_experiment(
        exp: Experiment,
        target: str,
        stark_drive_target: str,
        **kwargs: Any,
    ) -> SimpleNamespace:
        assert target == "Q18"
        assert stark_drive_target == "Q33"
        assert "frequencies" not in kwargs
        frequency = cast(Any, exp).targets["Q18_under_Q33_stark"].frequency
        rabi_call_frequencies.append(frequency)
        return SimpleNamespace(
            rabi_params={"Q18": SimpleNamespace(frequency=frequency)},
            data={
                "Q18": SimpleNamespace(
                    normalized=np.asarray([frequency, frequency + 1.0]),
                    rabi_param=None,
                )
            },
        )

    def fake_fit_detuned_rabi(**kwargs: Any) -> dict[str, float]:
        np.testing.assert_allclose(kwargs["control_frequencies"], [4.99, 5.01])
        np.testing.assert_allclose(kwargs["rabi_frequencies"], [4.99, 5.01])
        return {"f_resonance": 5.01}

    monkeypatch.setattr(
        sc,
        "spectator_stark_rabi_experiment",
        fake_spectator_stark_rabi_experiment,
    )
    monkeypatch.setattr(sc.fitting, "fit_detuned_rabi", fake_fit_detuned_rabi)
    monkeypatch.setattr(
        sc,
        "_stark_drive_amplitude",
        lambda exp, *, target, stark_amplitude: stark_amplitude,
    )
    monkeypatch.setattr(
        sc, "make_spectator_stark_channel", lambda *args, **kwargs: None
    )

    result = sc.spectator_stark_chevron_pattern(
        cast(Experiment, stub),
        target="Q18",
        stark_drive_target="Q33",
        stark_amplitude=0.1,
        detuning_range=[-0.01, 0.01],
        time_range=[0.0, 8.0],
        frequencies={"Q18_under_Q33_stark": 5.0},
        amplitude=0.3,
        rabi_params={"Q18": SimpleNamespace(frequency=0.1)},
        n_shots=1,
        shot_interval=1024.0,
        plot=False,
        save_image=False,
    )

    np.testing.assert_allclose(rabi_call_frequencies, [4.99, 5.01])
    np.testing.assert_allclose(result.data["control_frequencies"], [4.99, 5.01])
    assert [
        call["Q18_under_Q33_stark"] for call in stub.modified_frequency_calls
    ] == pytest.approx([4.99, 5.01])
    assert stub.targets["Q18_under_Q33_stark"].frequency == 5.0


def test_spectator_stark_gate_characterization_runs_full_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given spectator Stark workflow, then Rabi, gates, and coherence run in order."""
    stub = _SpectatorGateWorkflowExperimentStub()
    calls: list[str] = []
    rabi_param = RabiParam(
        target="Q18",
        amplitude=1.0,
        frequency=0.05,
        phase=0.0,
        offset=0.0,
        noise=0.0,
        angle=0.0,
        distance=1.0,
        r2=0.99,
        reference_phase=0.0,
    )

    def fake_rabi_experiment(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append("rabi")
        assert kwargs["store_params"] is False
        assert kwargs["target"] == "Q18"
        assert kwargs["stark_drive_target"] == "Q33"
        return SimpleNamespace(rabi_params={"Q18": rabi_param})

    def fake_result(name: str) -> Result:
        calls.append(name)
        return Result(data={"name": name})

    monkeypatch.setattr(sc, "spectator_stark_rabi_experiment", fake_rabi_experiment)
    monkeypatch.setattr(
        sc,
        "calibrate_spectator_stark_hpi_pulse",
        lambda *args, **kwargs: fake_result("hpi"),
    )
    monkeypatch.setattr(
        sc,
        "calibrate_spectator_stark_pi_pulse",
        lambda *args, **kwargs: fake_result("pi"),
    )
    monkeypatch.setattr(
        sc,
        "calibrate_spectator_stark_drag_hpi_pulse",
        lambda *args, **kwargs: fake_result("drag_hpi"),
    )
    monkeypatch.setattr(
        sc,
        "calibrate_spectator_stark_drag_pi_pulse",
        lambda *args, **kwargs: fake_result("drag_pi"),
    )
    monkeypatch.setattr(
        sc,
        "spectator_stark_t1_experiment",
        lambda *args, **kwargs: fake_result("t1"),
    )
    monkeypatch.setattr(
        sc,
        "spectator_stark_t2_experiment",
        lambda *args, **kwargs: fake_result("t2"),
    )
    monkeypatch.setattr(
        sc,
        "spectator_stark_ramsey_experiment",
        lambda *args, **kwargs: fake_result("ramsey"),
    )

    result = sc.spectator_stark_gate_characterization(
        cast(Experiment, stub),
        target="Q18",
        stark_drive_target="Q33",
        stark_amplitude=0.1,
        n_shots=10,
        shot_interval=1024.0,
        plot=False,
    )

    assert calls == ["rabi", "hpi", "pi", "drag_hpi", "drag_pi", "t1", "t2", "ramsey"]
    assert result.data["control_target"] == "Q18_under_Q33_stark"
    stored = stub.ctx.stored_rabi_params["Q18_under_Q33_stark"]
    assert stored.target == "Q18_under_Q33_stark"
    assert stored.frequency == rabi_param.frequency
    assert set(result.data["gate_calibrations"]) == {
        "hpi",
        "pi",
        "drag_hpi",
        "drag_pi",
    }
    assert set(result.data["coherence"]) == {"t1", "t2", "ramsey"}


def test_ac_stark_shift_spectroscopy_can_plot_applied_amplitude_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given amplitude axis, when wait times are swept, then P1 is keyed by Stark amplitude."""
    calls: list[dict[str, Any]] = []

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(kwargs)
        amplitude = np.asarray(kwargs["stark_amplitude_range"], dtype=float)
        wait_time = float(kwargs["wait_time"])
        return Result(data={"p1": amplitude + wait_time / 1000})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1, 0.2],
        stark_shift_model="amplitude",
        wait_time_range=[10, 20],
        n_shots=1,
        plot=False,
    )

    np.testing.assert_allclose(result.data["x_axis_range"], [0.0, 0.1, 0.2])
    assert result.data["x_axis"] == "stark_amplitude"
    assert np.asarray(result.data["p1"]).shape == (2, 3)
    assert result.figure is not None
    heatmap_trace = cast(Any, result.figure.data[0])
    assert heatmap_trace.type == "heatmap"
    assert result.figure.layout.xaxis.title.text == "Stark amplitude (GHz)"
    assert [call["wait_time"] for call in calls] == [10, 20]


def test_ac_stark_shift_spectroscopy_over_time_uses_line_plot_for_one_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one repeated measurement, when measured over time, then P1 is a line plot."""

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        return Result(data={"p1": np.asarray(kwargs["stark_amplitude_range"])})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy_over_time(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1, 0.2],
        stark_shift_model="amplitude",
        wait_time=100,
        n_iterations=1,
        n_shots=1,
        plot=False,
    )

    assert result.data["n_iterations"] == 1
    assert np.asarray(result.data["p1"]).shape == (1, 3)
    assert result.figure is not None
    scatter_trace = cast(Any, result.figure.data[0])
    assert scatter_trace.type == "scatter"
    assert result.figure.layout.yaxis.title.text == "P1"


def test_ac_stark_shift_spectroscopy_over_time_uses_heatmap_for_repetitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given repeated measurements, when repeated, then rows are stacked by iteration."""
    calls: list[dict[str, Any]] = []

    def fake_stark_p1_spectroscopy(
        exp: Experiment,
        target: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(kwargs)
        iteration = len(calls) - 1
        amplitude = np.asarray(kwargs["stark_amplitude_range"], dtype=float)
        return Result(data={"p1": amplitude + 0.01 * iteration})

    monkeypatch.setattr(sc, "stark_p1_spectroscopy", fake_stark_p1_spectroscopy)

    result = sc.ac_stark_shift_spectroscopy_over_time(
        cast(Experiment, _ExperimentStub()),
        "Q00",
        stark_detuning=0.15,
        stark_amplitude_range=[0.0, 0.1],
        stark_shift_model="amplitude",
        wait_time=100,
        n_iterations=2,
        n_shots=1,
        plot=False,
    )

    assert np.asarray(result.data["p1"]).shape == (2, 2)
    np.testing.assert_allclose(result.data["iteration_range"], [1, 2])
    assert len(result.data["elapsed_time_s"]) == 2
    assert len(calls) == 2
    assert result.figure is not None
    heatmap_trace = cast(Any, result.figure.data[0])
    assert heatmap_trace.type == "heatmap"
    np.testing.assert_allclose(heatmap_trace.y, [1, 2])
    assert result.figure.layout.yaxis.title.text == "Iteration"


def test_ac_stark_amplitude_axis_requires_single_detuning() -> None:
    """Given multiple Stark detunings, when amplitude axis is requested, then raise."""
    with pytest.raises(ValueError, match="requires a single Stark detuning"):
        sc.ac_stark_shift_spectroscopy_over_time(
            cast(Experiment, _ExperimentStub()),
            "Q00",
            stark_detuning=[-0.15, 0.15],
            stark_shift_model="amplitude",
            wait_time=100,
            n_iterations=1,
            n_shots=1,
            plot=False,
        )

"""Tests for leakage randomized benchmarking contrib APIs."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

from qubex.analysis import FitResult, FitStatus
from qubex.contrib import (
    fit_computational_leakage_rb,
    fit_leakage_rb,
    fit_stark_pulse_leakage,
    interleaved_leakage_rb_experiment_1q,
    interleaved_leakage_rb_experiment_2q,
    leakage_randomized_benchmarking,
    leakage_rb_experiment_1q,
    leakage_rb_experiment_2q,
    stark_interleaved_leakage_rb_experiment_1q,
    stark_interleaved_leakage_rb_experiment_2q,
    stark_leakage_randomized_benchmarking,
    stark_leakage_rb_experiment_1q,
    stark_leakage_rb_experiment_2q,
    stark_pair_pulse_leakage_experiment,
    stark_pulse_leakage_experiment,
)
from qubex.experiment import Experiment
from qubex.experiment.models import Result

lrb = importlib.import_module(
    "qubex.contrib.experiment.leakage_randomized_benchmarking"
)


class _ExperimentSystemStub:
    @staticmethod
    def get_target(target: str) -> SimpleNamespace:
        return SimpleNamespace(is_cr=target == "Q20-Q17")


class _ContextStub:
    experiment_system = _ExperimentSystemStub()

    @staticmethod
    def cr_pair(target: str) -> tuple[str, str]:
        assert target == "Q20-Q17"
        return "Q20", "Q17"


class _ExperimentStub:
    ctx = _ContextStub()


class _StarkPulseContextStub:
    experiment_system = _ExperimentSystemStub()

    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return str(target)

    @staticmethod
    def reset_awg_and_capunits(*, qubits: set[str]) -> None:
        assert qubits == {"Q37"}


class _ProbabilityResultStub:
    @staticmethod
    def get_probabilities(targets: list[str]) -> dict[str, float]:
        assert targets == ["Q37"]
        return {"0": 0.9, "1": 0.05, "2": 0.05}

    @staticmethod
    def get_mitigated_probabilities(targets: list[str]) -> dict[str, float]:
        assert targets == ["Q37"]
        return {"0": 0.88, "1": 0.04, "2": 0.08}


class _MeasurementServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def measure(self, **kwargs: Any) -> _ProbabilityResultStub:
        self.calls.append(kwargs)
        return _ProbabilityResultStub()


class _StarkPulseExperimentStub:
    def __init__(self) -> None:
        self.ctx = _StarkPulseContextStub()
        self.classifiers = {"Q37": SimpleNamespace(n_states=3)}
        self.measurement_service = _MeasurementServiceStub()


class _StarkPairPulseContextStub:
    @staticmethod
    def resolve_qubit_label(target: str) -> str:
        return str(target)

    @staticmethod
    def reset_awg_and_capunits(*, qubits: set[str]) -> None:
        assert qubits == {"Q36", "Q37"}


class _PairProbabilityResultStub:
    @staticmethod
    def get_probabilities(targets: list[str]) -> dict[str, float]:
        assert targets == ["Q36", "Q37"]
        return {"00": 0.9, "01": 0.05, "20": 0.05}

    @staticmethod
    def get_mitigated_probabilities(targets: list[str]) -> dict[str, float]:
        assert targets == ["Q36", "Q37"]
        return {"00": 0.88, "01": 0.04, "20": 0.08}


class _PairMeasurementServiceStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def measure(self, **kwargs: Any) -> _PairProbabilityResultStub:
        self.calls.append(kwargs)
        return _PairProbabilityResultStub()


class _StarkPairPulseExperimentStub:
    def __init__(self) -> None:
        self.ctx = _StarkPairPulseContextStub()
        self.classifiers = {
            "Q36": SimpleNamespace(n_states=3),
            "Q37": SimpleNamespace(n_states=3),
        }
        self.measurement_service = _PairMeasurementServiceStub()


def test_all_leakage_rb_functions_are_exported_from_contrib() -> None:
    """Given contrib package, when imported, then all LRB helpers are available."""
    assert callable(fit_computational_leakage_rb)
    assert callable(fit_leakage_rb)
    assert callable(fit_stark_pulse_leakage)
    assert callable(leakage_rb_experiment_1q)
    assert callable(leakage_rb_experiment_2q)
    assert callable(interleaved_leakage_rb_experiment_1q)
    assert callable(interleaved_leakage_rb_experiment_2q)
    assert callable(leakage_randomized_benchmarking)
    assert callable(stark_leakage_rb_experiment_1q)
    assert callable(stark_leakage_rb_experiment_2q)
    assert callable(stark_interleaved_leakage_rb_experiment_1q)
    assert callable(stark_interleaved_leakage_rb_experiment_2q)
    assert callable(stark_leakage_randomized_benchmarking)
    assert callable(stark_pair_pulse_leakage_experiment)
    assert callable(stark_pulse_leakage_experiment)


def test_stark_leakage_randomized_benchmarking_dispatches_1q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a 1Q target, then Stark LRB dispatches to the 1Q implementation."""
    calls: list[dict[str, Any]] = []

    def fake_1q(
        exp: Experiment,
        targets: list[str],
        *,
        stark_amplitude: float,
        **kwargs: Any,
    ) -> Result:
        calls.append(
            {
                "exp": exp,
                "targets": targets,
                "stark_amplitude": stark_amplitude,
                "kwargs": kwargs,
            }
        )
        return Result(data={"Q20": {"ok": True}})

    monkeypatch.setattr(lrb, "stark_leakage_rb_experiment_1q", fake_1q)
    exp = cast(Experiment, _ExperimentStub())

    result = lrb.stark_leakage_randomized_benchmarking(
        exp,
        "Q20",
        stark_amplitude=0.085,
        n_trials=1,
    )

    assert result.data == {"Q20": {"ok": True}}
    assert calls == [
        {
            "exp": exp,
            "targets": ["Q20"],
            "stark_amplitude": 0.085,
            "kwargs": {"n_trials": 1},
        }
    ]


def test_stark_leakage_randomized_benchmarking_dispatches_2q_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a CR target and interleaved gate, then Stark LRB resolves the CR pair."""
    calls: list[dict[str, Any]] = []

    def fake_2q(
        exp: Experiment,
        *,
        control_qubit: str,
        target_qubit: str,
        stark_amplitude: float,
        stark_drive_qubit: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(
            {
                "exp": exp,
                "control_qubit": control_qubit,
                "target_qubit": target_qubit,
                "stark_amplitude": stark_amplitude,
                "stark_drive_qubit": stark_drive_qubit,
                "kwargs": kwargs,
            }
        )
        return Result(data={"Q20-Q17": {"ok": True}})

    monkeypatch.setattr(lrb, "stark_interleaved_leakage_rb_experiment_2q", fake_2q)
    exp = cast(Experiment, _ExperimentStub())

    result = lrb.stark_leakage_randomized_benchmarking(
        exp,
        "Q20-Q17",
        stark_amplitude=0.085,
        stark_drive_qubit="target",
        interleaved_clifford="ZX90",
    )

    assert result.data == {"Q20-Q17": {"ok": True}}
    assert calls == [
        {
            "exp": exp,
            "control_qubit": "Q20",
            "target_qubit": "Q17",
            "stark_amplitude": 0.085,
            "stark_drive_qubit": "target",
            "kwargs": {"interleaved_clifford": "ZX90"},
        }
    ]


def test_stark_leakage_rb_experiment_2q_dispatches_interleaved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given interleaved Clifford, then direct 2Q Stark LRB runs paired SRB/IRB."""
    calls: list[dict[str, Any]] = []

    def fake_interleaved(
        exp: Experiment,
        *,
        control_qubit: str,
        target_qubit: str,
        stark_amplitude: float,
        interleaved_clifford: str,
        stark_drive_qubit: str,
        **kwargs: Any,
    ) -> Result:
        calls.append(
            {
                "exp": exp,
                "control_qubit": control_qubit,
                "target_qubit": target_qubit,
                "stark_amplitude": stark_amplitude,
                "interleaved_clifford": interleaved_clifford,
                "stark_drive_qubit": stark_drive_qubit,
                "kwargs": kwargs,
            }
        )
        return Result(data={"Q20-Q17": {"ok": True}})

    monkeypatch.setattr(
        lrb,
        "stark_interleaved_leakage_rb_experiment_2q",
        fake_interleaved,
    )
    exp = cast(Experiment, _ExperimentStub())

    result = lrb.stark_leakage_rb_experiment_2q(
        exp,
        "Q20",
        "Q17",
        stark_amplitude=0.085,
        stark_drive_qubit="target",
        interleaved_clifford="ZX90",
        n_trials=1,
    )

    assert result.data == {"Q20-Q17": {"ok": True}}
    assert calls == [
        {
            "exp": exp,
            "control_qubit": "Q20",
            "target_qubit": "Q17",
            "stark_amplitude": 0.085,
            "interleaved_clifford": "ZX90",
            "stark_drive_qubit": "target",
            "kwargs": {
                "stark_ramptime": None,
                "n_cliffords_range": None,
                "n_trials": 1,
                "seeds": None,
                "max_n_cliffords": None,
                "x90": None,
                "zx90": None,
                "interleaved_waveform": None,
                "mitigate_readout": None,
                "n_shots": None,
                "shot_interval": None,
                "xaxis_type": None,
                "plot": None,
                "save_image": None,
            },
        }
    ]


def test_stark_pulse_leakage_experiment_measures_stark_only_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a Stark-only leakage request, then it measures Stark pulse schedules."""
    exp = _StarkPulseExperimentStub()
    fit_calls: list[dict[str, Any]] = []

    def fake_fit_stark_pulse_leakage(**kwargs: Any) -> FitResult:
        fit_calls.append(kwargs)
        return FitResult(
            status=FitStatus.SUCCESS,
            data={
                "stark_leakage_rate_per_ns": 1e-4,
                "stark_leakage_rate_per_ns_err": 1e-5,
                "stark_seepage_rate_per_ns": 2e-4,
                "stark_seepage_rate_per_ns_err": 2e-5,
                "stark_leakage_plus_seepage_per_ns": 3e-4,
                "stark_leakage_plus_seepage_per_ns_err": 3e-5,
            },
        )

    monkeypatch.setattr(
        lrb,
        "_stark_drive_amplitude",
        lambda exp, *, target, stark_amplitude: stark_amplitude,
    )
    monkeypatch.setattr(lrb, "fit_stark_pulse_leakage", fake_fit_stark_pulse_leakage)

    result = lrb.stark_pulse_leakage_experiment(
        cast(Experiment, exp),
        "Q37",
        stark_amplitude=0.08,
        duration_range=[0.0, 8.0],
        initial_states=("0", "1"),
        measure_reference=False,
        n_trials=1,
        n_shots=4,
        shot_interval=1024.0,
        reset_awg_and_capunits=False,
        plot=False,
        save_image=False,
    )

    data = result.data["Q37"]
    assert data["stark_leakage_rate_per_ns"] == 1e-4
    assert data["reference"] is None
    assert data["stark"]["state_population_trials"].shape == (2, 2, 3)
    assert len(exp.measurement_service.calls) == 4
    assert [call["initial_states"] for call in exp.measurement_service.calls] == [
        {"Q37": "0"},
        {"Q37": "1"},
        {"Q37": "0"},
        {"Q37": "1"},
    ]
    assert all(
        call["sequence"].labels == ["Q37_stark"]
        for call in exp.measurement_service.calls
    )
    assert fit_calls[0]["duration"].tolist() == [0.0, 8.0]
    assert fit_calls[0]["leakage_population"].tolist() == [0.05, 0.05]


def test_stark_pair_pulse_leakage_experiment_measures_pair_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a pair Stark-only request, then it classifies the 2Q pair."""
    exp = _StarkPairPulseExperimentStub()
    fit_calls: list[dict[str, Any]] = []

    def fake_fit_stark_pulse_leakage(**kwargs: Any) -> FitResult:
        fit_calls.append(kwargs)
        return FitResult(
            status=FitStatus.SUCCESS,
            data={
                "stark_leakage_rate_per_ns": 1e-4,
                "stark_leakage_rate_per_ns_err": 1e-5,
                "stark_seepage_rate_per_ns": 2e-4,
                "stark_seepage_rate_per_ns_err": 2e-5,
                "stark_leakage_plus_seepage_per_ns": 3e-4,
                "stark_leakage_plus_seepage_per_ns_err": 3e-5,
            },
        )

    monkeypatch.setattr(
        lrb,
        "_stark_drive_amplitude",
        lambda exp, *, target, stark_amplitude: stark_amplitude,
    )
    monkeypatch.setattr(lrb, "fit_stark_pulse_leakage", fake_fit_stark_pulse_leakage)

    result = lrb.stark_pair_pulse_leakage_experiment(
        cast(Experiment, exp),
        control_qubit="Q36",
        target_qubit="Q37",
        stark_drive_qubit="target",
        stark_amplitude=0.08,
        duration_range=[0.0, 8.0],
        initial_states=("00", "11"),
        measure_reference=False,
        n_trials=1,
        n_shots=4,
        shot_interval=1024.0,
        reset_awg_and_capunits=False,
        plot=False,
        save_image=False,
    )

    data = result.data["Q36-Q37_insitu"]
    assert data["measured_qubits"].tolist() == ["Q36", "Q37"]
    assert data["stark_drive_target"] == "Q37"
    assert data["stark"]["state_population_trials"].shape == (2, 2, 9)
    assert len(exp.measurement_service.calls) == 4
    assert [call["initial_states"] for call in exp.measurement_service.calls] == [
        {"Q36": "0", "Q37": "0"},
        {"Q36": "1", "Q37": "1"},
        {"Q36": "0", "Q37": "0"},
        {"Q36": "1", "Q37": "1"},
    ]
    assert all(
        call["sequence"].labels == ["Q36", "Q37_stark", "Q37"]
        for call in exp.measurement_service.calls
    )
    assert fit_calls[0]["target"] == "Q36-Q37_insitu"
    assert fit_calls[0]["duration"].tolist() == [0.0, 8.0]
    assert fit_calls[0]["leakage_population"].tolist() == [0.05, 0.05]

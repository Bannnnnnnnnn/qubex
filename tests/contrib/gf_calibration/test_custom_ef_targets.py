"""Tests for custom EF targets in contributed GF calibration helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import qubex.contrib.experiment.gf_calibration as gf_calibration
from qubex.analysis import FitStatus
from qubex.contrib.experiment.gf_calibration import (
    calibrate_gf_pulse,
    gf_rabi_experiment,
    gf_ramsey_experiment,
    obtain_gf_rabi_params,
)
from qubex.experiment.models.experiment_result import ExperimentResult, SweepData
from qubex.experiment.models.rabi_param import RabiParam
from qubex.pulse import Blank
from qubex.system import TargetType


class _FitResult(dict[str, float]):
    def __init__(self) -> None:
        super().__init__(
            amplitude=1.0,
            frequency=0.2,
            phase=0.0,
            offset=0.0,
            noise=0.0,
            angle=0.0,
            distance=1.0,
            r2=0.99,
            reference_phase=0.0,
            tau=1000.0,
            f=0.001,
            phi=1.0,
        )
        self.status = FitStatus.SUCCESS


class _MeasurementService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def sweep_parameter(
        self,
        *,
        sequence: Any,
        sweep_range: Any,
        frequencies: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> ExperimentResult[SweepData]:
        schedule = sequence(int(np.asarray(sweep_range)[0]))
        self.calls.append(
            {
                "labels": list(schedule.labels),
                "frequencies": frequencies or {},
                "kwargs": kwargs,
            }
        )
        return ExperimentResult(
            data={
                "Q28": SweepData(
                    target="Q28",
                    data=np.asarray([0.0 + 0.0j]),
                    sweep_range=np.asarray(sweep_range),
                )
            }
        )


class _Params:
    def __init__(self) -> None:
        self.ef_amplitude_queries: list[str] = []

    def get_ef_control_amplitude(self, qubit: str) -> float:
        self.ef_amplitude_queries.append(qubit)
        return 0.125


def _rabi_param(target: str) -> RabiParam:
    return RabiParam(
        target=target,
        amplitude=1.0,
        frequency=0.1,
        phase=0.0,
        offset=0.0,
        noise=0.0,
        angle=0.0,
        distance=1.0,
        r2=1.0,
        reference_phase=0.0,
    )


def _make_exp() -> SimpleNamespace:
    targets = {
        "Q28": SimpleNamespace(type=TargetType.CTRL_GE, frequency=6.0),
        "Q28_ge": SimpleNamespace(type=TargetType.CTRL_GE, frequency=6.01),
        "Q28-ef": SimpleNamespace(type=TargetType.CTRL_EF, frequency=5.7),
        "Q28_ef": SimpleNamespace(type=TargetType.CTRL_EF, frequency=5.8),
        "not_ef": SimpleNamespace(type=TargetType.CTRL_GE, frequency=5.8),
    }

    def resolve_qubit_label(label: str) -> str:
        if label in targets:
            return "Q28"
        raise ValueError(f"Unknown target `{label}`.")

    ctx = SimpleNamespace(
        qubit_labels=["Q28"],
        targets=targets,
        resolve_qubit_label=resolve_qubit_label,
        resolve_ge_label=lambda _label: "Q28",
        resolve_ef_label=lambda label: "Q28-ef" if label in {"Q28", "Q28-ef"} else label,
        util=SimpleNamespace(
            resolve_sampling_period=lambda sampling_period: float(sampling_period),
            discretize_time_range=lambda values, sampling_period: np.asarray(
                values, dtype=float
            ),
        ),
        measurement=SimpleNamespace(sampling_period=1.0),
    )
    measurement_service = _MeasurementService()
    params = _Params()
    rabi_params = {
        "Q28": _rabi_param("Q28"),
        "Q28_ge": _rabi_param("Q28_ge"),
        "Q28_Q28-ef": _rabi_param("Q28_Q28-ef"),
        "Q28_ge_Q28_ef": _rabi_param("Q28_ge_Q28_ef"),
    }
    hpi_updates: list[tuple[str, dict[str, float | str]]] = []
    pi_updates: list[tuple[str, dict[str, float | str]]] = []
    stored: list[dict[str, RabiParam]] = []
    return SimpleNamespace(
        targets=targets,
        ctx=ctx,
        util=SimpleNamespace(create_qubit_subgroups=lambda labels: [list(labels), []]),
        pulse=SimpleNamespace(
            x180=lambda _label: Blank(0),
            validate_rabi_params=lambda _labels: None,
        ),
        ef_hpi_pulse={"Q28-ef": Blank(0), "Q28_ef": Blank(0)},
        measurement_service=measurement_service,
        ge_rabi_params={"Q28": rabi_params["Q28"], "Q28_ge": rabi_params["Q28_ge"]},
        get_rabi_param=lambda label: rabi_params.get(label),
        calib_note=SimpleNamespace(
            update_hpi_param=lambda label, value: hpi_updates.append((label, value)),
            update_pi_param=lambda label, value: pi_updates.append((label, value)),
        ),
        params=params,
        get_pulse_for_state=lambda target, state: Blank(0),
        store_rabi_params=lambda rabi_params: stored.append(rabi_params),
        stored_rabi_params=stored,
        hpi_updates=hpi_updates,
        pi_updates=pi_updates,
    )


@pytest.fixture(autouse=True)
def _patch_fit_rabi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gf_calibration.fitting,
        "fit_rabi",
        lambda **_kwargs: _FitResult(),
    )
    monkeypatch.setattr(
        gf_calibration.fitting,
        "fit_ampl_calib_data",
        lambda **_kwargs: _FitResult(),
    )
    monkeypatch.setattr(
        gf_calibration.fitting,
        "fit_ramsey",
        lambda **_kwargs: _FitResult(),
    )


def test_obtain_gf_rabi_params_uses_explicit_custom_ef_target() -> None:
    exp = _make_exp()

    result = obtain_gf_rabi_params(
        exp=exp,  # type: ignore[arg-type]
        targets="Q28",
        time_range=np.asarray([8.0]),
        frequencies={"Q28_ef": 5.81},
        ef_targets={"Q28": "Q28_ef"},
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28" in labels
    assert "Q28_ef" in labels
    assert "Q28-ef" not in labels
    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28_ef": 5.81}
    assert list(result.data) == ["Q28_Q28_ef"]
    assert list(result.rabi_params or {}) == ["Q28_Q28_ef"]
    assert exp.params.ef_amplitude_queries == ["Q28"]


def test_gf_rabi_experiment_uses_explicit_custom_ge_and_ef_targets() -> None:
    exp = _make_exp()

    result = gf_rabi_experiment(
        exp=exp,  # type: ignore[arg-type]
        amplitudes={"Q28": 0.125},
        time_range=np.asarray([8.0]),
        ge_targets={"Q28": "Q28_ge"},
        ef_targets={"Q28": "Q28_ef"},
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28_ge" in labels
    assert "Q28_ef" in labels
    assert "Q28-ef" not in labels
    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28_ef": 5.8}
    assert list(result.data) == ["Q28_ge_Q28_ef"]


def test_gf_rabi_experiment_defaults_to_canonical_ef_target() -> None:
    exp = _make_exp()

    result = gf_rabi_experiment(
        exp=exp,  # type: ignore[arg-type]
        amplitudes={"Q28": 0.125},
        time_range=np.asarray([8.0]),
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28-ef" in labels
    assert "Q28_ef" not in labels
    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28-ef": 5.7}
    assert list(result.data) == ["Q28_Q28-ef"]


def test_gf_rabi_experiment_accepts_qubit_keyed_frequency_for_custom_ef_target() -> None:
    exp = _make_exp()

    gf_rabi_experiment(
        exp=exp,  # type: ignore[arg-type]
        amplitudes={"Q28": 0.125},
        time_range=np.asarray([8.0]),
        frequencies={"Q28": 5.82},
        ef_targets={"Q28": "Q28_ef"},
        plot=False,
    )

    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28_ef": 5.82}


def test_gf_rabi_experiment_accepts_direct_custom_ef_target() -> None:
    exp = _make_exp()

    result = gf_rabi_experiment(
        exp=exp,  # type: ignore[arg-type]
        amplitudes={"Q28_ef": 0.125},
        time_range=np.asarray([8.0]),
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28" in labels
    assert "Q28_ef" in labels
    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28_ef": 5.8}
    assert list(result.data) == ["Q28_Q28_ef"]


def test_gf_rabi_experiment_rejects_unregistered_explicit_ef_target() -> None:
    exp = _make_exp()

    with pytest.raises(ValueError, match="EF target `missing` is not registered"):
        gf_rabi_experiment(
            exp=exp,  # type: ignore[arg-type]
            amplitudes={"Q28": 0.125},
            time_range=np.asarray([8.0]),
            ef_targets={"Q28": "missing"},
            plot=False,
        )


def test_gf_rabi_experiment_rejects_non_ef_explicit_target() -> None:
    exp = _make_exp()

    with pytest.raises(ValueError, match="Target `not_ef` is not an EF target"):
        gf_rabi_experiment(
            exp=exp,  # type: ignore[arg-type]
            amplitudes={"Q28": 0.125},
            time_range=np.asarray([8.0]),
            ef_targets={"Q28": "not_ef"},
            plot=False,
        )


def test_calibrate_gf_pulse_uses_explicit_custom_ge_and_ef_targets() -> None:
    exp = _make_exp()

    result = calibrate_gf_pulse(
        exp=exp,  # type: ignore[arg-type]
        targets="Q28",
        pulse_type="hpi",
        n_points=3,
        ge_targets={"Q28": "Q28_ge"},
        ef_targets={"Q28": "Q28_ef"},
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28_ge" in labels
    assert "Q28_ef" in labels
    assert "Q28-ef" not in labels
    assert list(result.data) == ["Q28"]
    assert exp.hpi_updates[0][0] == "Q28_ef"


def test_gf_ramsey_experiment_uses_explicit_custom_ge_and_ef_targets() -> None:
    exp = _make_exp()

    result = gf_ramsey_experiment(
        exp=exp,  # type: ignore[arg-type]
        targets="Q28",
        time_range=np.asarray([0.0]),
        ge_targets={"Q28": "Q28_ge"},
        ef_targets={"Q28": "Q28_ef"},
        plot=False,
    )

    labels = exp.measurement_service.calls[0]["labels"]
    assert "Q28_ge" in labels
    assert "Q28_ef" in labels
    assert "Q28-ef" not in labels
    assert exp.measurement_service.calls[0]["frequencies"] == {"Q28_ef": 5.801}
    assert list(result.data) == ["Q28"]

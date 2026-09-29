"""Tests for simultaneous randomized benchmarking measurement flow."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

from numpy.testing import assert_allclose

from qubex.pulse import Blank


class _FitResult(dict):
    def get_figure(self) -> object:
        """Return the stored test figure."""
        return self["fig"]


class _MeasurementService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def measure(self, **kwargs: Any) -> SimpleNamespace:
        """Record measurement calls and return deterministic target data."""
        self.calls.append(kwargs)
        call_index = len(self.calls)
        return SimpleNamespace(
            data={
                "Q00": SimpleNamespace(kerneled=0.80 - 0.01 * call_index),
                "Q01": SimpleNamespace(kerneled=0.70 - 0.02 * call_index),
            }
        )


class _PulseService:
    def __init__(self) -> None:
        self.rabi_params = {
            "Q00": SimpleNamespace(normalize=lambda iq: iq),
            "Q01": SimpleNamespace(normalize=lambda iq: iq),
        }

    def x90(self, _target: str) -> Blank:
        """Return a physical X90 placeholder pulse."""
        return Blank(20)

    def y90(self, _target: str) -> Blank:
        """Return a physical Y90 placeholder pulse."""
        return Blank(20)


def _experiment_stub() -> SimpleNamespace:
    """Build a minimal Experiment-like object for SRB tests."""
    return SimpleNamespace(
        ctx=SimpleNamespace(
            experiment_system=SimpleNamespace(
                get_target=lambda _target: SimpleNamespace(is_cr=False)
            )
        ),
        pulse=_PulseService(),
        measurement_service=_MeasurementService(),
    )


def test_simultaneous_randomized_benchmarking_measures_two_targets_together(
    monkeypatch: Any,
) -> None:
    """Given two 1Q targets, then each SRB point uses one schedule."""
    srb_module = importlib.import_module(
        "qubex.contrib.experiment.simultaneous_randomized_benchmarking"
    )
    fit_calls: list[dict[str, Any]] = []

    def _fake_fit_rb(**kwargs: Any) -> _FitResult:
        fit_calls.append(kwargs)
        return _FitResult(fig=object(), p=0.99, p_err=0.01)

    monkeypatch.setattr(srb_module.fitting, "fit_rb", _fake_fit_rb)
    monkeypatch.setattr(srb_module.viz, "save_figure", lambda **_kwargs: None)

    exp = _experiment_stub()
    result = srb_module.simultaneous_randomized_benchmarking(
        cast(Any, exp),
        ["Q00", "Q01"],
        n_cliffords_range=[0, 2],
        n_trials=2,
        seeds=[10, 11],
        n_shots=128,
        shot_interval=2048,
        plot=False,
        save_image=False,
    )

    assert len(exp.measurement_service.calls) == 4
    for call in exp.measurement_service.calls:
        assert call["sequence"].labels == ["Q00", "Q01"]
        assert call["sequence"].is_valid()
        assert call["mode"] == "avg"
        assert call["n_shots"] == 128
        assert call["shot_interval"] == 2048
        assert call["time_integration"] is True
        assert call["reset_awg_and_capunits"] is True
        assert call["plot"] is False

    assert set(result.data) == {"Q00", "Q01"}
    assert_allclose(result.data["Q00"]["n_cliffords"], [0, 2])
    assert_allclose(result.data["Q00"]["seeds"], [10, 11])
    assert result.data["Q00"]["trials"].shape == (2, 2)
    assert result.data["Q01"]["trials"].shape == (2, 2)
    assert len(fit_calls) == 2
    assert fit_calls[0]["title"] == "Simultaneous randomized benchmarking"
    assert fit_calls[0]["bounds"] == ((0, 0, 0), (0.5, 1, 1))

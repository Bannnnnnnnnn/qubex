"""Tests for the contributed JPA calibration workflow."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from qubex.contrib.experiment import jpa_calibration as jpa
from qubex.contrib.experiment.jpa_calibration import (
    _acquire_state_samples,
    _JPAScan,
    _measure_jpa_point,
    _MuxOptimum,
)
from qubex.experiment import Experiment
from qubex.pulse import Blank
from qubex.system import Mux


class _FakeControlParameters:
    """Provide one MUX-scoped JPA configuration."""

    def __init__(self) -> None:
        self.jpa_params = {
            0: {
                "dc_voltage": 0.25,
                "pump_frequency": 10.0,
                "pump_amplitude": 0.1,
            }
        }

    def get_dc_voltage(self, mux: int) -> float:
        """Return the configured DC voltage."""
        return float(self.jpa_params[mux]["dc_voltage"])

    def get_pump_frequency(self, mux: int) -> float:
        """Return the configured pump frequency."""
        return float(self.jpa_params[mux]["pump_frequency"])

    def get_pump_amplitude(self, mux: int) -> float:
        """Return the configured pump amplitude."""
        return float(self.jpa_params[mux]["pump_amplitude"])


class _FakeContext:
    """Resolve one anchor qubit and record temporary frequencies."""

    def __init__(self, experiment: _FakeExperiment) -> None:
        self._experiment = experiment
        self.qubit_labels = ["Q00", "Q02"]
        self.frequency_contexts: list[float] = []
        resonators = (
            SimpleNamespace(qubit="Q00", is_valid=True),
            SimpleNamespace(qubit="Q01", is_valid=True),
            SimpleNamespace(qubit="Q02", is_valid=True),
        )
        self.mux = SimpleNamespace(index=0, label="M000", resonators=resonators)
        self.experiment_system = SimpleNamespace(
            control_params=_FakeControlParameters(),
            get_mux_by_qubit=lambda _qubit: self.mux,
            get_boxes_for_qubits=lambda _qubits: [
                SimpleNamespace(id="BOX_CTRL"),
                SimpleNamespace(id="BOX_READOUT"),
            ],
            get_target=lambda _label: SimpleNamespace(
                channel=SimpleNamespace(
                    port=SimpleNamespace(box_id="BOX_PUMP"),
                )
            ),
        )
        self.reset_calls: list[dict[str, set[str] | None]] = []

    def resolve_qubit_label(self, label: str) -> str:
        """Resolve qubit and readout aliases."""
        return {"RQ00": "Q00"}.get(label, label)

    def resolve_read_label(self, label: str) -> str:
        """Resolve one qubit to its readout label."""
        return f"R{self.resolve_qubit_label(label)}"

    def reset_awg_and_capunits(
        self,
        *,
        box_ids: set[str] | None = None,
        qubits: set[str] | None = None,
    ) -> None:
        """Record one reset request."""
        self.reset_calls.append(
            {
                "box_ids": None if box_ids is None else set(box_ids),
                "qubits": None if qubits is None else set(qubits),
            }
        )

    @contextmanager
    def modified_frequencies(self, frequencies: dict[str, float]) -> Iterator[None]:
        """Record and restore a temporary pump frequency."""
        frequency = float(frequencies[self.mux.label])
        previous = self._experiment.current_frequency
        self._experiment.current_frequency = frequency
        self.frequency_contexts.append(frequency)
        try:
            yield
        finally:
            self._experiment.current_frequency = previous


class _FakeExperiment:
    """Provide the topology needed by the public calibration workflow."""

    def __init__(self) -> None:
        self.current_frequency = 10.0
        self.current_dc_voltage = 0.25
        self.dc_output_enabled = False
        self.ctx = _FakeContext(self)


class _FakeSupply:
    """Record bounded DC voltage updates."""

    def __init__(self, experiment: _FakeExperiment) -> None:
        self._experiment = experiment
        self.on_calls: list[int] = []

    def set_voltage(self, *, channel: int, voltage: float) -> None:
        """Set the fake output voltage."""
        assert channel == 1
        self._experiment.current_dc_voltage = float(voltage)

    def get_voltage(self, *, channel: int) -> float:
        """Return the fake output voltage."""
        assert channel == 1
        return self._experiment.current_dc_voltage

    def on(self, *, channel: int) -> None:
        """Record that the fake channel is enabled."""
        assert channel == 1
        self.on_calls.append(channel)
        self._experiment.dc_output_enabled = True

    def off(self, *, channel: int) -> None:
        """Record that the fake channel is disabled."""
        assert channel == 1
        self._experiment.dc_output_enabled = False

    def get_output_state(self, *, channel: int) -> int:
        """Return the fake enabled state."""
        assert channel == 1
        return int(self._experiment.dc_output_enabled)


def test_calibrate_jpa_resolves_mux_peers_and_returns_best_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given one qubit, calibration should optimize all active peers on its MUX."""
    experiment = _FakeExperiment()
    context_exited = False
    measurement_calls: list[dict[str, float | int | None]] = []

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield one fake supply for the complete sweep."""
        nonlocal context_exited
        assert voltages == {1: 0.0}
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            context_exited = True

    def fake_measure_point(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        *,
        mux: Any,
        pump_amplitude: float,
        n_shots: int,
        shot_interval: float | None,
        readout_amplitude: float | None,
        readout_duration: float | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return deterministic scores with one known optimum."""
        del mux
        assert n_shots == 4
        measurement_calls.append(
            {
                "dc_voltage": experiment.current_dc_voltage,
                "pump_frequency": experiment.current_frequency,
                "pump_amplitude": pump_amplitude,
                "n_shots": n_shots,
                "shot_interval": shot_interval,
                "readout_amplitude": readout_amplitude,
                "readout_duration": readout_duration,
            }
        )
        distance = (
            abs(experiment.current_dc_voltage - 0.5)
            + abs(experiment.current_frequency - 10.1)
            + abs(pump_amplitude - 0.2)
        )
        score = 10.0 - distance
        return (
            dict.fromkeys(evaluated_qubits, score),
            dict.fromkeys(evaluated_qubits, 1.0),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fake_measure_point)

    result = jpa.calibrate_jpa(
        cast(Experiment, experiment),
        "RQ00",
        dc_voltage_range=[0.0, 0.5],
        pump_frequency_range=[10.0, 10.1],
        pump_amplitude_range=[0.1, 0.2],
        n_shots=4,
        shot_interval=1000.0,
        readout_amplitude=0.13,
        readout_duration=640.0,
        fine_points=None,
        dc_settle_time=0.0,
        reset_awg_and_capunits=False,
        enable_tqdm=False,
    )

    assert result["anchor_qubit"] == "Q00"
    assert result["evaluated_qubits"] == ("Q00", "Q02")
    assert result["mux_label"] == "M000"
    assert result["optimal_parameters"] == {
        "dc_voltage": 0.5,
        "pump_frequency": 10.1,
        "pump_amplitude": 0.2,
    }
    assert result["score"] == pytest.approx(10.0)
    assert result["score_gain"] == pytest.approx(10.0 / 9.2)
    assert result["scores_by_qubit"] == {"Q00": 10.0, "Q02": 10.0}
    assert result["score_gains_by_qubit"] == pytest.approx(
        {"Q00": 10.0 / 9.2, "Q02": 10.0 / 9.2}
    )
    assert result["flatness_by_qubit"] == {"Q00": 1.0, "Q02": 1.0}
    assert result["flatness_ratios_by_qubit"] == {"Q00": 1.0, "Q02": 1.0}
    baseline = cast(dict[str, Any], result["baseline"])
    assert baseline["parameters"] == {
        "dc_voltage": 0.0,
        "pump_amplitude": 0.0,
    }
    assert baseline["score"] == pytest.approx(9.2)
    assert baseline["scores_by_qubit"] == pytest.approx({"Q00": 9.2, "Q02": 9.2})
    assert baseline["flatness_by_qubit"] == {"Q00": 1.0, "Q02": 1.0}
    assert result["success"] is True
    assert result["status"] == "success"
    assert result["fine_scan"] is None
    assert len(measurement_calls) == 9
    baseline_calls = [
        call
        for call in measurement_calls
        if call["dc_voltage"] == 0.0 and call["pump_amplitude"] == 0.0
    ]
    assert baseline_calls == [
        {
            "dc_voltage": 0.0,
            "pump_frequency": 10.0,
            "pump_amplitude": 0.0,
            "n_shots": 4,
            "shot_interval": 1000.0,
            "readout_amplitude": 0.13,
            "readout_duration": 640.0,
        }
    ]
    assert all(
        call["n_shots"] == 4
        and call["shot_interval"] == 1000.0
        and call["readout_amplitude"] == 0.13
        and call["readout_duration"] == 640.0
        for call in measurement_calls
    )
    assert context_exited
    assert experiment.current_frequency == pytest.approx(10.0)
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_calibrate_jpa_uses_config_derived_defaults_with_only_qubit_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The qubit-only call should run coarse and fine scans from configured defaults."""
    experiment = _FakeExperiment()
    control_params = experiment.ctx.experiment_system.control_params
    control_params.jpa_params[0].update(
        {
            "dc_voltage": 1.0,
            "pump_frequency": 10.0,
            "pump_amplitude": 0.0,
        }
    )
    dc_context_entries: list[dict[int, float]] = []
    scan_calls: list[dict[str, Any]] = []
    baseline_calls: list[tuple[float, float]] = []

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Apply the requested initial voltage and restore a disabled output."""
        dc_context_entries.append(dict(voltages))
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)

    def fake_scan(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        *,
        dc_voltages: np.ndarray,
        pump_frequencies: np.ndarray,
        pump_amplitudes: np.ndarray,
        description: str,
        **_kwargs: Any,
    ) -> _JPAScan:
        """Record one default scan and select its center point."""
        axes = (
            np.asarray(dc_voltages, dtype=np.float64),
            np.asarray(pump_frequencies, dtype=np.float64),
            np.asarray(pump_amplitudes, dtype=np.float64),
        )
        scan_calls.append(
            {
                "description": description,
                "dc_voltages": axes[0].copy(),
                "pump_frequencies": axes[1].copy(),
                "pump_amplitudes": axes[2].copy(),
            }
        )
        index = tuple(len(axis) // 2 for axis in axes)
        grid_shape = tuple(len(axis) for axis in axes)
        score_map = np.ones(grid_shape, dtype=np.float64)
        score_gain_map = np.full(grid_shape, 1.1, dtype=np.float64)
        peer_grid_shape = (len(evaluated_qubits), *grid_shape)
        return _JPAScan(
            dc_voltages=axes[0],
            pump_frequencies=axes[1],
            pump_amplitudes=axes[2],
            scores=np.ones(
                (len(evaluated_qubits), *grid_shape),
                dtype=np.float64,
            ),
            flatness=np.ones(
                (len(evaluated_qubits), *grid_shape),
                dtype=np.float64,
            ),
            optimum=_MuxOptimum(
                index=index,
                aggregate_score=1.0,
                aggregate_score_gain=1.1,
                peer_scores=np.ones(len(evaluated_qubits), dtype=np.float64),
                peer_score_gains=np.full(
                    len(evaluated_qubits),
                    1.1,
                    dtype=np.float64,
                ),
                peer_flatness=np.ones(len(evaluated_qubits), dtype=np.float64),
                peer_flatness_ratios=np.ones(
                    len(evaluated_qubits),
                    dtype=np.float64,
                ),
                aggregate_score_map=score_map,
                aggregate_score_gain_map=score_gain_map,
                score_gains=np.full(peer_grid_shape, 1.1, dtype=np.float64),
                flatness_ratios=np.ones(peer_grid_shape, dtype=np.float64),
                measurement_valid_mask=np.ones(grid_shape, dtype=np.bool_),
                gain_valid_mask=np.ones(grid_shape, dtype=np.bool_),
                flatness_valid_mask=np.ones(grid_shape, dtype=np.bool_),
                valid_mask=np.ones(grid_shape, dtype=np.bool_),
                on_boundary=False,
            ),
        )

    def fake_measure_point(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        *,
        pump_amplitude: float,
        **_kwargs: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return the one JPA-off baseline used by both fake scans."""
        baseline_calls.append((experiment.current_dc_voltage, pump_amplitude))
        return (
            dict.fromkeys(evaluated_qubits, 1.0),
            dict.fromkeys(evaluated_qubits, 1.0),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_scan_jpa_grid", fake_scan)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fake_measure_point)

    result = jpa.calibrate_jpa(cast(Experiment, experiment), "Q00")

    assert dc_context_entries == [{1: 0.0}]
    assert baseline_calls == [(0.0, 0.0)]
    assert len(scan_calls) == 2
    coarse_scan, fine_scan = scan_calls
    assert "coarse" in coarse_scan["description"]
    assert "fine" in fine_scan["description"]
    for key in ("dc_voltages", "pump_frequencies", "pump_amplitudes"):
        assert len(coarse_scan[key]) == 9
        assert len(fine_scan[key]) == 9
        coarse_center = coarse_scan[key][len(coarse_scan[key]) // 2]
        assert np.any(np.isclose(fine_scan[key], coarse_center))
    np.testing.assert_allclose(
        coarse_scan["dc_voltages"][[0, -1]],
        [0.0, 2.0],
    )
    assert np.any(np.isclose(coarse_scan["dc_voltages"], 1.0))
    np.testing.assert_allclose(
        coarse_scan["pump_frequencies"][[0, -1]],
        [10.0, 10.5],
    )
    np.testing.assert_allclose(
        coarse_scan["pump_amplitudes"][[0, -1]],
        [0.0, 0.5],
    )
    assert result["fine_scan"] is not None
    assert experiment.ctx.reset_calls == [
        {
            "box_ids": {"BOX_CTRL", "BOX_READOUT", "BOX_PUMP"},
            "qubits": None,
        }
    ]
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"n_shots": 2}, "n_shots"),
        ({"fine_points": 2}, "fine_points"),
        ({"minimum_score_gain": 0.99}, "minimum_score_gain"),
        ({"maximum_flatness_ratio": 0.99}, "maximum_flatness_ratio"),
        ({"flatness_threshold": 0.99}, "flatness_threshold"),
        ({"baseline_dc_voltage": 4.1}, "baseline_dc_voltage"),
        ({"dc_voltage_range": [4.1]}, "dc_voltage_range"),
    ],
)
def test_calibrate_jpa_rejects_invalid_inputs_before_dc_connection(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, Any],
    match: str,
) -> None:
    """Invalid public options should fail before opening the DC supply."""
    experiment = _FakeExperiment()
    dc_connection_attempted = False

    def fail_dc_voltage(_voltages: dict[int, float]) -> None:
        """Record an unexpected attempt to open the DC supply."""
        nonlocal dc_connection_attempted
        dc_connection_attempted = True
        pytest.fail("DC supply should not be opened for invalid inputs.")

    monkeypatch.setattr(jpa, "dc_voltage", fail_dc_voltage)

    with pytest.raises(ValueError, match=match):
        jpa.calibrate_jpa(
            cast(Experiment, experiment),
            "Q00",
            reset_awg_and_capunits=False,
            enable_tqdm=False,
            **options,
        )

    assert not dc_connection_attempted


def test_calibrate_jpa_retains_coarse_constraint_diagnostics_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected coarse scan should retain raw and relative maps after cleanup."""
    experiment = _FakeExperiment()
    context_exited = False
    measurement_calls: list[tuple[float, float]] = []

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield a fake supply and restore its original disabled state."""
        nonlocal context_exited
        assert voltages == {1: 0.0}
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            context_exited = True

    def fake_measure_point(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        *,
        pump_amplitude: float,
        **_kwargs: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return a usable baseline followed by a flat but weaker scan point."""
        measurement_calls.append((experiment.current_dc_voltage, pump_amplitude))
        if experiment.current_dc_voltage == 0.0 and pump_amplitude == 0.0:
            return (
                dict(zip(evaluated_qubits, (10.0, 20.0), strict=True)),
                dict(zip(evaluated_qubits, (1.4, 1.2), strict=True)),
            )
        return (
            dict(zip(evaluated_qubits, (9.0, 19.0), strict=True)),
            dict(zip(evaluated_qubits, (1.45, 1.25), strict=True)),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fake_measure_point)

    with pytest.raises(jpa.JPAConstraintError, match="no point improves") as exc_info:
        jpa.calibrate_jpa(
            cast(Experiment, experiment),
            "Q00",
            dc_voltage_range=[0.5],
            pump_frequency_range=[10.1],
            pump_amplitude_range=[0.2],
            n_shots=4,
            fine_points=None,
            dc_settle_time=0.0,
            reset_awg_and_capunits=False,
            enable_tqdm=False,
        )

    error = exc_info.value
    diagnostics = error.diagnostics
    assert error.stage == "coarse"
    assert diagnostics["success"] is False
    assert diagnostics["status"] == "no_improvement"
    assert diagnostics["failure_stage"] == "coarse"
    assert diagnostics["optimal_parameters"] is None
    assert diagnostics["score"] is None
    assert diagnostics["score_gain"] is None
    assert diagnostics["fine_scan"] is None
    assert diagnostics["baseline"] == {
        "parameters": {"dc_voltage": 0.0, "pump_amplitude": 0.0},
        "score": 10.0,
        "scores_by_qubit": {"Q00": 10.0, "Q02": 20.0},
        "flatness_by_qubit": {"Q00": 1.4, "Q02": 1.2},
    }

    coarse = cast(dict[str, Any], diagnostics["coarse_scan"])
    np.testing.assert_array_equal(coarse["dc_voltages"], [0.5])
    np.testing.assert_array_equal(coarse["pump_frequencies"], [10.1])
    np.testing.assert_array_equal(coarse["pump_amplitudes"], [0.2])
    np.testing.assert_allclose(coarse["scores"]["Q00"], [[[9.0]]])
    np.testing.assert_allclose(coarse["scores"]["Q02"], [[[19.0]]])
    np.testing.assert_allclose(coarse["score_gains"]["Q00"], [[[0.9]]])
    np.testing.assert_allclose(coarse["score_gains"]["Q02"], [[[0.95]]])
    np.testing.assert_allclose(
        coarse["flatness_ratios"]["Q00"],
        [[[1.45 / 1.4]]],
    )
    np.testing.assert_allclose(
        coarse["flatness_ratios"]["Q02"],
        [[[1.25 / 1.2]]],
    )
    np.testing.assert_array_equal(coarse["measurement_valid_mask"], [[[True]]])
    np.testing.assert_array_equal(coarse["gain_valid_mask"], [[[False]]])
    np.testing.assert_array_equal(coarse["flatness_valid_mask"], [[[True]]])
    np.testing.assert_array_equal(coarse["valid_mask"], [[[False]]])
    assert coarse["optimal_index"] is None
    assert coarse["on_boundary"] is None
    assert measurement_calls == [(0.0, 0.0), (0.5, 0.2)]
    assert context_exited
    assert experiment.current_frequency == pytest.approx(10.0)
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_calibrate_jpa_retains_valid_coarse_scan_when_fine_scan_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected fine scan should expose both it and the valid coarse candidate."""
    experiment = _FakeExperiment()
    context_exited = False
    measurement_count = 0

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield a fake supply and record cleanup before the public error."""
        nonlocal context_exited
        assert voltages == {1: 0.0}
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            context_exited = True

    def fake_measure_point(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        **_kwargs: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Make coarse points improve and the subsequent fine points regress."""
        nonlocal measurement_count
        measurement_count += 1
        if measurement_count == 1:
            scores = (10.0, 20.0)
        elif measurement_count <= 3:
            scores = (12.0, 24.0)
        else:
            scores = (9.0, 18.0)
        return (
            dict(zip(evaluated_qubits, scores, strict=True)),
            dict.fromkeys(evaluated_qubits, 1.2),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fake_measure_point)

    with pytest.raises(jpa.JPAConstraintError, match="fine stage") as exc_info:
        jpa.calibrate_jpa(
            cast(Experiment, experiment),
            "Q00",
            dc_voltage_range=[0.25, 0.5],
            pump_frequency_range=[10.1],
            pump_amplitude_range=[0.2],
            n_shots=4,
            fine_points=3,
            dc_settle_time=0.0,
            reset_awg_and_capunits=False,
            enable_tqdm=False,
        )

    error = exc_info.value
    diagnostics = error.diagnostics
    assert error.stage == "fine"
    assert diagnostics["status"] == "no_improvement"
    assert diagnostics["failure_stage"] == "fine"
    assert diagnostics["optimal_parameters"] is None

    coarse = cast(dict[str, Any], diagnostics["coarse_scan"])
    fine = cast(dict[str, Any], diagnostics["fine_scan"])
    assert coarse["optimal_index"] == (0, 0, 0)
    assert coarse["on_boundary"] is True
    np.testing.assert_allclose(
        coarse["aggregate_score_gain"],
        [[[1.2]], [[1.2]]],
    )
    np.testing.assert_array_equal(coarse["valid_mask"], [[[True]], [[True]]])
    np.testing.assert_allclose(fine["dc_voltages"], [0.25, 0.375, 0.5])
    np.testing.assert_allclose(
        fine["aggregate_score_gain"], [[[0.9]], [[0.9]], [[0.9]]]
    )
    np.testing.assert_array_equal(fine["valid_mask"], [[[False]], [[False]], [[False]]])
    assert fine["optimal_index"] is None
    assert fine["on_boundary"] is None
    assert measurement_count == 6
    assert context_exited
    assert experiment.current_frequency == pytest.approx(10.0)
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_calibrate_jpa_refines_coarse_candidate_into_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sub-threshold coarse candidate should seed a successful fine scan."""
    experiment = _FakeExperiment()
    context_exited = False
    measurement_count = 0

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield a fake supply and record complete restoration."""
        nonlocal context_exited
        assert voltages == {1: 0.0}
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            context_exited = True

    def fake_measure_point(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        **_kwargs: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return a weak coarse grid and one improved fine-grid center."""
        nonlocal measurement_count
        measurement_count += 1
        scores_by_call = {
            1: (10.0, 20.0),
            2: (10.2, 20.4),
            3: (10.4, 20.8),
            4: (10.3, 20.6),
            5: (11.0, 22.0),
            6: (10.4, 20.8),
        }
        scores = scores_by_call[measurement_count]
        return (
            dict(zip(evaluated_qubits, scores, strict=True)),
            dict.fromkeys(evaluated_qubits, 1.2),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fake_measure_point)

    result = jpa.calibrate_jpa(
        cast(Experiment, experiment),
        "Q00",
        dc_voltage_range=[0.25, 0.5],
        pump_frequency_range=[10.1],
        pump_amplitude_range=[0.2],
        n_shots=4,
        fine_points=3,
        dc_settle_time=0.0,
        reset_awg_and_capunits=False,
        enable_tqdm=False,
    )

    assert result["success"] is True
    assert result["failure_stage"] is None
    assert result["optimal_parameters"] == {
        "dc_voltage": 0.375,
        "pump_frequency": 10.1,
        "pump_amplitude": 0.2,
    }
    assert result["score"] == pytest.approx(11.0)
    assert result["score_gain"] == pytest.approx(1.1)

    coarse = cast(dict[str, Any], result["coarse_scan"])
    fine = cast(dict[str, Any], result["fine_scan"])
    assert coarse["optimal_index"] is None
    assert coarse["candidate_index"] == (1, 0, 0)
    assert coarse["candidate_on_boundary"] is True
    np.testing.assert_array_equal(coarse["valid_mask"], [[[False]], [[False]]])
    np.testing.assert_allclose(
        coarse["aggregate_score_gain"],
        [[[1.02]], [[1.04]]],
    )
    np.testing.assert_allclose(fine["dc_voltages"], [0.25, 0.375, 0.5])
    assert fine["optimal_index"] == (1, 0, 0)
    np.testing.assert_array_equal(fine["valid_mask"], [[[False]], [[True]], [[False]]])
    assert measurement_count == 6
    assert context_exited
    assert experiment.current_frequency == pytest.approx(10.0)
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_calibrate_jpa_reports_invalid_baseline_after_hardware_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable OFF reference should be diagnostic and still clean up DC."""
    experiment = _FakeExperiment()
    context_exited = False

    @contextmanager
    def fake_dc_voltage(voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield a fake supply and record cleanup before the public error."""
        nonlocal context_exited
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            context_exited = True

    def invalid_baseline(
        _exp: Experiment,
        evaluated_qubits: tuple[str, ...],
        **_kwargs: Any,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return a zero separation that cannot define a gain denominator."""
        return (
            dict.fromkeys(evaluated_qubits, 0.0),
            dict.fromkeys(evaluated_qubits, 1.0),
        )

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", invalid_baseline)

    with pytest.raises(jpa.JPAConstraintError, match="baseline stage") as exc_info:
        jpa.calibrate_jpa(
            cast(Experiment, experiment),
            "Q00",
            dc_voltage_range=[0.25],
            pump_frequency_range=[10.0],
            pump_amplitude_range=[0.1],
            n_shots=4,
            fine_points=None,
            dc_settle_time=0.0,
            reset_awg_and_capunits=False,
            enable_tqdm=False,
        )

    error = exc_info.value
    assert error.stage == "baseline"
    assert error.diagnostics["status"] == "invalid_baseline"
    assert error.diagnostics["failure_stage"] == "baseline"
    assert error.diagnostics["coarse_scan"] is None
    assert error.diagnostics["fine_scan"] is None
    assert context_exited
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_calibrate_jpa_restores_hardware_context_when_measurement_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measurement failure should still exit DC and frequency contexts."""
    experiment = _FakeExperiment()
    dc_context_exited = False

    @contextmanager
    def fake_dc_voltage(_voltages: dict[int, float]) -> Iterator[_FakeSupply]:
        """Yield a fake supply and record cleanup."""
        nonlocal dc_context_exited
        original_voltage = experiment.current_dc_voltage
        supply = _FakeSupply(experiment)
        supply.set_voltage(channel=1, voltage=_voltages[1])
        supply.on(channel=1)
        try:
            yield supply
        finally:
            supply.set_voltage(channel=1, voltage=original_voltage)
            supply.off(channel=1)
            dc_context_exited = True

    def fail_measurement(*_args: Any, **_kwargs: Any) -> Any:
        """Raise a deterministic acquisition failure."""
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(jpa, "dc_voltage", fake_dc_voltage)
    monkeypatch.setattr(jpa, "_measure_jpa_point", fail_measurement)

    with pytest.raises(RuntimeError, match="acquisition failed"):
        jpa.calibrate_jpa(
            cast(Experiment, experiment),
            "Q00",
            dc_voltage_range=[0.25],
            pump_frequency_range=[10.0],
            pump_amplitude_range=[0.1],
            n_shots=4,
            fine_points=None,
            dc_settle_time=0.0,
            reset_awg_and_capunits=False,
            enable_tqdm=False,
        )

    assert dc_context_exited
    assert experiment.current_frequency == pytest.approx(10.0)
    assert experiment.current_dc_voltage == pytest.approx(0.25)
    assert not experiment.dc_output_enabled


def test_measure_jpa_point_marks_constant_state_cloud_as_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constant ground or excited IQ cloud should produce invalid flatness."""
    acquisition_count = 0

    def fake_acquire(*_args: Any, **_kwargs: Any) -> dict[str, np.ndarray]:
        """Return one finite cloud followed by one degenerate cloud."""
        nonlocal acquisition_count
        acquisition_count += 1
        if acquisition_count == 1:
            samples = np.array([1.0, -1.0, 1.0j, -1.0j])
        else:
            samples = np.ones(4, dtype=np.complex128)
        return {"Q00": samples}

    monkeypatch.setattr(jpa, "_acquire_state_samples", fake_acquire)

    _, flatness = _measure_jpa_point(
        cast(Experiment, object()),
        ("Q00",),
        mux=cast(Mux, SimpleNamespace(index=0, label="M000")),
        pump_amplitude=0.2,
        n_shots=4,
        shot_interval=None,
        readout_amplitude=None,
        readout_duration=None,
    )

    assert np.isnan(flatness["Q00"])


def test_acquire_state_samples_uses_explicit_readout_and_pump_schedule() -> None:
    """Single-state acquisition should use modern execution flags and one pump pulse."""
    calls: dict[str, Any] = {}
    mux = SimpleNamespace(index=0, label="M000")

    class FakePulse:
        """Build short fake control and readout waveforms."""

        def x180(self, qubit: str) -> Blank:
            """Record and return one pi pulse."""
            calls.setdefault("x180", []).append(qubit)
            return Blank(duration=16)

        def readout(
            self,
            qubit: str,
            *,
            amplitude: float | None,
            duration: float | None,
        ) -> Blank:
            """Record and return one readout pulse."""
            calls.setdefault("readout", []).append((qubit, amplitude, duration))
            return Blank(duration=32)

    class FakePulseFactory:
        """Build one fake pump waveform."""

        def pump_pulse(self, **kwargs: Any) -> Blank:
            """Record and return one pump pulse."""
            calls["pump"] = kwargs
            return Blank(duration=float(kwargs["duration"]))

    class FakeMeasurement:
        """Return deterministic per-peer IQ data."""

        pulse_factory = FakePulseFactory()

        def execute(self, schedule: Any, **kwargs: Any) -> Any:
            """Record execution and return one capture per peer."""
            assert schedule.is_valid()
            calls["schedule_labels"] = tuple(schedule.labels)
            calls["execute"] = kwargs
            return SimpleNamespace(
                data={
                    qubit: [
                        SimpleNamespace(
                            kerneled=np.array([0.0, 1.0], dtype=np.complex128)
                        )
                    ]
                    for qubit in ("Q00", "Q02")
                }
            )

    fake_exp = SimpleNamespace(
        ctx=SimpleNamespace(resolve_read_label=lambda qubit: f"R{qubit}"),
        pulse=FakePulse(),
        measurement=FakeMeasurement(),
    )

    samples = _acquire_state_samples(
        cast(Experiment, fake_exp),
        ("Q00", "Q02"),
        mux=cast(Mux, mux),
        pump_amplitude=0.2,
        excited=True,
        n_shots=2,
        shot_interval=1000.0,
        readout_amplitude=0.1,
        readout_duration=32.0,
    )

    assert calls["x180"] == ["Q00", "Q02"]
    assert calls["pump"]["mux_index"] == 0
    assert calls["pump"]["amplitude"] == pytest.approx(0.2)
    assert calls["pump"]["duration"] == pytest.approx(32.0)
    assert set(calls["schedule_labels"]) == {"Q00", "Q02", "RQ00", "RQ02", "M000"}
    assert calls["execute"] == {
        "n_shots": 2,
        "shot_interval": 1000.0,
        "shot_averaging": False,
        "time_integration": True,
        "state_classification": False,
        "readout_amplification": False,
        "final_measurement": False,
        "plot": False,
    }
    np.testing.assert_array_equal(samples["Q00"], [0.0, 1.0])

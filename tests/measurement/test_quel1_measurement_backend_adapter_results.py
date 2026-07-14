"""Tests for QuEL-1 adapter result conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numpy.testing import assert_allclose
from qxdriver_quel1.e7awg import CaptureParamTools

from qubex.backend.quel1 import Quel1BackendExecutionResult
from qubex.measurement.adapters.backend_adapter import Quel1MeasurementBackendAdapter
from qubex.measurement.measurement_constraint_profile import (
    MeasurementConstraintProfile,
)
from qubex.measurement.models.measurement_config import MeasurementConfig
from qubex.measurement.models.quel1_measurement_options import Quel1MeasurementOptions
from qubex.typing import MeasurementMode


@dataclass
class _Target:
    sideband: str


@dataclass
class _ExperimentSystemStub:
    sideband_by_target: dict[str, str]

    def get_target(self, target: str) -> _Target:
        return _Target(sideband=self.sideband_by_target[target])


def _make_config(
    *,
    mode: MeasurementMode,
    shots: int,
    time_integration: bool = False,
    state_classification: bool = False,
) -> MeasurementConfig:
    return MeasurementConfig(
        n_shots=shots,
        shot_interval=100.0,
        shot_averaging=(mode == "avg"),
        time_integration=time_integration,
        state_classification=state_classification,
    )


def test_build_measurement_result_converts_single_mode_to_qubit_labels() -> None:
    """Given QuEL-1 waveform shots, conversion should expose canonical waveform-series data."""
    norm_factor = 2 ** (-16)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array(
                    [[9.0 + 3.0j], [10.0 + 4.0j], [11.0 + 5.0j], [12.0 + 6.0j]],
                    dtype=np.complex128,
                ),
                np.array(
                    [[4.0 + 2.0j], [5.0 + 3.0j], [6.0 + 4.0j], [7.0 + 5.0j]],
                    dtype=np.complex128,
                ),
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "L"}),
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="single", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert result.measurement_config.shot_averaging is False
    assert result.device_config == {"kind": "quel1"}
    assert set(result.data.keys()) == {"Q00"}
    assert len(result.data["Q00"]) == 1
    assert result.data["Q00"][0].sampling_period == 2.0
    assert_allclose(
        result.data["Q00"][0].data,
        np.array(
            [[4.0 - 2.0j], [5.0 - 3.0j], [6.0 - 4.0j], [7.0 - 5.0j]],
            dtype=np.complex128,
        )
        * norm_factor,
    )


def test_build_measurement_result_converts_avg_mode_with_shot_scaling() -> None:
    """Given QuEL-1 raw shots, avg mode should average waveforms in software."""
    norm_factor = 2 ** (-16)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex128),
                np.array(
                    [
                        [2.0 + 2.0j, 4.0 + 4.0j],
                        [4.0 + 2.0j, 6.0 + 4.0j],
                        [6.0 + 2.0j, 8.0 + 4.0j],
                        [8.0 + 2.0j, 10.0 + 4.0j],
                    ],
                    dtype=np.complex128,
                ),
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="avg", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert result.measurement_config.shot_averaging is True
    assert set(result.data.keys()) == {"Q00"}
    assert len(result.data["Q00"]) == 1
    assert_allclose(
        result.data["Q00"][0].data,
        np.array([5.0 + 2.0j, 7.0 + 4.0j], dtype=np.complex128) * norm_factor,
    )


def test_build_measurement_result_keeps_single_point_avg_mode_as_length_one_waveform() -> (
    None
):
    """Given one averaged waveform sample, conversion should keep a length-one waveform axis."""
    norm_factor = 2 ** (-16)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array(
                    [
                        [2.0 + 1.0j],
                        [4.0 + 2.0j],
                        [6.0 + 3.0j],
                        [8.0 + 4.0j],
                    ],
                    dtype=np.complex128,
                )
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
        constraint_profile=replace(
            MeasurementConstraintProfile.quel1(),
            require_workaround_capture=False,
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(mode="avg", shots=4),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert_allclose(
        result.data["Q00"][0].data,
        np.array([5.0 + 2.5j], dtype=np.complex128) * norm_factor,
    )


def test_build_measurement_result_software_integrates_single_mode_to_1d() -> None:
    """Given raw waveform shots, time integration should sum each shot in software."""
    norm_factor = 2 ** (-16)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array(
                    [
                        [8.0 + 4.0j, 2.0 + 1.0j],
                        [12.0 + 6.0j, 3.0 + 2.0j],
                    ],
                    dtype=np.complex128,
                )
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
        constraint_profile=replace(
            MeasurementConstraintProfile.quel1(),
            require_workaround_capture=False,
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(
            mode="single",
            shots=2,
            time_integration=True,
        ),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    assert_allclose(
        result.data["Q00"][0].data,
        np.array([10.0 + 5.0j, 15.0 + 8.0j], dtype=np.complex128) * norm_factor,
    )


def test_build_measurement_result_software_demodulates_before_summing(
    monkeypatch,
) -> None:
    """Given raw carrier shots, conversion should emulate QuEL-1 capture DSP."""
    norm_factor = 2 ** (-16)
    frequency = 0.125
    sample_index = np.arange(64, dtype=np.float64)
    carrier = np.exp(1j * 2.0 * np.pi * frequency * 2.0 * sample_index)
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.vstack([carrier, 2.0 * carrier]).astype(np.complex128),
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
        constraint_profile=replace(
            MeasurementConstraintProfile.quel1(),
            require_workaround_capture=False,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_result_demodulation_frequency",
        lambda *, target, schedule: frequency,
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(
            mode="single",
            shots=2,
            time_integration=True,
        ),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
    )

    fir = (
        np.asarray(CaptureParamTools.fir_coefficient(frequency), dtype=np.complex128)
        / 2**15
    )
    window = (
        np.asarray(
            CaptureParamTools.window_coefficient(frequency),
            dtype=np.complex128,
        )
        / 2**31
    )
    demodulated = np.convolve(carrier, fir, mode="full")[: carrier.size][::4]
    demodulated *= window[: demodulated.size]
    assert_allclose(
        result.data["Q00"][0].data,
        np.array(
            [np.sum(demodulated), 2.0 * np.sum(demodulated)],
            dtype=np.complex128,
        )
        * norm_factor,
        atol=1e-20,
    )
    assert result.data["Q00"][0].sampling_period == 8.0


def test_build_measurement_result_software_classifies_with_line_params() -> None:
    """Given line parameters, software classification should populate states."""
    backend_result = Quel1BackendExecutionResult(
        status={},
        data={
            "RQ00": [
                np.array(
                    [
                        [1.0 + 1.0j],
                        [-1.0 + 1.0j],
                        [1.0 - 1.0j],
                        [-1.0 - 1.0j],
                    ],
                    dtype=np.complex128,
                )
            ]
        },
        config={},
    )
    adapter = Quel1MeasurementBackendAdapter(
        backend_controller=cast(Any, object()),
        experiment_system=cast(
            Any,
            _ExperimentSystemStub(sideband_by_target={"RQ00": "U"}),
        ),
        constraint_profile=replace(
            MeasurementConstraintProfile.quel1(),
            require_workaround_capture=False,
        ),
    )

    result = adapter.build_measurement_result(
        backend_result=backend_result,
        measurement_config=_make_config(
            mode="single",
            shots=4,
            time_integration=True,
            state_classification=True,
        ),
        device_config={"kind": "quel1"},
        sampling_period=2.0,
        quel1_options=Quel1MeasurementOptions(
            classification_line_param0=(1.0, 0.0, 0.0),
            classification_line_param1=(0.0, 1.0, 0.0),
        ),
    )

    capture = result.data["Q00"][0]
    assert_allclose(
        capture.data,
        np.array([1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, -1.0 - 1.0j]) * 2 ** (-16),
    )
    assert_allclose(capture.state_series, np.array([3, 2, 1, 0]))

"""Tests for temporary S159A mux measurement helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from qubex.contrib import (
    disable_s159a_mux_measurement,
    enable_s159a_mux_measurement,
    s159a_mux_measurement,
)
from qubex.system import (
    Box,
    BoxType,
    CapPort,
    Chip,
    ControlParameters,
    ControlSystem,
    ExperimentSystem,
    GenPort,
    QuantumSystem,
    Target,
    WiringInfo,
)


class _SystemManagerStub:
    def __init__(self) -> None:
        self.events: list[tuple[str, list[str]]] = []
        self.push_calls: list[dict[str, Any]] = []
        self.sync_calls = 0
        self.backend_controller = _BackendControllerStub(self.events)

    def sync_experiment_system_to_backend_controller(self) -> None:
        self.sync_calls += 1
        self.events.append(("sync", []))

    def push(
        self,
        *,
        box_ids: list[str],
        target_labels: list[str],
        confirm: bool,
    ) -> None:
        self.push_calls.append(
            {
                "box_ids": list(box_ids),
                "target_labels": list(target_labels),
                "confirm": confirm,
            }
        )
        self.events.append(("push", list(box_ids)))


class _BackendControllerStub:
    def __init__(self, events: list[tuple[str, list[str]]]) -> None:
        self._events = events
        self.connect_calls: list[dict[str, Any]] = []

    def connect(self, box_names: list[str], *, parallel: bool | None = None) -> None:
        self.connect_calls.append(
            {
                "box_names": list(box_names),
                "parallel": parallel,
            }
        )
        self._events.append(("connect", list(box_names)))


@dataclass
class _ContextStub:
    experiment_system: ExperimentSystem
    system_manager: _SystemManagerStub
    qubit_labels: list[str]

    @property
    def mux_labels(self) -> list[str]:
        labels = {
            self.experiment_system.get_mux_by_qubit(qubit).label
            for qubit in self.qubit_labels
        }
        return sorted(labels)

    @property
    def box_ids(self) -> list[str]:
        return [
            box.id
            for box in self.experiment_system.get_boxes_for_qubits(self.qubit_labels)
        ]

    @property
    def targets(self) -> dict[str, Target]:
        return {
            target.label: target
            for target in self.experiment_system.targets
            if target.is_related_to_qubits(self.qubit_labels)
        }

    @property
    def backend_controller(self) -> _BackendControllerStub:
        return self.system_manager.backend_controller


@dataclass
class _ExperimentStub:
    ctx: _ContextStub


def _make_exp(*, active_mux: int = 8) -> tuple[_ExperimentStub, _SystemManagerStub]:
    chip = Chip.new("64Qv3", "test-chip", 64)
    for qubit in chip.qubits:
        qubit.control_frequency_ge_value = 6.6 + qubit.index * 0.001
        qubit.anharmonicity_value = -0.3
    for resonator in chip.resonators:
        resonator.readout_frequency_value = 10.0 + resonator.index * 0.001

    quantum_system = QuantumSystem(chip)
    s159a = Box.new(
        id="S159A",
        name="S159A",
        type=BoxType.QUEL1SE_A,
        address="10.1.0.159",
        adapter="dummy",
        port_numbers=[0, 1, 2, 3, 4, 9, 11],
    )
    r20a = Box.new(
        id="R20A",
        name="R20A",
        type=BoxType.QUBE_RIKEN_A,
        address="10.1.0.20",
        adapter="dummy",
        port_numbers=[0, 1, 2, 5, 6, 7, 8],
    )
    r26a = Box.new(
        id="R26A",
        name="R26A",
        type=BoxType.QUBE_RIKEN_A,
        address="10.1.0.26",
        adapter="dummy",
        port_numbers=[0, 1, 2, 5, 6, 7, 8],
    )
    control_system = ControlSystem([s159a, r20a, r26a])
    mux2 = quantum_system.get_mux(2)
    mux6 = quantum_system.get_mux(6)
    mux8 = quantum_system.get_mux(8)

    wiring_info = WiringInfo(
        ctrl=[
            *[
                (quantum_system.get_qubit(resonator.qubit), _get_gen_port(s159a, port))
                for resonator, port in zip(
                    mux6.resonators,
                    [2, 4, 9, 11],
                    strict=True,
                )
            ],
            *[
                (quantum_system.get_qubit(resonator.qubit), _get_gen_port(r20a, port))
                for resonator, port in zip(
                    mux2.resonators,
                    [5, 6, 7, 8],
                    strict=True,
                )
            ],
            *[
                (quantum_system.get_qubit(resonator.qubit), _get_gen_port(r26a, port))
                for resonator, port in zip(
                    mux8.resonators,
                    [5, 6, 7, 8],
                    strict=True,
                )
            ],
        ],
        read_out=[
            (mux2, _get_gen_port(r20a, 0)),
            (mux6, _get_gen_port(s159a, 1)),
            (mux8, _get_gen_port(r26a, 0)),
        ],
        read_in=[
            (mux2, _get_cap_port(r20a, 1)),
            (mux6, _get_cap_port(s159a, 0)),
            (mux8, _get_cap_port(r26a, 1)),
        ],
        pump=[
            (mux2, _get_gen_port(r20a, 2)),
            (mux6, _get_gen_port(s159a, 3)),
            (mux8, _get_gen_port(r26a, 2)),
        ],
    )
    control_params = _make_control_params(chip)
    experiment_system = ExperimentSystem(
        quantum_system=quantum_system,
        control_system=control_system,
        wiring_info=wiring_info,
        control_params=control_params,
    )
    system_manager = _SystemManagerStub()
    exp = _ExperimentStub(
        ctx=_ContextStub(
            experiment_system=experiment_system,
            system_manager=system_manager,
            qubit_labels=[
                resonator.qubit
                for resonator in quantum_system.get_mux(active_mux).resonators
            ],
        )
    )
    return exp, system_manager


def _make_control_params(chip: Chip) -> ControlParameters:
    qubits = [qubit.label for qubit in chip.qubits]
    muxes = [mux.index for mux in chip.muxes]
    return ControlParameters(
        frequency_margin={},
        control_amplitude={qubit: 0.1 for qubit in qubits},
        readout_amplitude={qubit: 0.01 for qubit in qubits},
        control_vatt={qubit: 100 + index for index, qubit in enumerate(qubits)},
        readout_vatt={mux: 200 + mux for mux in muxes},
        pump_vatt={mux: 300 + mux for mux in muxes},
        control_fsc={qubit: 4000 + index for index, qubit in enumerate(qubits)},
        readout_fsc={mux: 5000 + mux for mux in muxes},
        pump_fsc={mux: 6000 + mux for mux in muxes},
        capture_delay={mux: 100 + mux for mux in muxes},
        capture_delay_word={mux: 10 + mux for mux in muxes},
        jpa_params={
            mux: {
                "dc_voltage": 0.0,
                "pump_frequency": 10.0 + mux * 0.01,
                "pump_amplitude": 0.1,
            }
            for mux in muxes
        },
    )


def _get_gen_port(box: Box, port_number: int) -> GenPort:
    return cast(GenPort, box.get_port(port_number))


def _get_cap_port(box: Box, port_number: int) -> CapPort:
    return cast(CapPort, box.get_port(port_number))


def test_enable_routes_active_mux_through_s159a_and_pushes_active_boxes() -> None:
    """Given mux08 experiment, enabling override routes Q32 through S159A."""
    exp, system_manager = _make_exp()
    system = exp.ctx.experiment_system

    assert system.get_ge_target("Q32").channel.port.box_id == "R26A"
    assert system.get_read_out_target("Q32").channel.port.box_id == "R26A"

    enable_s159a_mux_measurement(exp, target_mux=8, confirm=False)

    ge_target = system.get_ge_target("Q32")
    read_out_target = system.get_read_out_target("Q32")
    read_in_target = system.get_read_in_target("Q32")
    assert ge_target.channel.port.box_id == "S159A"
    assert read_out_target.channel.port.box_id == "S159A"
    assert read_in_target.channel.port.box_id == "S159A"
    assert ge_target.channel.port.vatt == 132
    assert read_out_target.channel.port.vatt == 208
    assert read_in_target.channel.ndelay == 108
    assert read_out_target.frequency == system.get_resonator("RQ32").frequency
    assert system_manager.sync_calls == 1
    assert system_manager.backend_controller.connect_calls == [
        {
            "box_names": ["S159A"],
            "parallel": None,
        }
    ]
    assert len(system_manager.push_calls) == 1
    assert set(system_manager.push_calls[0]["box_ids"]) == {"S159A"}
    assert system_manager.push_calls[0]["confirm"] is False
    assert system_manager.events == [
        ("sync", []),
        ("connect", ["S159A"]),
        ("push", ["S159A"]),
    ]


def test_enable_target_mux2_routes_q10_through_s159a() -> None:
    """Given mux02 experiment, enabling override routes Q10 through S159A."""
    exp, system_manager = _make_exp(active_mux=2)
    system = exp.ctx.experiment_system

    assert system.get_ge_target("Q10").channel.port.box_id == "R20A"
    assert system.get_read_out_target("Q10").channel.port.box_id == "R20A"

    enable_s159a_mux_measurement(exp, target_mux=2, confirm=False)

    assert system.get_ge_target("Q10").channel.port.box_id == "S159A"
    assert system.get_read_out_target("Q10").channel.port.box_id == "S159A"
    assert system.get_read_in_target("Q10").channel.port.box_id == "S159A"
    assert system_manager.sync_calls == 1
    assert system_manager.backend_controller.connect_calls == [
        {
            "box_names": ["S159A"],
            "parallel": None,
        }
    ]
    assert set(system_manager.push_calls[0]["box_ids"]) == {"S159A"}


def test_disable_restores_original_mux_wiring_and_pushes_active_boxes() -> None:
    """Given an active override, disabling restores mux08 to its original box."""
    exp, system_manager = _make_exp()
    system = exp.ctx.experiment_system

    enable_s159a_mux_measurement(exp, target_mux=8, confirm=False)
    disable_s159a_mux_measurement(exp, confirm=False)

    assert system.get_ge_target("Q32").channel.port.box_id == "R26A"
    assert system.get_read_out_target("Q32").channel.port.box_id == "R26A"
    assert system_manager.sync_calls == 2
    assert system_manager.backend_controller.connect_calls == [
        {
            "box_names": ["S159A"],
            "parallel": None,
        },
        {
            "box_names": ["R26A"],
            "parallel": None,
        },
    ]
    assert len(system_manager.push_calls) == 2
    assert set(system_manager.push_calls[-1]["box_ids"]) == {"R26A"}


def test_context_manager_restores_original_wiring_after_exception() -> None:
    """Given context-managed override, exit restores runtime wiring."""
    exp, system_manager = _make_exp()
    system = exp.ctx.experiment_system

    with pytest.raises(RuntimeError, match="boom"):
        with s159a_mux_measurement(exp, target_mux=8, push=False):
            assert system.get_ge_target("Q32").channel.port.box_id == "S159A"
            raise RuntimeError("boom")

    assert system.get_ge_target("Q32").channel.port.box_id == "R26A"
    assert system_manager.sync_calls == 0
    assert system_manager.backend_controller.connect_calls == []
    assert system_manager.push_calls == []


def test_enable_push_false_does_not_push_hardware_settings() -> None:
    """Given push disabled, enabling only changes runtime wiring."""
    exp, system_manager = _make_exp()
    system = exp.ctx.experiment_system

    enable_s159a_mux_measurement(exp, target_mux=8, push=False)

    assert system.get_ge_target("Q32").channel.port.box_id == "S159A"
    assert system_manager.sync_calls == 0
    assert system_manager.backend_controller.connect_calls == []
    assert system_manager.push_calls == []

"""Temporary S159A mux-wiring overrides for experiment measurements."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from rich.prompt import Confirm

from qubex.system import CapPort, ExperimentSystem, GenPort, Mux, Qubit, WiringInfo

__all__ = [
    "disable_s159a_mux_measurement",
    "enable_s159a_mux_measurement",
    "s159a_mux_measurement",
]

_DEFAULT_S159A_MUX = 6
_STATE_ATTR = "_s159a_mux_measurement_state"

PortT = TypeVar("PortT", GenPort, CapPort)


@dataclass(frozen=True)
class _S159AMuxMeasurementState:
    original_wiring_info: WiringInfo
    target_mux_label: str
    s159a_mux_label: str


def enable_s159a_mux_measurement(
    exp: Any,
    *,
    target_mux: int | str | None = None,
    s159a_mux: int | str = _DEFAULT_S159A_MUX,
    push: bool = True,
    confirm: bool = True,
) -> None:
    """
    Temporarily route one active mux through the S159A mux wiring.

    The config files are not modified. Runtime wiring is swapped in memory,
    port parameters and target mappings are rebuilt. When ``push`` is true,
    the backend model is rebuilt, active boxes are reconnected, and settings are
    pushed to hardware.
    """
    system = _get_experiment_system(exp)
    target_mux_obj = _resolve_target_mux(exp, target_mux)
    s159a_mux_obj = _resolve_mux(system, s159a_mux)
    if target_mux_obj.label == s159a_mux_obj.label:
        raise ValueError("`target_mux` and `s159a_mux` must be different muxes.")

    active_state = _get_state(system)
    if active_state is not None:
        if (
            active_state.target_mux_label == target_mux_obj.label
            and active_state.s159a_mux_label == s159a_mux_obj.label
        ):
            return
        raise RuntimeError(
            "S159A mux measurement override is already active for "
            f"{active_state.s159a_mux_label}<->{active_state.target_mux_label}. "
            "Disable it before enabling another mux override."
        )

    original_wiring_info = system.copy_wiring_info()
    swapped_wiring_info = _swap_mux_wiring(
        wiring_info=original_wiring_info,
        s159a_mux=s159a_mux_obj,
        target_mux=target_mux_obj,
    )
    system.replace_wiring_info(swapped_wiring_info)
    if push and not _confirm_active_boxes(
        exp,
        confirm=confirm,
        action="enable S159A mux measurement override",
    ):
        system.replace_wiring_info(original_wiring_info)
        raise RuntimeError("S159A mux measurement override was cancelled.")
    try:
        if push:
            _configure_active_boxes(exp)
    except Exception:
        system.replace_wiring_info(original_wiring_info)
        _restore_backend_runtime(exp)
        raise

    _set_state(
        system,
        _S159AMuxMeasurementState(
            original_wiring_info=original_wiring_info,
            target_mux_label=target_mux_obj.label,
            s159a_mux_label=s159a_mux_obj.label,
        ),
    )


def disable_s159a_mux_measurement(
    exp: Any,
    *,
    push: bool = True,
    confirm: bool = True,
) -> None:
    """
    Restore wiring previously changed by ``enable_s159a_mux_measurement``.

    If no override is active on the current experiment system, this function is
    a no-op.
    """
    system = _get_experiment_system(exp)
    active_state = _get_state(system)
    if active_state is None:
        return

    current_wiring_info = system.copy_wiring_info()
    system.replace_wiring_info(active_state.original_wiring_info)
    if push and not _confirm_active_boxes(
        exp,
        confirm=confirm,
        action="disable S159A mux measurement override",
    ):
        system.replace_wiring_info(current_wiring_info)
        raise RuntimeError("S159A mux measurement restore was cancelled.")
    try:
        if push:
            _configure_active_boxes(exp)
    except Exception:
        system.replace_wiring_info(current_wiring_info)
        _restore_backend_runtime(exp)
        raise

    _clear_state(system)


@contextmanager
def s159a_mux_measurement(
    exp: Any,
    *,
    target_mux: int | str | None = None,
    s159a_mux: int | str = _DEFAULT_S159A_MUX,
    push: bool = True,
    confirm: bool = True,
) -> Iterator[Any]:
    """Temporarily enable S159A mux measurement within a context block."""
    system = _get_experiment_system(exp)
    was_active = _get_state(system) is not None
    enable_s159a_mux_measurement(
        exp,
        target_mux=target_mux,
        s159a_mux=s159a_mux,
        push=push,
        confirm=confirm,
    )
    try:
        yield exp
    finally:
        if not was_active:
            disable_s159a_mux_measurement(
                exp,
                push=push,
                confirm=confirm,
            )


def _get_experiment_system(exp: Any) -> ExperimentSystem:
    try:
        system = exp.ctx.experiment_system
    except AttributeError:
        raise TypeError("`exp` must be a qubex Experiment-like object.") from None
    if not isinstance(system, ExperimentSystem):
        raise TypeError("`exp.ctx.experiment_system` must be an ExperimentSystem.")
    return system


def _resolve_target_mux(exp: Any, target_mux: int | str | None) -> Mux:
    system = _get_experiment_system(exp)
    if target_mux is None:
        active_mux_labels = list(exp.ctx.mux_labels)
        if len(active_mux_labels) != 1:
            raise ValueError(
                "`target_mux` must be specified when the experiment does not have "
                "exactly one active mux."
            )
        target_mux = active_mux_labels[0]

    mux = _resolve_mux(system, target_mux)
    active_mux_labels = set(exp.ctx.mux_labels)
    if active_mux_labels and mux.label not in active_mux_labels:
        raise ValueError(
            f"`target_mux` {mux.label} is not active in this Experiment "
            f"(active muxes: {sorted(active_mux_labels)})."
        )
    return mux


def _resolve_mux(system: ExperimentSystem, mux: int | str) -> Mux:
    try:
        return system.get_mux(mux)
    except Exception:
        raise ValueError(f"Mux `{mux}` could not be resolved.") from None


def _swap_mux_wiring(
    *,
    wiring_info: WiringInfo,
    s159a_mux: Mux,
    target_mux: Mux,
) -> WiringInfo:
    return WiringInfo(
        ctrl=_swap_ctrl_ports(
            wiring_info.ctrl,
            s159a_mux=s159a_mux,
            target_mux=target_mux,
        ),
        read_out=_swap_mux_ports(
            wiring_info.read_out,
            s159a_mux=s159a_mux,
            target_mux=target_mux,
            group_name="read_out",
        ),
        read_in=_swap_mux_ports(
            wiring_info.read_in,
            s159a_mux=s159a_mux,
            target_mux=target_mux,
            group_name="read_in",
        ),
        pump=_swap_mux_ports(
            wiring_info.pump,
            s159a_mux=s159a_mux,
            target_mux=target_mux,
            group_name="pump",
        ),
    )


def _swap_ctrl_ports(
    pairs: list[tuple[Qubit, GenPort]],
    *,
    s159a_mux: Mux,
    target_mux: Mux,
) -> list[tuple[Qubit, GenPort]]:
    port_by_qubit = {qubit.label: port for qubit, port in pairs}
    s159a_qubits = _mux_qubit_labels(s159a_mux)
    target_qubits = _mux_qubit_labels(target_mux)
    if len(s159a_qubits) != len(target_qubits):
        raise ValueError("S159A mux and target mux must contain the same qubit count.")

    missing = [
        qubit
        for qubit in [*s159a_qubits, *target_qubits]
        if qubit not in port_by_qubit
    ]
    if missing:
        raise ValueError(f"Control wiring is missing qubits: {missing}.")

    s159a_ports = [port_by_qubit[qubit] for qubit in s159a_qubits]
    target_ports = [port_by_qubit[qubit] for qubit in target_qubits]
    for qubit, port in zip(s159a_qubits, target_ports, strict=True):
        port_by_qubit[qubit] = port
    for qubit, port in zip(target_qubits, s159a_ports, strict=True):
        port_by_qubit[qubit] = port

    return [(qubit, port_by_qubit[qubit.label]) for qubit, _port in pairs]


def _swap_mux_ports(
    pairs: list[tuple[Mux, PortT]],
    *,
    s159a_mux: Mux,
    target_mux: Mux,
    group_name: str,
) -> list[tuple[Mux, PortT]]:
    port_by_mux = {mux.label: port for mux, port in pairs}
    missing = [
        mux.label
        for mux in (s159a_mux, target_mux)
        if mux.label not in port_by_mux
    ]
    if missing:
        raise ValueError(f"{group_name} wiring is missing muxes: {missing}.")

    s159a_port = port_by_mux[s159a_mux.label]
    target_port = port_by_mux[target_mux.label]
    swapped: list[tuple[Mux, PortT]] = []
    for mux, port in pairs:
        if mux.label == s159a_mux.label:
            swapped.append((mux, target_port))
        elif mux.label == target_mux.label:
            swapped.append((mux, s159a_port))
        else:
            swapped.append((mux, port))
    return swapped


def _mux_qubit_labels(mux: Mux) -> list[str]:
    return [resonator.qubit for resonator in mux.resonators]


def _confirm_active_boxes(
    exp: Any,
    *,
    confirm: bool,
    action: str,
) -> bool:
    box_ids = exp.ctx.box_ids
    if not box_ids:
        return True
    if confirm and not _confirm_push(exp, box_ids=box_ids, action=action):
        return False
    return True


def _configure_active_boxes(exp: Any) -> None:
    box_ids = list(exp.ctx.box_ids)
    _sync_backend_model(exp)
    if not box_ids:
        return
    _reconnect_active_boxes(exp, box_ids=box_ids)
    _push_active_boxes(exp, box_ids=box_ids)


def _restore_backend_runtime(exp: Any) -> None:
    _sync_backend_model(exp)
    box_ids = list(exp.ctx.box_ids)
    if box_ids:
        _reconnect_active_boxes(exp, box_ids=box_ids)


def _sync_backend_model(exp: Any) -> None:
    system_manager = exp.ctx.system_manager
    sync_backend_model = getattr(
        system_manager,
        "sync_experiment_system_to_backend_controller",
        None,
    )
    if callable(sync_backend_model):
        sync_backend_model()


def _reconnect_active_boxes(exp: Any, *, box_ids: list[str]) -> None:
    backend_controller = getattr(exp.ctx, "backend_controller", None)
    if backend_controller is None:
        backend_controller = getattr(exp.ctx.system_manager, "backend_controller", None)
    connect = getattr(backend_controller, "connect", None)
    if callable(connect):
        connect(box_ids, parallel=None)


def _push_active_boxes(exp: Any, *, box_ids: list[str]) -> None:
    system_manager = exp.ctx.system_manager
    system_manager.push(
        box_ids=box_ids,
        target_labels=list(exp.ctx.targets),
        confirm=False,
    )


def _confirm_push(exp: Any, *, box_ids: list[str], action: str) -> bool:
    boxes = [exp.ctx.experiment_system.get_box(box_id) for box_id in box_ids]
    boxes_str = "\n".join(f"{box.id} ({box.name})" for box in boxes)
    return Confirm.ask(
        f"""
You are going to {action} and configure the following boxes:

[bold bright_green]{boxes_str}[/bold bright_green]

This operation will overwrite the existing backend settings. Do you want to continue?
"""
    )


def _get_state(system: ExperimentSystem) -> _S159AMuxMeasurementState | None:
    state = getattr(system, _STATE_ATTR, None)
    if state is not None and not isinstance(state, _S159AMuxMeasurementState):
        raise RuntimeError("Invalid S159A mux measurement override state.")
    return state


def _set_state(system: ExperimentSystem, state: _S159AMuxMeasurementState) -> None:
    setattr(system, _STATE_ATTR, state)


def _clear_state(system: ExperimentSystem) -> None:
    if hasattr(system, _STATE_ATTR):
        delattr(system, _STATE_ATTR)

"""
Build and run a distance-2 surface-code experiment with serialized CR gates.

This module deliberately keeps the surface-code circuit separate from the Qubex
adapter.  Every entangling operation exposed to the circuit is a logical
``CX(data, ancilla)`` macro; a raw ZX90 pulse is never treated as a CNOT.

Qubex 1.5 has no mid-circuit conditional feedback or qubit active-reset API.
Consequently, the safe default is one stabilizer round.  Repeated rounds require
either a user-supplied reset schedule or an explicitly configured, calibrated
passive-reset wait.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from qxpulse import Blank, PulseArray, PulseSchedule, Waveform

from qubex.experiment import Experiment
from qubex.experiment.experiment_exceptions import CalibrationMissingError
from qubex.measurement import MeasurementSchedule
from qubex.measurement.models.measure_result import MultipleMeasureResult

SymbolicQubit = Literal["D1", "D2", "D3", "D4", "A1", "A2", "A3"]
DataQubit = Literal["D1", "D2", "D3", "D4"]
AncillaQubit = Literal["A1", "A2", "A3"]
Basis = Literal["Z", "X"]
MeasurementAxis = Literal["X", "Y", "Z"]
LogicalState = Literal["0L", "1L", "+L", "-L", "custom"]
SyndromeMode = Literal["record_only", "postselect", "pauli_frame"]
ResetStrategy = Literal["shot_boundary", "passive", "custom"]
HadamardDecomposition = Literal["Z180-Y90", "Y90-X180"]
OperationKind = Literal[
    "RESET",
    "XPI",
    "H",
    "BASIS_ROTATION",
    "CX_CR",
    "READOUT",
    "DD",
    "DELAY",
]

DATA_QUBITS: tuple[DataQubit, ...] = ("D1", "D2", "D3", "D4")
ANCILLA_QUBITS: tuple[AncillaQubit, ...] = ("A1", "A2", "A3")
SYMBOLIC_QUBITS: tuple[SymbolicQubit, ...] = (*DATA_QUBITS, *ANCILLA_QUBITS)

_CANONICAL_SEEDS: dict[str, tuple[str, Basis]] = {
    "0L": ("0000", "Z"),
    "1L": ("0011", "Z"),
    "+L": ("0000", "X"),
    "-L": ("0101", "X"),
}
_ALLOWED_CR_OVERRIDE_KEYS = frozenset(
    {
        "cr_duration",
        "cr_ramptime",
        "cr_amplitude",
        "cr_phase",
        "cr_beta",
        "cancel_amplitude",
        "cancel_phase",
        "cancel_beta",
        "rotary_amplitude",
        "echo",
        "x180_margin",
    }
)


@dataclass(frozen=True)
class DirectedEdge:
    """
    Describe one mandatory data-control, ancilla-target edge.

    Parameters
    ----------
    edge_id
        Stable symbolic edge identifier.
    control
        Data-qubit control.
    target
        Ancilla-qubit target.
    """

    edge_id: str
    control: DataQubit
    target: AncillaQubit

    @property
    def qubits(self) -> tuple[DataQubit, AncillaQubit]:
        """Return the directed symbolic pair."""
        return self.control, self.target


@dataclass(frozen=True)
class Layer:
    """
    Describe one serialized layer of the stabilizer round.

    ``edges`` retains the order of the two CR slots from the original parallel
    layer table.  It must never be sorted by qubit label.
    """

    name: Literal["L1", "L2", "L3", "L4"]
    x_check_data: DataQubit
    edges: tuple[DirectedEdge, DirectedEdge]


LAYERS: tuple[Layer, ...] = (
    Layer(
        name="L1",
        x_check_data="D1",
        edges=(
            DirectedEdge("D1_A2", "D1", "A2"),
            DirectedEdge("D3_A3", "D3", "A3"),
        ),
    ),
    Layer(
        name="L2",
        x_check_data="D3",
        edges=(
            DirectedEdge("D3_A2", "D3", "A2"),
            DirectedEdge("D4_A3", "D4", "A3"),
        ),
    ),
    Layer(
        name="L3",
        x_check_data="D2",
        edges=(
            DirectedEdge("D1_A1", "D1", "A1"),
            DirectedEdge("D2_A2", "D2", "A2"),
        ),
    ),
    Layer(
        name="L4",
        x_check_data="D4",
        edges=(
            DirectedEdge("D2_A1", "D2", "A1"),
            DirectedEdge("D4_A2", "D4", "A2"),
        ),
    ),
)
REQUIRED_EDGES: tuple[DirectedEdge, ...] = tuple(
    edge for layer in LAYERS for edge in layer.edges
)
_ALLOWED_CR_OVERRIDE_TARGETS = frozenset(
    target
    for layer in LAYERS
    for slot, edge in enumerate(layer.edges, start=1)
    for target in (
        edge.edge_id,
        f"{edge.edge_id}@serial:{layer.name}:S{slot}",
    )
)


@dataclass(frozen=True)
class AbstractOperation:
    """Represent one semantic operation in the fixed serial round."""

    kind: Literal["H", "CX_CR"]
    qubits: tuple[SymbolicQubit, ...]
    layer: str
    slot: int | None = None
    edge_id: str | None = None


def serial_round_operations() -> tuple[AbstractOperation, ...]:
    """
    Return the immutable semantic operation list for one stabilizer round.

    Returns
    -------
    tuple[AbstractOperation, ...]
        Operations in strict execution order.  Each layer is
        ``pre-H, slot-1 CX, slot-2 CX, post-H``.
    """
    operations: list[AbstractOperation] = []
    for layer in LAYERS:
        operations.append(AbstractOperation("H", (layer.x_check_data,), layer.name))
        for slot, edge in enumerate(layer.edges, start=1):
            operations.append(
                AbstractOperation(
                    "CX_CR",
                    edge.qubits,
                    layer.name,
                    slot=slot,
                    edge_id=edge.edge_id,
                )
            )
        operations.append(AbstractOperation("H", (layer.x_check_data,), layer.name))
    return tuple(operations)


def canonical_seed(logical_state: str) -> tuple[str, str]:
    """
    Resolve a canonical logical-state label to a seed and preparation basis.

    Parameters
    ----------
    logical_state
        One of ``0L``, ``1L``, ``+L``, or ``-L``.

    Returns
    -------
    tuple[str, str]
        Four-bit seed and ``Z``/``X`` basis.

    Raises
    ------
    ValueError
        If ``logical_state`` is not canonical.
    """
    try:
        bitstring, basis = _CANONICAL_SEEDS[logical_state]
    except KeyError as exc:
        raise ValueError("logical_state must be one of 0L, 1L, +L, or -L.") from exc
    return bitstring, basis


def _validate_bitstring(bitstring: str, *, length: int, name: str) -> None:
    """Validate a fixed-length binary string."""
    if len(bitstring) != length or set(bitstring) - {"0", "1"}:
        raise ValueError(f"{name} must be a {length}-bit binary string.")


def _validate_bit_tuple(bits: Sequence[int], *, length: int, name: str) -> None:
    """Validate a fixed-length sequence of binary integers."""
    if len(bits) != length or any(bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must contain exactly {length} binary values.")


def _strict_bool(
    values: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    """Read a JSON boolean without accepting truthy strings or numbers."""
    value = values.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a JSON boolean.")
    return value


def _strict_bool_mapping(
    values: Mapping[str, Any],
    key: str,
) -> dict[str, bool]:
    """Read a mapping whose values must be JSON booleans."""
    raw = values.get(key, {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"{key} must be a JSON object.")
    result: dict[str, bool] = {}
    for raw_key, value in raw.items():
        if not isinstance(value, bool):
            raise TypeError(f"{key}.{raw_key} must be a JSON boolean.")
        result[str(raw_key)] = value
    return result


def _cr_overrides_from_mapping(
    values: Mapping[str, Any],
) -> dict[str, dict[str, float | bool | None]]:
    """Decode the nested CR-override object without truthy coercions."""
    raw = values.get("cr_overrides", {})
    if not isinstance(raw, Mapping):
        raise TypeError("cr_overrides must be a JSON object.")
    result: dict[str, dict[str, float | bool | None]] = {}
    for raw_target, raw_override in raw.items():
        if not isinstance(raw_override, Mapping):
            raise TypeError(f"cr_overrides.{raw_target} must be a JSON object.")
        decoded: dict[str, float | bool | None] = {}
        for raw_name, value in raw_override.items():
            name = str(raw_name)
            if value is not None and not isinstance(value, (bool, int, float)):
                raise TypeError(
                    f"cr_overrides.{raw_target}.{name} must be numeric, bool, or null."
                )
            decoded[name] = value
        result[str(raw_target)] = decoded
    return result


def _pauli_word_is_identity(axes: Sequence[str], repeat_count: int) -> bool:
    """Return whether ideal pi rotations compose to identity up to global phase."""
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    }
    unitary = np.eye(2, dtype=np.complex128)
    for _ in range(repeat_count):
        for axis in axes:
            unitary = (-1j * paulis[axis]) @ unitary
    phase = unitary[0, 0]
    if np.isclose(abs(phase), 0.0):
        return False
    return bool(np.allclose(unitary, phase * np.eye(2), rtol=1e-10, atol=1e-12))


@dataclass(frozen=True)
class DDConfig:
    """
    Configure optional data-qubit dynamical decoupling.

    Pulse axes and repetition count are user supplied rather than embedded in
    the surface-code circuit.  The resulting ideal action must be identity up
    to global phase.
    """

    enabled: bool = False
    pulse_axes: tuple[Literal["X", "Y"], ...] = ()
    repeat_count: int = 1
    during_syndrome_readout: bool = True
    during_reset: bool = True

    def __post_init__(self) -> None:
        """Validate the configured DD word."""
        for name in ("enabled", "during_syndrome_readout", "during_reset"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"DD {name} must be bool.")
        if self.repeat_count < 1:
            raise ValueError("DD repeat_count must be at least 1.")
        if self.enabled and not self.pulse_axes:
            raise ValueError("DD pulse_axes must be supplied when DD is enabled.")
        if any(axis not in ("X", "Y") for axis in self.pulse_axes):
            raise ValueError("DD pulse_axes may contain only X and Y.")
        if self.enabled and not _pauli_word_is_identity(
            self.pulse_axes, self.repeat_count
        ):
            raise ValueError(
                "The configured DD word must implement identity up to global phase."
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> DDConfig:
        """Construct a DD configuration from decoded JSON."""
        if values is None:
            return cls()
        axes = tuple(str(axis).upper() for axis in values.get("pulse_axes", ()))
        return cls(
            enabled=_strict_bool(values, "enabled", default=False),
            pulse_axes=cast(tuple[Literal["X", "Y"], ...], axes),
            repeat_count=int(values.get("repeat_count", 1)),
            during_syndrome_readout=_strict_bool(
                values, "during_syndrome_readout", default=True
            ),
            during_reset=_strict_bool(values, "during_reset", default=True),
        )


@dataclass(frozen=True)
class SurfaceCodeConfig:
    """
    Configure the logical circuit, execution, and analysis policy.

    All physical durations specific to an experiment remain ``None`` unless the
    user supplies them or Qubex loads them from calibration/configuration data.
    """

    qubit_map: Mapping[str, str]
    logical_state: LogicalState = "0L"
    custom_bitstring: str | None = None
    initial_basis: Basis | None = None
    n_stabilizer_rounds: int = 1
    final_basis: Basis = "Z"
    final_measurement_axes: tuple[MeasurementAxis, ...] | None = None
    syndrome_mode: SyndromeMode = "record_only"
    postselect_syndrome: tuple[int, int, int] = (0, 0, 0)
    n_shots: int = 1_024
    shot_interval_ns: float | None = None
    use_simultaneous_cr: bool = False
    reset_strategy: ResetStrategy = "shot_boundary"
    passive_reset_delay_ns: float | None = None
    hadamard_decomposition: HadamardDecomposition = "Y90-X180"
    readout_inversion: Mapping[str, bool] = field(default_factory=dict)
    data_readout_inversion: Mapping[str, bool] = field(default_factory=dict)
    dd: DDConfig = field(default_factory=DDConfig)
    cr_overrides: Mapping[str, Mapping[str, float | bool | None]] = field(
        default_factory=dict
    )
    classifier_threshold: float | None = None
    initial_ground_postselection: bool = False
    initial_readout_recovery_delay_ns: float | None = None
    acknowledge_native_frame_calibration: bool = False
    acknowledge_initial_ground_state: bool = False
    acknowledge_initial_readout_calibration: bool = False
    max_waveform_amplitude: float | None = None

    def __post_init__(self) -> None:
        """Validate the configuration without touching hardware."""
        keys: set[str] = set(self.qubit_map)
        expected: set[str] = set(SYMBOLIC_QUBITS)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"qubit_map keys must be exactly {SYMBOLIC_QUBITS}; "
                f"missing={missing}, extra={extra}."
            )
        physical = [self.qubit_map[symbol] for symbol in SYMBOLIC_QUBITS]
        if len(set(physical)) != len(physical):
            raise ValueError("qubit_map must be one-to-one.")
        if any(not label for label in physical):
            raise ValueError("Physical qubit labels must be non-empty.")
        if self.logical_state not in (*_CANONICAL_SEEDS, "custom"):
            raise ValueError("Invalid logical_state.")
        if self.logical_state == "custom":
            if self.custom_bitstring is None or self.initial_basis is None:
                raise ValueError(
                    "custom logical_state requires custom_bitstring and initial_basis."
                )
            _validate_bitstring(
                self.custom_bitstring, length=4, name="custom_bitstring"
            )
        if self.initial_basis is not None and self.initial_basis not in ("Z", "X"):
            raise ValueError("initial_basis must be Z or X.")
        if self.n_stabilizer_rounds < 1:
            raise ValueError("n_stabilizer_rounds must be at least 1.")
        if self.final_basis not in ("Z", "X"):
            raise ValueError("final_basis must be Z or X.")
        if self.final_measurement_axes is not None:
            if len(self.final_measurement_axes) != len(DATA_QUBITS):
                raise ValueError(
                    "final_measurement_axes must contain one axis for each data qubit."
                )
            if any(axis not in ("X", "Y", "Z") for axis in self.final_measurement_axes):
                raise ValueError("final_measurement_axes may contain only X, Y, or Z.")
        if self.syndrome_mode not in (
            "record_only",
            "postselect",
            "pauli_frame",
        ):
            raise ValueError("Invalid syndrome_mode.")
        _validate_bit_tuple(
            self.postselect_syndrome,
            length=3,
            name="postselect_syndrome",
        )
        if self.n_shots < 1:
            raise ValueError("n_shots must be at least 1.")
        if self.shot_interval_ns is not None and self.shot_interval_ns <= 0:
            raise ValueError("shot_interval_ns must be positive when supplied.")
        if not isinstance(self.use_simultaneous_cr, bool):
            raise TypeError("use_simultaneous_cr must be bool.")
        if self.use_simultaneous_cr:
            raise ValueError("This implementation supports serial CR only.")
        if not isinstance(self.acknowledge_native_frame_calibration, bool):
            raise TypeError("acknowledge_native_frame_calibration must be bool.")
        if not isinstance(self.acknowledge_initial_ground_state, bool):
            raise TypeError("acknowledge_initial_ground_state must be bool.")
        if not isinstance(self.initial_ground_postselection, bool):
            raise TypeError("initial_ground_postselection must be bool.")
        if not isinstance(self.acknowledge_initial_readout_calibration, bool):
            raise TypeError("acknowledge_initial_readout_calibration must be bool.")
        if self.initial_readout_recovery_delay_ns is not None:
            if (
                not math.isfinite(self.initial_readout_recovery_delay_ns)
                or self.initial_readout_recovery_delay_ns < 0
            ):
                raise ValueError(
                    "initial_readout_recovery_delay_ns must be non-negative."
                )
        if self.reset_strategy not in ("shot_boundary", "passive", "custom"):
            raise ValueError("Invalid reset_strategy.")
        if self.n_stabilizer_rounds > 1 and self.reset_strategy == "shot_boundary":
            raise ValueError(
                "Repeated rounds require an explicit ancilla reset strategy."
            )
        if self.reset_strategy == "passive":
            if self.passive_reset_delay_ns is None or self.passive_reset_delay_ns <= 0:
                raise ValueError(
                    "Passive reset requires a positive passive_reset_delay_ns."
                )
        if self.hadamard_decomposition not in ("Z180-Y90", "Y90-X180"):
            raise ValueError("Invalid hadamard_decomposition.")
        self._validate_inversion_map(
            self.readout_inversion, ANCILLA_QUBITS, "readout_inversion"
        )
        self._validate_inversion_map(
            self.data_readout_inversion,
            DATA_QUBITS,
            "data_readout_inversion",
        )
        for override_key, values in self.cr_overrides.items():
            if override_key not in _ALLOWED_CR_OVERRIDE_TARGETS:
                raise ValueError(
                    f"Unknown CR override target {override_key!r}; expected an edge "
                    "ID or its exact EDGE@serial:LAYER:SLOT context."
                )
            unknown = set(values) - _ALLOWED_CR_OVERRIDE_KEYS
            if unknown:
                raise ValueError(
                    f"Unsupported CR override(s) for {override_key}: {sorted(unknown)}."
                )
            for name, value in values.items():
                if value is None:
                    continue
                if name == "echo":
                    if not isinstance(value, bool):
                        raise TypeError(
                            f"CR override {override_key}.echo must be bool."
                        )
                    continue
                if isinstance(value, bool):
                    raise TypeError(
                        f"CR override {override_key}.{name} must be numeric."
                    )
                if not math.isfinite(float(value)):
                    raise ValueError(
                        f"CR override {override_key}.{name} must be finite."
                    )
        if self.classifier_threshold is not None and not (
            0.0 <= self.classifier_threshold <= 1.0
        ):
            raise ValueError("classifier_threshold must lie in [0, 1].")
        if self.max_waveform_amplitude is not None:
            if (
                not math.isfinite(self.max_waveform_amplitude)
                or self.max_waveform_amplitude <= 0
            ):
                raise ValueError("max_waveform_amplitude must be positive.")

    @staticmethod
    def _validate_inversion_map(
        values: Mapping[str, bool],
        allowed: Sequence[str],
        name: str,
    ) -> None:
        """Validate readout-inversion keys and values."""
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown {name} keys: {sorted(unknown)}.")
        if any(not isinstance(value, bool) for value in values.values()):
            raise ValueError(f"All {name} values must be bool.")

    @property
    def resolved_seed(self) -> tuple[str, Basis]:
        """Return the configured four-bit seed and preparation basis."""
        if self.logical_state == "custom":
            if self.custom_bitstring is None or self.initial_basis is None:
                raise RuntimeError("Validated custom seed is unexpectedly incomplete.")
            return self.custom_bitstring, self.initial_basis
        bitstring, basis = canonical_seed(self.logical_state)
        return bitstring, cast(Basis, basis)

    @property
    def resolved_final_measurement_axes(
        self,
    ) -> tuple[MeasurementAxis, MeasurementAxis, MeasurementAxis, MeasurementAxis]:
        """Return final data measurement axes in `D1,D2,D3,D4` order."""
        if self.final_measurement_axes is None:
            return cast(
                tuple[
                    MeasurementAxis,
                    MeasurementAxis,
                    MeasurementAxis,
                    MeasurementAxis,
                ],
                (self.final_basis,) * len(DATA_QUBITS),
            )
        return cast(
            tuple[
                MeasurementAxis,
                MeasurementAxis,
                MeasurementAxis,
                MeasurementAxis,
            ],
            self.final_measurement_axes,
        )

    @property
    def logical_measurement_basis(self) -> Basis | None:
        """Return the measured logical basis, or `None` for mixed tomography axes."""
        axes = self.resolved_final_measurement_axes
        if all(axis == "Z" for axis in axes):
            return "Z"
        if all(axis == "X" for axis in axes):
            return "X"
        return None

    def cr_override(
        self,
        edge_id: str,
        context: str,
    ) -> dict[str, float | bool | None]:
        """
        Resolve a context-specific CR override.

        ``EDGE@CONTEXT`` has priority over ``EDGE``.  The context form replaces
        only keys it defines, allowing a shared edge default plus a serial-slot
        correction.
        """
        result = dict(self.cr_overrides.get(edge_id, {}))
        result.update(self.cr_overrides.get(f"{edge_id}@{context}", {}))
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SurfaceCodeConfig:
        """Construct a surface-code configuration from decoded JSON."""
        postselect = tuple(
            int(bit) for bit in values.get("postselect_syndrome", (0, 0, 0))
        )
        return cls(
            qubit_map={
                str(key): str(value)
                for key, value in cast(Mapping[str, Any], values["qubit_map"]).items()
            },
            logical_state=cast(LogicalState, values.get("logical_state", "0L")),
            custom_bitstring=cast(str | None, values.get("custom_bitstring")),
            initial_basis=cast(Basis | None, values.get("initial_basis")),
            n_stabilizer_rounds=int(values.get("n_stabilizer_rounds", 1)),
            final_basis=cast(Basis, values.get("final_basis", "Z")),
            final_measurement_axes=(
                None
                if values.get("final_measurement_axes") is None
                else cast(
                    tuple[MeasurementAxis, ...],
                    tuple(
                        str(axis).upper()
                        for axis in cast(
                            Sequence[Any], values["final_measurement_axes"]
                        )
                    ),
                )
            ),
            syndrome_mode=cast(
                SyndromeMode, values.get("syndrome_mode", "record_only")
            ),
            postselect_syndrome=cast(tuple[int, int, int], postselect),
            n_shots=int(values.get("n_shots", 1_024)),
            shot_interval_ns=(
                None
                if values.get("shot_interval_ns") is None
                else float(values["shot_interval_ns"])
            ),
            use_simultaneous_cr=_strict_bool(
                values, "use_simultaneous_cr", default=False
            ),
            reset_strategy=cast(
                ResetStrategy, values.get("reset_strategy", "shot_boundary")
            ),
            passive_reset_delay_ns=(
                None
                if values.get("passive_reset_delay_ns") is None
                else float(values["passive_reset_delay_ns"])
            ),
            hadamard_decomposition=cast(
                HadamardDecomposition,
                values.get("hadamard_decomposition", "Y90-X180"),
            ),
            readout_inversion=_strict_bool_mapping(values, "readout_inversion"),
            data_readout_inversion=_strict_bool_mapping(
                values, "data_readout_inversion"
            ),
            dd=DDConfig.from_mapping(cast(Mapping[str, Any] | None, values.get("dd"))),
            cr_overrides=_cr_overrides_from_mapping(values),
            classifier_threshold=(
                None
                if values.get("classifier_threshold") is None
                else float(values["classifier_threshold"])
            ),
            initial_ground_postselection=_strict_bool(
                values,
                "initial_ground_postselection",
                default=False,
            ),
            initial_readout_recovery_delay_ns=(
                None
                if values.get("initial_readout_recovery_delay_ns") is None
                else float(values["initial_readout_recovery_delay_ns"])
            ),
            acknowledge_native_frame_calibration=_strict_bool(
                values,
                "acknowledge_native_frame_calibration",
                default=False,
            ),
            acknowledge_initial_ground_state=_strict_bool(
                values,
                "acknowledge_initial_ground_state",
                default=False,
            ),
            acknowledge_initial_readout_calibration=_strict_bool(
                values,
                "acknowledge_initial_readout_calibration",
                default=False,
            ),
            max_waveform_amplitude=(
                None
                if values.get("max_waveform_amplitude") is None
                else float(values["max_waveform_amplitude"])
            ),
        )


@dataclass(frozen=True)
class QubexConnectionConfig:
    """Describe how to construct a Qubex ``Experiment``."""

    system_id: str
    config_dir: str
    params_dir: str
    calib_note_path: str
    configuration_mode: str = "ge-cr-cr"
    calibration_valid_days: int | None = None

    def __post_init__(self) -> None:
        """Validate required connection fields."""
        for name in ("system_id", "config_dir", "params_dir", "calib_note_path"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be supplied.")
        if self.calibration_valid_days is not None and self.calibration_valid_days < 0:
            raise ValueError("calibration_valid_days must be non-negative.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> QubexConnectionConfig:
        """Construct a connection configuration from decoded JSON."""
        return cls(
            system_id=str(values["system_id"]),
            config_dir=str(values["config_dir"]),
            params_dir=str(values["params_dir"]),
            calib_note_path=str(values["calib_note_path"]),
            configuration_mode=str(values.get("configuration_mode", "ge-cr-cr")),
            calibration_valid_days=(
                None
                if values.get("calibration_valid_days") is None
                else int(values["calibration_valid_days"])
            ),
        )

    def create_experiment(self, qubit_map: Mapping[str, str]) -> Experiment:
        """
        Construct an unconnected Qubex experiment.

        Notes
        -----
        This method loads local configuration and calibration data but does not
        connect to or configure hardware.
        """
        kwargs: dict[str, Any] = {
            "system_id": self.system_id,
            "qubits": [qubit_map[symbol] for symbol in SYMBOLIC_QUBITS],
            "config_dir": self.config_dir,
            "params_dir": self.params_dir,
            "calib_note_path": self.calib_note_path,
            "configuration_mode": self.configuration_mode,
        }
        if self.calibration_valid_days is not None:
            kwargs["calibration_valid_days"] = self.calibration_valid_days
        return Experiment(**kwargs)


@dataclass(frozen=True)
class ExperimentFileConfig:
    """Combine Qubex connection and surface-code settings."""

    qubex: QubexConnectionConfig
    surface_code: SurfaceCodeConfig

    @classmethod
    def load(cls, path: Path | str) -> ExperimentFileConfig:
        """Load and validate a JSON experiment configuration."""
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as stream:
            values = json.load(stream)
        if not isinstance(values, dict):
            raise TypeError("Top-level configuration must be a JSON object.")
        return cls(
            qubex=QubexConnectionConfig.from_mapping(values["qubex"]),
            surface_code=SurfaceCodeConfig.from_mapping(values["surface_code"]),
        )


@dataclass(frozen=True)
class TimelineEvent:
    """Record one semantic event and its logical schedule interval."""

    kind: OperationKind
    name: str
    start_ns: float
    end_ns: float
    qubits: tuple[SymbolicQubit, ...]
    hardware_qubits: tuple[str, ...]
    round_index: int | None = None
    layer: str | None = None
    slot: int | None = None
    edge_id: str | None = None
    context: str | None = None
    acquisition_label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_ns(self) -> float:
        """Return event duration in nanoseconds."""
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class CaptureRef:
    """Map a semantic acquisition label to a Qubex result capture."""

    acquisition_label: str
    symbolic_qubit: SymbolicQubit
    hardware_qubit: str
    readout_label: str
    capture_index: int
    capture_start_ns: float
    capture_duration_ns: float
    role: Literal["initial_ground", "syndrome", "final_data"]
    round_index: int | None


@dataclass(frozen=True)
class ValidationIssue:
    """Describe one static or calibration validation finding."""

    severity: Literal["error", "warning"]
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Collect validation findings."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether the report contains no errors."""
        return all(issue.severity != "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Return only error findings."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Return only warning findings."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def require_ok(self) -> None:
        """Raise one actionable error when validation failed."""
        if self.ok:
            return
        details = "\n".join(
            f"- [{issue.code}] {issue.message}" for issue in self.errors
        )
        raise ValueError(f"Surface-code validation failed:\n{details}")


@dataclass(frozen=True)
class BuiltSurfaceCodeExperiment:
    """Hold the Qubex schedule, semantic timeline, and acquisition map."""

    config: SurfaceCodeConfig
    schedule: PulseSchedule
    timeline: tuple[TimelineEvent, ...]
    captures: tuple[CaptureRef, ...]
    validation: ValidationReport
    measurement_schedule: MeasurementSchedule | None = None
    compiled_time_offset_ns: float = 0.0
    execution_provenance: Literal["adapter_only", "qubex_native_cnot"] = "adapter_only"

    @property
    def round_start_times_ns(self) -> tuple[float, ...]:
        """Return the start time of each stabilizer round."""
        starts: dict[int, float] = {}
        for event in self.timeline:
            if event.round_index is not None and event.kind == "RESET":
                starts.setdefault(event.round_index, event.start_ns)
        return tuple(starts[index] for index in sorted(starts))

    @property
    def layer_start_times_ns(self) -> dict[int, dict[str, float]]:
        """Return each round's L1-L4 start times."""
        starts: dict[int, dict[str, float]] = {}
        for event in self.timeline:
            if event.round_index is None or event.layer is None:
                continue
            starts.setdefault(event.round_index, {}).setdefault(
                event.layer, event.start_ns
            )
        return {
            round_index: {
                layer.name: starts[round_index][layer.name] for layer in LAYERS
            }
            for round_index in sorted(starts)
        }

    @property
    def syndrome_readout_start_times_ns(self) -> tuple[float, ...]:
        """Return the simultaneous ancilla-readout start for each round."""
        starts: dict[int, float] = {}
        for event in self.timeline:
            if (
                event.kind == "READOUT"
                and event.round_index is not None
                and event.qubits[0] in ANCILLA_QUBITS
            ):
                starts.setdefault(event.round_index, event.start_ns)
        return tuple(starts[index] for index in sorted(starts))

    def export_timeline(self, path: Path | str) -> Path:
        """Write semantic events and capture references as JSON."""
        destination = Path(path)
        payload = {
            "execution_mode": "serial_cr",
            "execution_provenance": self.execution_provenance,
            "schedule_duration_ns": self.schedule.duration,
            "compiled_schedule_duration_ns": (
                None
                if self.measurement_schedule is None
                else self.measurement_schedule.pulse_schedule.duration
            ),
            "compiled_time_offset_ns": self.compiled_time_offset_ns,
            "round_start_times_ns": self.round_start_times_ns,
            "layer_start_times_ns": self.layer_start_times_ns,
            "syndrome_readout_start_times_ns": (self.syndrome_readout_start_times_ns),
            "events": [asdict(event) for event in self.timeline],
            "captures": [asdict(capture) for capture in self.captures],
            "validation": {
                "ok": self.validation.ok,
                "issues": [asdict(issue) for issue in self.validation.issues],
            },
        }
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        return destination


@dataclass(frozen=True)
class CalibrationEdgeStatus:
    """Report preflight status for one required directed edge."""

    edge_id: str
    control: str
    target: str
    cr_label: str
    ok: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class CalibrationReport:
    """Collect directed-CR calibration preflight results."""

    edges: tuple[CalibrationEdgeStatus, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether every required edge passed."""
        return all(edge.ok for edge in self.edges)

    def require_ok(self) -> None:
        """Raise an actionable error when a required calibration is unusable."""
        if self.ok:
            return
        messages = []
        for edge in self.edges:
            if edge.ok:
                continue
            messages.append(
                f"{edge.edge_id} ({edge.cr_label}): {'; '.join(edge.issues)}"
            )
        raise ValueError(
            "Directed CR calibration preflight failed:\n- " + "\n- ".join(messages)
        )


@dataclass(frozen=True)
class IdealCheck:
    """Describe one ideal-circuit validation result."""

    passed: bool
    detail: str


@dataclass(frozen=True)
class SurfaceCodeAnalysis:
    """Hold per-shot syndrome, frame, and logical readout arrays."""

    source_shot_indices: NDArray[np.int64]
    initial_ground_bits: NDArray[np.int8]
    initial_ground_accepted_shots: NDArray[np.bool_]
    syndrome_bits: NDArray[np.int8]
    syndrome_accepted_shots: NDArray[np.bool_]
    detection_events: NDArray[np.int8]
    final_data_bits: NDArray[np.int8]
    logical_bits: NDArray[np.int8]
    accepted_shots: NDArray[np.bool_]
    frame_x: NDArray[np.int8]
    frame_z: NDArray[np.int8]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible analysis payload."""
        accepted = self.logical_bits[self.accepted_shots]
        counts = {str(bit): int(np.count_nonzero(accepted == bit)) for bit in (0, 1)}
        defined_total = sum(counts.values())
        classified_total = int(self.accepted_shots.size)
        herald_count = int(np.count_nonzero(self.initial_ground_accepted_shots))
        syndrome_after_herald_count = int(
            np.count_nonzero(
                self.initial_ground_accepted_shots & self.syndrome_accepted_shots
            )
        )
        return {
            "source_shot_indices": self.source_shot_indices.tolist(),
            "initial_ground_bits": self.initial_ground_bits.tolist(),
            "initial_ground_accepted_shots": (
                self.initial_ground_accepted_shots.tolist()
            ),
            "syndrome_bits": self.syndrome_bits.tolist(),
            "syndrome_accepted_shots": self.syndrome_accepted_shots.tolist(),
            "detection_events": self.detection_events.tolist(),
            "final_data_bits": self.final_data_bits.tolist(),
            "logical_bits": self.logical_bits.tolist(),
            "accepted_shots": self.accepted_shots.tolist(),
            "frame_x": self.frame_x.tolist(),
            "frame_z": self.frame_z.tolist(),
            "frame_state_after_round": {
                "x": self.frame_x.tolist(),
                "z": self.frame_z.tolist(),
            },
            "logical_counts": counts,
            "logical_probabilities": {
                key: (value / defined_total if defined_total else 0.0)
                for key, value in counts.items()
            },
            "selection": {
                "classified_shots": classified_total,
                "initial_ground_accepted_shots": herald_count,
                "combined_accepted_shots": syndrome_after_herald_count,
                "initial_ground_acceptance": (
                    herald_count / classified_total if classified_total else 0.0
                ),
                "syndrome_acceptance_given_initial_ground": (
                    syndrome_after_herald_count / herald_count if herald_count else 0.0
                ),
                "combined_acceptance": (
                    syndrome_after_herald_count / classified_total
                    if classified_total
                    else 0.0
                ),
            },
        }


def _as_binary_array(
    values: NDArray[Any] | Sequence[Any],
    *,
    name: str,
) -> NDArray[np.int8]:
    """Convert and validate an array containing only 0/1 values."""
    array = np.asarray(values)
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only 0/1 values.")
    return array.astype(np.int8, copy=False)


def analyze_bit_arrays(
    *,
    config: SurfaceCodeConfig,
    syndrome_bits: NDArray[Any] | Sequence[Any],
    final_data_bits: NDArray[Any] | Sequence[Any],
    initial_ground_bits: NDArray[Any] | Sequence[Any] | None = None,
    source_shot_indices: NDArray[Any] | Sequence[Any] | None = None,
) -> SurfaceCodeAnalysis:
    """
    Analyze classified syndrome and final-data bit arrays.

    Parameters
    ----------
    config
        Experiment and analysis configuration.
    syndrome_bits
        Shape ``(shots, rounds, 3)`` in ``A1,A2,A3`` order.
    final_data_bits
        Shape ``(shots, 4)`` in ``D1,D2,D3,D4`` order.
    initial_ground_bits
        Optional shape ``(shots, 7)`` pre-sequence readout in
        ``D1,D2,D3,D4,A1,A2,A3`` order. Required when initial ground-state
        post-selection is enabled.

    Returns
    -------
    SurfaceCodeAnalysis
        Inversion-corrected syndrome, detection events, pure-error frame, and
        logical bit.
    """
    syndrome = _as_binary_array(syndrome_bits, name="syndrome_bits").copy()
    data = _as_binary_array(final_data_bits, name="final_data_bits").copy()
    if initial_ground_bits is None:
        if config.initial_ground_postselection:
            raise ValueError(
                "initial_ground_bits are required when "
                "initial_ground_postselection is enabled."
            )
        initial = np.empty((config.n_shots, 0), dtype=np.int8)
    else:
        initial = _as_binary_array(
            initial_ground_bits,
            name="initial_ground_bits",
        ).copy()
        if not config.initial_ground_postselection:
            raise ValueError(
                "initial_ground_bits were supplied while "
                "initial_ground_postselection is disabled."
            )
    expected_syndrome_shape = (
        config.n_shots,
        config.n_stabilizer_rounds,
        len(ANCILLA_QUBITS),
    )
    expected_data_shape = (config.n_shots, len(DATA_QUBITS))
    expected_initial_shape = (config.n_shots, len(SYMBOLIC_QUBITS))
    if syndrome.shape != expected_syndrome_shape:
        raise ValueError(
            f"syndrome_bits shape must be {expected_syndrome_shape}, "
            f"received {syndrome.shape}."
        )
    if data.shape != expected_data_shape:
        raise ValueError(
            f"final_data_bits shape must be {expected_data_shape}, "
            f"received {data.shape}."
        )
    if config.initial_ground_postselection and initial.shape != expected_initial_shape:
        raise ValueError(
            f"initial_ground_bits shape must be {expected_initial_shape}, "
            f"received {initial.shape}."
        )
    if source_shot_indices is None:
        source_indices = np.arange(config.n_shots, dtype=np.int64)
    else:
        source_indices = np.asarray(source_shot_indices, dtype=np.int64)
        if source_indices.shape != (config.n_shots,):
            raise ValueError(
                "source_shot_indices shape must be "
                f"{(config.n_shots,)}, received {source_indices.shape}."
            )
        if (
            np.any(source_indices < 0)
            or np.unique(source_indices).size != config.n_shots
        ):
            raise ValueError("source_shot_indices must be unique and non-negative.")

    syndrome_inversion = np.array(
        [int(config.readout_inversion.get(symbol, False)) for symbol in ANCILLA_QUBITS],
        dtype=np.int8,
    )
    data_inversion = np.array(
        [
            int(config.data_readout_inversion.get(symbol, False))
            for symbol in DATA_QUBITS
        ],
        dtype=np.int8,
    )
    syndrome ^= syndrome_inversion[np.newaxis, np.newaxis, :]
    data ^= data_inversion[np.newaxis, :]
    if config.initial_ground_postselection:
        initial_inversion = np.concatenate((data_inversion, syndrome_inversion))
        initial ^= initial_inversion[np.newaxis, :]

    detection = np.bitwise_xor(syndrome[:, 1:, :], syndrome[:, :-1, :])
    logical_basis = config.logical_measurement_basis
    if logical_basis == "Z":
        logical = np.bitwise_xor(data[:, 0], data[:, 2])
    elif logical_basis == "X":
        logical = np.bitwise_xor(data[:, 0], data[:, 1])
    else:
        logical = np.full(config.n_shots, -1, dtype=np.int8)

    initial_accepted = np.ones(config.n_shots, dtype=np.bool_)
    if config.initial_ground_postselection:
        initial_accepted = np.all(initial == 0, axis=1)
    syndrome_accepted = np.ones(config.n_shots, dtype=np.bool_)
    if config.syndrome_mode == "postselect":
        desired = np.asarray(config.postselect_syndrome, dtype=np.int8)
        syndrome_accepted = np.all(
            syndrome == desired[np.newaxis, np.newaxis, :],
            axis=(1, 2),
        )
    accepted = initial_accepted & syndrome_accepted

    frame_x = np.zeros(
        (config.n_shots, config.n_stabilizer_rounds, len(DATA_QUBITS)),
        dtype=np.int8,
    )
    frame_z = np.zeros_like(frame_x)
    if config.syndrome_mode == "pauli_frame":
        frame_x[:, :, DATA_QUBITS.index("D2")] = syndrome[
            :, :, ANCILLA_QUBITS.index("A1")
        ]
        frame_z[:, :, DATA_QUBITS.index("D4")] = syndrome[
            :, :, ANCILLA_QUBITS.index("A2")
        ]
        frame_x[:, :, DATA_QUBITS.index("D4")] = syndrome[
            :, :, ANCILLA_QUBITS.index("A3")
        ]

    return SurfaceCodeAnalysis(
        source_shot_indices=source_indices,
        initial_ground_bits=initial,
        initial_ground_accepted_shots=initial_accepted,
        syndrome_bits=syndrome,
        syndrome_accepted_shots=syndrome_accepted,
        detection_events=detection,
        final_data_bits=data,
        logical_bits=logical.astype(np.int8, copy=False),
        accepted_shots=accepted,
        frame_x=frame_x,
        frame_z=frame_z,
    )


def _single_qubit_state(bit: int, basis: Basis) -> NDArray[np.complex128]:
    """Return a Z- or X-basis single-qubit state."""
    if basis == "Z":
        return np.array([1, 0] if bit == 0 else [0, 1], dtype=np.complex128)
    sign = 1 if bit == 0 else -1
    return np.array([1, sign], dtype=np.complex128) / np.sqrt(2)


def _product_state(
    bitstring: str, basis: Basis, ancillas: int = 0
) -> NDArray[np.complex128]:
    """Return a data product state followed by ground-state ancillas."""
    states = [_single_qubit_state(int(bit), basis) for bit in bitstring]
    states.extend(np.array([1, 0], dtype=np.complex128) for _ in range(ancillas))
    result = states[0]
    for state in states[1:]:
        result = np.kron(result, state)
    return result


def _apply_1q(
    state: NDArray[np.complex128],
    gate: NDArray[np.complex128],
    qubit: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    """Apply a single-qubit gate with qubit zero as the most-significant bit."""
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    updated = gate @ moved
    return np.moveaxis(updated.reshape((2,) + (2,) * (n_qubits - 1)), 0, qubit).reshape(
        -1
    )


def _apply_cnot(
    state: NDArray[np.complex128],
    control: int,
    target: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    """Apply an ideal CNOT with qubit zero as the most-significant bit."""
    result = np.zeros_like(state)
    control_mask = 1 << (n_qubits - 1 - control)
    target_mask = 1 << (n_qubits - 1 - target)
    for index, amplitude in enumerate(state):
        destination = index ^ target_mask if index & control_mask else index
        result[destination] += amplitude
    return result


def _simulate_serial_round_from_data(
    data_state: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Simulate one ideal round after resetting all three ancillas to ground."""
    if data_state.shape != (16,):
        raise ValueError("data_state must be a four-qubit statevector.")
    state = np.kron(data_state, _product_state("000", "Z"))
    h_gate = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    indices = {symbol: index for index, symbol in enumerate(SYMBOLIC_QUBITS)}
    for operation in serial_round_operations():
        if operation.kind == "H":
            state = _apply_1q(state, h_gate, indices[operation.qubits[0]], 7)
        else:
            state = _apply_cnot(
                state,
                indices[operation.qubits[0]],
                indices[operation.qubits[1]],
                7,
            )
    return state


def _simulate_serial_round(
    bitstring: str,
    basis: Basis,
) -> NDArray[np.complex128]:
    """Simulate the fixed seven-qubit ideal round for a product-state input."""
    return _simulate_serial_round_from_data(_product_state(bitstring, basis))


def _marginal_probability(
    state: NDArray[np.complex128],
    assignments: Mapping[int, int],
    n_qubits: int,
) -> float:
    """Return probability of selected computational-basis assignments."""
    probability = 0.0
    for index, amplitude in enumerate(state):
        matches = all(
            ((index >> (n_qubits - 1 - qubit)) & 1) == bit
            for qubit, bit in assignments.items()
        )
        if matches:
            probability += float(abs(amplitude) ** 2)
    return probability


def _condition_data_on_ancillas(
    state: NDArray[np.complex128],
    outcomes: tuple[int, int, int],
) -> NDArray[np.complex128] | None:
    """Return the normalized data state conditioned on one ancilla outcome."""
    if state.shape != (128,):
        raise ValueError("state must be a seven-qubit statevector.")
    ancilla_index = (outcomes[0] << 2) | (outcomes[1] << 1) | outcomes[2]
    conditioned = state.reshape(16, 8)[:, ancilla_index].copy()
    norm = float(np.linalg.norm(conditioned))
    if np.isclose(norm, 0.0):
        return None
    return conditioned / norm


_PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _basis_ket(bitstring: str) -> NDArray[np.complex128]:
    """Return a four-qubit computational-basis ket."""
    return _product_state(bitstring, "Z")


def _states_equal_up_to_phase(
    state_a: NDArray[np.complex128],
    state_b: NDArray[np.complex128],
) -> bool:
    """Return whether normalized pure states agree up to global phase."""
    return bool(np.isclose(abs(np.vdot(state_a, state_b)), 1.0, atol=1e-10))


def run_ideal_validations() -> dict[str, IdealCheck]:
    """
    Run dependency-free ideal checks for the fixed circuit.

    Returns
    -------
    dict[str, IdealCheck]
        Z/X parity, canonical encoding, repeated projection, and error
        detection checks.
    """
    z_failures: list[str] = []
    x_failures: list[str] = []
    for value in range(16):
        bitstring = f"{value:04b}"
        z_state = _simulate_serial_round(bitstring, "Z")
        expected_a1 = int(bitstring[0]) ^ int(bitstring[1])
        expected_a3 = int(bitstring[2]) ^ int(bitstring[3])
        z_probability = _marginal_probability(
            z_state, {4: expected_a1, 6: expected_a3}, 7
        )
        if not np.isclose(z_probability, 1.0, atol=1e-10):
            z_failures.append(bitstring)

        x_state = _simulate_serial_round(bitstring, "X")
        expected_a2 = sum(int(bit) for bit in bitstring) % 2
        x_probability = _marginal_probability(x_state, {5: expected_a2}, 7)
        if not np.isclose(x_probability, 1.0, atol=1e-10):
            x_failures.append(bitstring)

    zero_l = (_basis_ket("0000") + _basis_ket("1111")) / np.sqrt(2)
    one_l = (_basis_ket("0011") + _basis_ket("1100")) / np.sqrt(2)
    expected_states = {
        "0L": zero_l,
        "1L": one_l,
        "+L": (zero_l + one_l) / np.sqrt(2),
        "-L": (zero_l - one_l) / np.sqrt(2),
    }
    encoding_failures: list[str] = []
    for logical_state, expected in expected_states.items():
        bitstring, basis_string = canonical_seed(logical_state)
        measured = _simulate_serial_round(bitstring, cast(Basis, basis_string))
        conditioned = _condition_data_on_ancillas(measured, (0, 0, 0))
        if conditioned is None or not _states_equal_up_to_phase(conditioned, expected):
            encoding_failures.append(logical_state)

    repeat_failures: list[str] = []
    seed = _product_state("0101", "X")
    first_round = _simulate_serial_round_from_data(seed)
    for outcomes in ((a1, a2, a3) for a1 in (0, 1) for a2 in (0, 1) for a3 in (0, 1)):
        once = _condition_data_on_ancillas(first_round, outcomes)
        if once is None:
            continue
        repeated_state = _simulate_serial_round_from_data(once)
        repeated_probability = _marginal_probability(
            repeated_state,
            {4 + index: bit for index, bit in enumerate(outcomes)},
            7,
        )
        if not np.isclose(repeated_probability, 1.0, atol=1e-10):
            repeat_failures.append("".join(map(str, outcomes)))
            continue
        twice = _condition_data_on_ancillas(repeated_state, outcomes)
        if twice is None or not _states_equal_up_to_phase(once, twice):
            repeat_failures.append("".join(map(str, outcomes)))

    expected_error_events = {
        "X_D1": (1, 0, 0),
        "X_D2": (1, 0, 0),
        "X_D3": (0, 0, 1),
        "X_D4": (0, 0, 1),
        "Z_D1": (0, 1, 0),
        "Z_D2": (0, 1, 0),
        "Z_D3": (0, 1, 0),
        "Z_D4": (0, 1, 0),
    }
    error_failures: list[str] = []
    for axis, gate in {"X": _PAULI_X, "Z": _PAULI_Z}.items():
        for data_index, symbol in enumerate(DATA_QUBITS):
            errored = _apply_1q(zero_l, gate, data_index, len(DATA_QUBITS))
            measured = _simulate_serial_round_from_data(errored)
            expected = expected_error_events[f"{axis}_{symbol}"]
            probability = _marginal_probability(
                measured,
                {4 + index: bit for index, bit in enumerate(expected)},
                7,
            )
            if not np.isclose(probability, 1.0, atol=1e-10):
                error_failures.append(f"{axis}_{symbol}")

    return {
        "z_parity_all_16": IdealCheck(
            not z_failures,
            "passed" if not z_failures else f"failed seeds: {z_failures}",
        ),
        "x_parity_all_16": IdealCheck(
            not x_failures,
            "passed" if not x_failures else f"failed seeds: {x_failures}",
        ),
        "canonical_encoding": IdealCheck(
            not encoding_failures,
            (
                "passed"
                if not encoding_failures
                else f"failed states: {encoding_failures}"
            ),
        ),
        "repeated_round_stability": IdealCheck(
            not repeat_failures,
            (
                "passed"
                if not repeat_failures
                else f"failed syndromes: {repeat_failures}"
            ),
        ),
        "single_pauli_error_events": IdealCheck(
            not error_failures,
            (
                "passed"
                if not error_failures
                else f"failed injections: {error_failures}"
            ),
        ),
    }


class SurfaceCodePulseAdapter(Protocol):
    """Define the device-facing operations required by the circuit builder."""

    @property
    def sampling_period_ns(self) -> float:
        """Return the control sample period in nanoseconds."""
        ...

    @property
    def capture_alignment_ns(self) -> float:
        """Return the required capture-start alignment in nanoseconds."""
        ...

    @property
    def initial_labels(self) -> list[str]:
        """Return physical qubit-drive labels used by the schedule."""
        ...

    def qubit_label(self, symbol: SymbolicQubit) -> str:
        """Resolve a symbolic qubit."""
        ...

    def xpi(self, symbol: DataQubit) -> Waveform:
        """Return a calibrated pi pulse."""
        ...

    def hadamard(self, symbol: DataQubit) -> Waveform:
        """Return a calibrated Hadamard decomposition."""
        ...

    def measurement_rotation(
        self,
        symbol: DataQubit,
        axis: MeasurementAxis,
    ) -> Waveform | None:
        """Return the pre-rotation for a final Pauli-basis measurement."""
        ...

    def cx_cr(
        self,
        edge: DirectedEdge,
        *,
        context: str,
    ) -> PulseSchedule:
        """Return a logical directed CX macro."""
        ...

    def readout(
        self,
        symbol: SymbolicQubit,
    ) -> tuple[str, Waveform, float, float]:
        """Return readout label, waveform, pre-margin, and capture duration."""
        ...

    def dd_waveform(
        self,
        symbol: DataQubit,
        *,
        duration_ns: float,
        config: DDConfig,
    ) -> Waveform:
        """Return an identity DD waveform fitting the requested window."""
        ...

    def reset_schedule(
        self,
        round_index: int,
    ) -> PulseSchedule:
        """Return a user-supplied calibrated ancilla reset schedule."""
        ...


ResetScheduleFactory = Callable[
    [Experiment, tuple[str, str, str], int],
    PulseSchedule,
]


def _is_grid_aligned(value: float, grid: float) -> bool:
    """Return whether a time lies on a positive sample grid."""
    if grid <= 0:
        return False
    return bool(np.isclose(value / grid, round(value / grid), atol=1e-8))


def _ceil_to_grid(value: float, grid: float) -> float:
    """Round a non-negative time upward to the specified grid."""
    if value < 0 or grid <= 0:
        raise ValueError("Grid alignment requires non-negative time and positive grid.")
    return math.ceil((value / grid) - 1e-12) * grid


def _distribute_gap_samples(
    total_samples: int,
    n_gaps: int,
) -> list[int]:
    """Distribute integer idle samples symmetrically across pulse gaps."""
    if total_samples < 0 or n_gaps < 1:
        raise ValueError("Invalid DD gap dimensions.")
    gaps = [0] * n_gaps
    left = 0
    right = n_gaps - 1
    while total_samples:
        gaps[left] += 1
        total_samples -= 1
        if total_samples == 0:
            break
        if right != left:
            gaps[right] += 1
            total_samples -= 1
        left += 1
        right -= 1
        if left > right:
            left = 0
            right = n_gaps - 1
    return gaps


class QubexPulseAdapter:
    """
    Map abstract surface-code operations to inspected Qubex 1.5 APIs.

    The adapter validates a direct ``data-ancilla`` CR calibration before calling
    Qubex's complete ``Experiment.cnot`` macro. Raw ZX90 schedules are never
    accepted as logical CX operations.
    """

    def __init__(
        self,
        exp: Experiment,
        config: SurfaceCodeConfig,
        *,
        reset_schedule_factory: ResetScheduleFactory | None = None,
    ) -> None:
        """Initialize the adapter without connecting to hardware."""
        self.exp = exp
        self.config = config
        self.reset_schedule_factory = reset_schedule_factory

    @property
    def sampling_period_ns(self) -> float:
        """Return Qubex's backend-specific control sample period."""
        return float(self.exp.ctx.measurement.sampling_period)

    @property
    def capture_alignment_ns(self) -> float:
        """Return the backend's capture word or sample-grid duration."""
        profile = self.exp.ctx.measurement.constraint_profile
        if profile.enforce_word_alignment:
            word = profile.word_duration_ns
            if word is None:
                raise ValueError("Backend requires word alignment without a word size.")
            return float(word)
        return self.sampling_period_ns

    @property
    def initial_labels(self) -> list[str]:
        """Return all seven mapped qubit-drive labels."""
        return [self.config.qubit_map[symbol] for symbol in SYMBOLIC_QUBITS]

    def qubit_label(self, symbol: SymbolicQubit) -> str:
        """Resolve one symbolic qubit to its physical Qubex label."""
        return self.config.qubit_map[symbol]

    def xpi(self, symbol: DataQubit) -> Waveform:
        """Return Qubex's calibrated X180 pulse."""
        return self.exp.x180(self.qubit_label(symbol))

    def hadamard(self, symbol: DataQubit) -> Waveform:
        """
        Return Qubex's calibrated Hadamard decomposition.

        The default surface-code configuration uses ``Y90-X180`` because
        Qubex's source marks the virtual-Z decomposition as needing additional
        phase correction for CR targets.
        """
        return self.exp.hadamard(
            self.qubit_label(symbol),
            decomposition=self.config.hadamard_decomposition,
        )

    def measurement_rotation(
        self,
        symbol: DataQubit,
        axis: MeasurementAxis,
    ) -> Waveform | None:
        """
        Return the Qubex-standard rotation from a Pauli basis to Z readout.

        Qubex's existing multi-qubit tomography uses `Y90m` for X and `X90`
        for Y. The Y-axis sign must be verified with the device frame
        convention before interpreting the imaginary density matrix.
        """
        qubit = self.qubit_label(symbol)
        if axis == "X":
            return self.exp.y90m(qubit)
        if axis == "Y":
            return self.exp.x90(qubit)
        if axis == "Z":
            return None
        raise ValueError(f"Unsupported measurement axis {axis!r}.")

    def _direct_cr_parameter(
        self,
        edge: DirectedEdge,
    ) -> tuple[str, Mapping[str, Any]]:
        """Return a fresh direct calibration or raise before reverse fallback."""
        control = self.qubit_label(edge.control)
        target = self.qubit_label(edge.target)
        cr_label = f"{control}-{target}"
        if cr_label not in self.exp.calib_note.cr_params:
            raise ValueError(
                f"{edge.edge_id} requires direct CR calibration {cr_label}; "
                "reverse-direction fallback is forbidden."
            )
        parameter = self.exp.calib_note.get_cr_param(
            cr_label,
            valid_days=self.exp.ctx.calibration_valid_days,
        )
        if parameter is None:
            raise ValueError(
                f"Direct CR calibration {cr_label} is missing or outside "
                "calibration_valid_days."
            )
        return cr_label, parameter

    def cx_cr(
        self,
        edge: DirectedEdge,
        *,
        context: str,
    ) -> PulseSchedule:
        """Build a complete logical CX with physical direction data to ancilla."""
        control = self.qubit_label(edge.control)
        target = self.qubit_label(edge.target)
        override = self.config.cr_override(edge.edge_id, context)
        cr_label, _ = self._direct_cr_parameter(edge)
        echo = override.get("echo")
        if echo is not None and not isinstance(echo, bool):
            raise TypeError(f"CR override {edge.edge_id}.echo must be bool.")

        def optional_float(name: str) -> float | None:
            value = override.get(name)
            if value is None:
                return None
            if isinstance(value, bool):
                raise TypeError(f"CR override {edge.edge_id}.{name} must be numeric.")
            return float(value)

        zx90 = self.exp.zx90(
            control,
            target,
            cr_duration=optional_float("cr_duration"),
            cr_ramptime=optional_float("cr_ramptime"),
            cr_amplitude=optional_float("cr_amplitude"),
            cr_phase=optional_float("cr_phase"),
            cr_beta=optional_float("cr_beta"),
            cancel_amplitude=optional_float("cancel_amplitude"),
            cancel_phase=optional_float("cancel_phase"),
            cancel_beta=optional_float("cancel_beta"),
            rotary_amplitude=optional_float("rotary_amplitude"),
            echo=echo,
            x180_margin=optional_float("x180_margin"),
        )
        schedule = self.exp.cnot(
            control,
            target,
            zx90=zx90,
            only_low_to_high=False,
        )
        reverse_label = f"{target}-{control}"
        if cr_label not in schedule.labels:
            raise ValueError(
                f"Logical CX for {edge.edge_id} did not use required {cr_label}."
            )
        if reverse_label in schedule.labels and reverse_label != cr_label:
            raise ValueError(
                f"Logical CX for {edge.edge_id} used forbidden reverse CR "
                f"{reverse_label}."
            )
        return schedule

    def readout(
        self,
        symbol: SymbolicQubit,
    ) -> tuple[str, Waveform, float, float]:
        """Return a configured Qubex readout waveform and capture timing."""
        qubit = self.qubit_label(symbol)
        read_label = self.exp.ctx.resolve_read_label(qubit)
        waveform = self.exp.pulse.readout(read_label)
        pre_margin = float(self.exp.readout_pre_margin)
        capture_duration = float(waveform.duration) - pre_margin
        if capture_duration <= 0:
            raise ValueError(f"Readout waveform for {symbol} has no capture window.")
        return (
            read_label,
            waveform,
            pre_margin,
            capture_duration,
        )

    def dd_waveform(
        self,
        symbol: DataQubit,
        *,
        duration_ns: float,
        config: DDConfig,
    ) -> Waveform:
        """Fit the configured identity DD word to a calibrated idle window."""
        axes = config.pulse_axes * config.repeat_count
        qubit = self.qubit_label(symbol)
        pulses = [
            self.exp.x180(qubit) if axis == "X" else self.exp.y180(qubit)
            for axis in axes
        ]
        grid = self.sampling_period_ns
        total_samples = round(duration_ns / grid)
        if not _is_grid_aligned(duration_ns, grid):
            raise ValueError("DD window is not on the control sample grid.")
        pulse_samples = [round(pulse.duration / grid) for pulse in pulses]
        if any(not _is_grid_aligned(pulse.duration, grid) for pulse in pulses):
            raise ValueError(f"DD pulse for {symbol} is off the sample grid.")
        idle_samples = total_samples - sum(pulse_samples)
        if idle_samples < 0:
            raise ValueError(
                f"DD pulses for {symbol} exceed the {duration_ns} ns window."
            )
        gaps = _distribute_gap_samples(idle_samples, len(pulses) + 1)
        elements: list[Waveform] = []
        for gap, pulse in zip(gaps, pulses, strict=False):
            if gap:
                elements.append(Blank(gap * grid))
            elements.append(pulse)
        if gaps[-1]:
            elements.append(Blank(gaps[-1] * grid))
        result = PulseArray(elements)
        if not np.isclose(result.duration, duration_ns):
            raise ValueError(
                f"DD construction for {symbol} has duration {result.duration}, "
                f"expected {duration_ns}."
            )
        return result

    def reset_schedule(self, round_index: int) -> PulseSchedule:
        """Call the user-provided calibrated reset factory."""
        if self.reset_schedule_factory is None:
            raise NotImplementedError(
                "Qubex has no built-in qubit active reset; provide "
                "reset_schedule_factory for reset_strategy='custom'."
            )
        ancillas = cast(
            tuple[str, str, str],
            tuple(self.qubit_label(symbol) for symbol in ANCILLA_QUBITS),
        )
        return self.reset_schedule_factory(self.exp, ancillas, round_index)


def preflight_directed_cr_calibrations(
    adapter: QubexPulseAdapter,
    *,
    construct_macros: bool = True,
) -> CalibrationReport:
    """
    Validate all eight directed CR calibrations before schedule construction.

    The check covers target registration, direction, freshness, required finite
    values, pulse-shape duration, and optionally logical-CX construction.
    """
    statuses: list[CalibrationEdgeStatus] = []
    for layer in LAYERS:
        for slot, edge in enumerate(layer.edges, start=1):
            control = adapter.qubit_label(edge.control)
            target = adapter.qubit_label(edge.target)
            cr_label = f"{control}-{target}"
            issues: list[str] = []
            raw = adapter.exp.calib_note.cr_params.get(cr_label)
            if raw is None:
                issues.append("direct calibration is absent")

            target_entry = adapter.exp.ctx.targets.get(cr_label)
            if target_entry is None:
                issues.append("direct CR target is not registered")
            else:
                try:
                    resolved = adapter.exp.ctx.resolve_2q_qubits(cr_label)
                except (
                    CalibrationMissingError,
                    KeyError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as exc:
                    issues.append(f"cannot resolve registered direction: {exc}")
                else:
                    if tuple(resolved) != (control, target):
                        issues.append(
                            f"registered direction is {tuple(resolved)}, "
                            f"expected {(control, target)}"
                        )

            fresh = adapter.exp.calib_note.get_cr_param(
                cr_label,
                valid_days=adapter.exp.ctx.calibration_valid_days,
            )
            if raw is not None and fresh is None:
                issues.append("calibration is outside calibration_valid_days")
            parameter = fresh if fresh is not None else raw
            if parameter is not None:
                numeric_keys = (
                    "duration",
                    "ramptime",
                    "cr_amplitude",
                    "cr_phase",
                    "cr_beta",
                    "cancel_amplitude",
                    "cancel_phase",
                    "cancel_beta",
                    "rotary_amplitude",
                )
                for name in numeric_keys:
                    value = parameter.get(name)
                    if value is None:
                        issues.append(f"{name} is missing")
                    elif not math.isfinite(float(value)):
                        issues.append(f"{name} is not finite")
                duration = parameter.get("duration")
                ramptime = parameter.get("ramptime")
                if duration is not None and ramptime is not None:
                    if float(duration) <= 2 * float(ramptime):
                        issues.append("duration must be greater than 2 * ramptime")

            if not issues and construct_macros:
                context = f"serial:{layer.name}:S{slot}"
                try:
                    macro = adapter.cx_cr(edge, context=context)
                except (
                    CalibrationMissingError,
                    KeyError,
                    NotImplementedError,
                    TypeError,
                    ValueError,
                ) as exc:
                    issues.append(f"logical CX construction failed: {exc}")
                else:
                    if macro.duration <= 0:
                        issues.append("logical CX has non-positive duration")

            statuses.append(
                CalibrationEdgeStatus(
                    edge_id=edge.edge_id,
                    control=control,
                    target=target,
                    cr_label=cr_label,
                    ok=not issues,
                    issues=tuple(issues),
                )
            )

    warnings = [
        "Qubex native CNOT phase shifts are tracked per schedule label only. "
        "Validate each serial context by truth table/process characterization "
        "before hardware execution."
    ]
    return CalibrationReport(edges=tuple(statuses), warnings=tuple(warnings))


class _SurfaceCodeSequenceBuilder:
    """Build a fully explicit fixed-timeline Qubex pulse schedule."""

    def __init__(
        self,
        adapter: SurfaceCodePulseAdapter,
        config: SurfaceCodeConfig,
    ) -> None:
        """Initialize builder state."""
        self.adapter = adapter
        self.config = config
        self.schedule = PulseSchedule(adapter.initial_labels)
        self.timeline: list[TimelineEvent] = []
        self.captures: list[CaptureRef] = []
        self._capture_count: dict[SymbolicQubit, int] = dict.fromkeys(
            SYMBOLIC_QUBITS, 0
        )

    def _now(self) -> float:
        """Synchronize all labels and return current global time."""
        self.schedule.barrier()
        return float(self.schedule.duration)

    def _hardware(self, symbols: Sequence[SymbolicQubit]) -> tuple[str, ...]:
        """Resolve multiple symbolic labels."""
        return tuple(self.adapter.qubit_label(symbol) for symbol in symbols)

    def _add_parallel_1q(
        self,
        kind: Literal["XPI", "H"],
        symbols: Sequence[DataQubit],
        *,
        round_index: int | None,
        layer: str | None = None,
        name: str,
    ) -> None:
        """Add calibrated single-qubit pulses at a common start."""
        if not symbols:
            return
        start = self._now()
        pulses: list[tuple[DataQubit, Waveform]] = []
        for symbol in symbols:
            pulse = (
                self.adapter.xpi(symbol)
                if kind == "XPI"
                else self.adapter.hadamard(symbol)
            )
            self.schedule.add(self.adapter.qubit_label(symbol), pulse)
            pulses.append((symbol, pulse))
        self.schedule.barrier()
        for symbol, pulse in pulses:
            self.timeline.append(
                TimelineEvent(
                    kind=kind,
                    name=name,
                    start_ns=start,
                    end_ns=start + pulse.duration,
                    qubits=(symbol,),
                    hardware_qubits=(self.adapter.qubit_label(symbol),),
                    round_index=round_index,
                    layer=layer,
                )
            )

    def _add_final_measurement_rotations(self) -> None:
        """Rotate each data qubit from its requested Pauli basis to Z."""
        start = self._now()
        pulses: list[tuple[DataQubit, MeasurementAxis, Waveform]] = []
        for symbol, axis in zip(
            DATA_QUBITS,
            self.config.resolved_final_measurement_axes,
            strict=True,
        ):
            pulse = self.adapter.measurement_rotation(symbol, axis)
            if pulse is None:
                continue
            self.schedule.add(self.adapter.qubit_label(symbol), pulse)
            pulses.append((symbol, axis, pulse))
        if not pulses:
            return
        self.schedule.barrier()
        for symbol, axis, pulse in pulses:
            self.timeline.append(
                TimelineEvent(
                    kind="BASIS_ROTATION",
                    name=f"final_{axis.lower()}_basis_rotation_{symbol}",
                    start_ns=start,
                    end_ns=start + pulse.duration,
                    qubits=(symbol,),
                    hardware_qubits=(self.adapter.qubit_label(symbol),),
                    metadata={"axis": axis},
                )
            )

    def _add_global_wait(
        self,
        duration_ns: float,
        *,
        kind: Literal["DELAY", "RESET"],
        name: str,
        round_index: int | None,
        dd_enabled: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Advance every physical lane, optionally applying data DD."""
        if duration_ns < 0:
            raise ValueError("Delay duration must be non-negative.")
        start = self._now()
        if duration_ns > 0:
            for symbol in DATA_QUBITS:
                waveform: Waveform
                if dd_enabled and self.config.dd.enabled:
                    waveform = self.adapter.dd_waveform(
                        symbol,
                        duration_ns=duration_ns,
                        config=self.config.dd,
                    )
                    self.timeline.append(
                        TimelineEvent(
                            kind="DD",
                            name=f"{name}_dd_{symbol}",
                            start_ns=start,
                            end_ns=start + duration_ns,
                            qubits=(symbol,),
                            hardware_qubits=(self.adapter.qubit_label(symbol),),
                            round_index=round_index,
                            metadata={"pulse_axes": self.config.dd.pulse_axes},
                        )
                    )
                else:
                    waveform = Blank(duration_ns)
                self.schedule.add(self.adapter.qubit_label(symbol), waveform)
            for symbol in ANCILLA_QUBITS:
                self.schedule.add(self.adapter.qubit_label(symbol), Blank(duration_ns))
            self.schedule.barrier()
        self.timeline.append(
            TimelineEvent(
                kind=kind,
                name=name,
                start_ns=start,
                end_ns=start + duration_ns,
                qubits=ANCILLA_QUBITS if kind == "RESET" else SYMBOLIC_QUBITS,
                hardware_qubits=self._hardware(
                    ANCILLA_QUBITS if kind == "RESET" else SYMBOLIC_QUBITS
                ),
                round_index=round_index,
                metadata={} if metadata is None else metadata,
            )
        )

    def _prepare_seed(self) -> None:
        """Prepare the configured four-data-qubit product seed."""
        bitstring, basis = self.config.resolved_seed
        excited: list[DataQubit] = [
            symbol
            for symbol, bit in zip(DATA_QUBITS, bitstring, strict=True)
            if bit == "1"
        ]
        self._add_parallel_1q(
            "XPI",
            excited,
            round_index=None,
            name=f"prepare_{bitstring}",
        )
        if basis == "X":
            self._add_parallel_1q(
                "H",
                DATA_QUBITS,
                round_index=None,
                name="prepare_x_basis",
            )

    def _reset_ancillas(self, round_index: int) -> None:
        """Apply the explicitly selected ancilla-reset strategy."""
        if self.config.reset_strategy == "shot_boundary":
            self._add_global_wait(
                0.0,
                kind="RESET",
                name="shot_boundary_ground_assumption",
                round_index=round_index,
                dd_enabled=False,
                metadata={
                    "warning": (
                        "No qubit reset pulse is emitted; ground state is assumed "
                        "from shot repetition."
                    )
                },
            )
            return
        if self.config.reset_strategy == "passive":
            if self.config.passive_reset_delay_ns is None:
                raise RuntimeError(
                    "Validated passive reset delay is unexpectedly missing."
                )
            self._add_global_wait(
                self.config.passive_reset_delay_ns,
                kind="RESET",
                name="passive_ancilla_reset_wait",
                round_index=round_index,
                dd_enabled=self.config.dd.during_reset,
                metadata={
                    "strategy": "passive",
                    "requires_calibration": True,
                },
            )
            return

        start = self._now()
        reset = self.adapter.reset_schedule(round_index)
        forbidden_labels = {self.adapter.qubit_label(symbol) for symbol in DATA_QUBITS}
        for edge in REQUIRED_EDGES:
            control, target = self._hardware(edge.qubits)
            forbidden_labels.update((f"{control}-{target}", f"{target}-{control}"))
        touched = sorted(set(reset.labels) & forbidden_labels)
        if touched:
            raise ValueError(
                "Custom ancilla reset must not touch data-drive or CR labels: "
                f"{touched}."
            )
        self.schedule.call(reset)
        self.schedule.barrier()
        self.timeline.append(
            TimelineEvent(
                kind="RESET",
                name="custom_calibrated_ancilla_reset",
                start_ns=start,
                end_ns=float(self.schedule.duration),
                qubits=ANCILLA_QUBITS,
                hardware_qubits=self._hardware(ANCILLA_QUBITS),
                round_index=round_index,
                metadata={"strategy": "custom"},
            )
        )

    def _add_serial_cx(
        self,
        edge: DirectedEdge,
        *,
        round_index: int,
        layer: Layer,
        slot: int,
    ) -> None:
        """Add one logical CX surrounded by global barriers."""
        context = f"serial:{layer.name}:S{slot}"
        start = self._now()
        macro = self.adapter.cx_cr(edge, context=context)
        control, target = self._hardware(edge.qubits)
        direct_cr_label = f"{control}-{target}"
        reverse_cr_label = f"{target}-{control}"
        if direct_cr_label not in macro.labels:
            raise ValueError(
                f"{edge.edge_id} logical CX must expose direct CR label "
                f"{direct_cr_label}."
            )
        if reverse_cr_label in macro.labels and reverse_cr_label != direct_cr_label:
            raise ValueError(
                f"{edge.edge_id} logical CX contains forbidden reverse CR label "
                f"{reverse_cr_label}."
            )
        self.schedule.call(macro)
        self.schedule.barrier()
        end = float(self.schedule.duration)
        self.timeline.append(
            TimelineEvent(
                kind="CX_CR",
                name=f"{layer.name}_slot_{slot}_{edge.edge_id}",
                start_ns=start,
                end_ns=end,
                qubits=edge.qubits,
                hardware_qubits=self._hardware(edge.qubits),
                round_index=round_index,
                layer=layer.name,
                slot=slot,
                edge_id=edge.edge_id,
                context=context,
                metadata={
                    "logical_action": "CX",
                    "physical_direction": "data_to_ancilla",
                    "direct_cr_label": direct_cr_label,
                },
            )
        )

    def _add_layer(self, layer: Layer, *, round_index: int) -> None:
        """Add one explicit pre-H, two-CX, post-H layer."""
        self._add_parallel_1q(
            "H",
            (layer.x_check_data,),
            round_index=round_index,
            layer=layer.name,
            name=f"{layer.name}_pre_h",
        )
        for slot, edge in enumerate(layer.edges, start=1):
            self._add_serial_cx(
                edge,
                round_index=round_index,
                layer=layer,
                slot=slot,
            )
        self._add_parallel_1q(
            "H",
            (layer.x_check_data,),
            round_index=round_index,
            layer=layer.name,
            name=f"{layer.name}_post_h",
        )

    def _align_readout_capture(self, pre_margin_ns: float) -> None:
        """Pad globally so the non-blank readout start satisfies backend alignment."""
        current = self._now()
        capture_start = current + pre_margin_ns
        aligned = _ceil_to_grid(
            capture_start,
            self.adapter.capture_alignment_ns,
        )
        padding = aligned - capture_start
        if padding > 1e-9:
            self._add_global_wait(
                padding,
                kind="DELAY",
                name="capture_alignment_padding",
                round_index=None,
                dd_enabled=False,
                metadata={"alignment_ns": self.adapter.capture_alignment_ns},
            )

    def _add_readout(
        self,
        symbols: Sequence[SymbolicQubit],
        *,
        role: Literal["initial_ground", "syndrome", "final_data"],
        round_index: int | None,
    ) -> None:
        """Add simultaneous readout pulses and optional data DD."""
        prepared: list[tuple[SymbolicQubit, str, Waveform, float, float]] = [
            (symbol, *self.adapter.readout(symbol)) for symbol in symbols
        ]
        pre_margins = {item[3] for item in prepared}
        if len(pre_margins) != 1:
            raise ValueError(
                "Simultaneous readout requires a common pre-margin for alignment."
            )
        pre_margin = next(iter(pre_margins))
        self._align_readout_capture(pre_margin)
        start = self._now()
        window_ns = max(waveform.duration for _, _, waveform, _, _ in prepared)
        for symbol, read_label, waveform, read_pre, capture_duration in prepared:
            self.schedule.add(read_label, waveform)
            index = self._capture_count[symbol]
            self._capture_count[symbol] += 1
            acquisition_label = (
                f"m_{symbol}[{round_index}]"
                if role == "syndrome"
                else (
                    f"initial_{symbol}"
                    if role == "initial_ground"
                    else f"final_{symbol}"
                )
            )
            self.captures.append(
                CaptureRef(
                    acquisition_label=acquisition_label,
                    symbolic_qubit=symbol,
                    hardware_qubit=self.adapter.qubit_label(symbol),
                    readout_label=read_label,
                    capture_index=index,
                    capture_start_ns=start + read_pre,
                    capture_duration_ns=capture_duration,
                    role=role,
                    round_index=round_index,
                )
            )
            self.timeline.append(
                TimelineEvent(
                    kind="READOUT",
                    name=acquisition_label,
                    start_ns=start,
                    end_ns=start + waveform.duration,
                    qubits=(symbol,),
                    hardware_qubits=(self.adapter.qubit_label(symbol),),
                    round_index=round_index,
                    acquisition_label=acquisition_label,
                    metadata={
                        "readout_label": read_label,
                        "capture_index": index,
                        "capture_start_ns": start + read_pre,
                        "capture_duration_ns": capture_duration,
                        "basis": "Z",
                        "role": role,
                    },
                )
            )

        if (
            role == "syndrome"
            and self.config.dd.enabled
            and self.config.dd.during_syndrome_readout
        ):
            for symbol in DATA_QUBITS:
                waveform = self.adapter.dd_waveform(
                    symbol,
                    duration_ns=window_ns,
                    config=self.config.dd,
                )
                self.schedule.add(self.adapter.qubit_label(symbol), waveform)
                self.timeline.append(
                    TimelineEvent(
                        kind="DD",
                        name=f"syndrome_readout_dd_{symbol}",
                        start_ns=start,
                        end_ns=start + window_ns,
                        qubits=(symbol,),
                        hardware_qubits=(self.adapter.qubit_label(symbol),),
                        round_index=round_index,
                        metadata={"pulse_axes": self.config.dd.pulse_axes},
                    )
                )
        self.schedule.barrier()

    def build(self) -> BuiltSurfaceCodeExperiment:
        """Build and statically validate the configured fixed timeline."""
        with self.schedule:
            if self.config.initial_ground_postselection:
                self._add_readout(
                    SYMBOLIC_QUBITS,
                    role="initial_ground",
                    round_index=None,
                )
                recovery_delay = (
                    0.0
                    if self.config.initial_readout_recovery_delay_ns is None
                    else self.config.initial_readout_recovery_delay_ns
                )
                self._add_global_wait(
                    recovery_delay,
                    kind="DELAY",
                    name="initial_readout_recovery",
                    round_index=None,
                    dd_enabled=False,
                    metadata={
                        "requires_calibration": True,
                        "configured": (
                            self.config.initial_readout_recovery_delay_ns is not None
                        ),
                    },
                )
            self._prepare_seed()
            for round_index in range(self.config.n_stabilizer_rounds):
                self._reset_ancillas(round_index)
                for layer in LAYERS:
                    self._add_layer(layer, round_index=round_index)
                self._add_readout(
                    ANCILLA_QUBITS,
                    role="syndrome",
                    round_index=round_index,
                )
            self._add_final_measurement_rotations()
            self._add_readout(
                DATA_QUBITS,
                role="final_data",
                round_index=None,
            )

        timeline = tuple(self.timeline)
        captures = tuple(self.captures)
        validation = validate_built_schedule(
            schedule=self.schedule,
            timeline=timeline,
            captures=captures,
            config=self.config,
            sampling_period_ns=self.adapter.sampling_period_ns,
            capture_alignment_ns=self.adapter.capture_alignment_ns,
        )
        validation.require_ok()
        return BuiltSurfaceCodeExperiment(
            config=self.config,
            schedule=self.schedule,
            timeline=timeline,
            captures=captures,
            validation=validation,
        )


def validate_built_schedule(
    *,
    schedule: PulseSchedule,
    timeline: Sequence[TimelineEvent],
    captures: Sequence[CaptureRef],
    config: SurfaceCodeConfig,
    sampling_period_ns: float,
    capture_alignment_ns: float,
) -> ValidationReport:
    """
    Validate serial ordering, direction, timing grid, and measurement placement.

    This validation is semantic and pulse-schedule level.  Backend deployment
    validation remains authoritative for physical AWG/resource aliases.
    """
    issues: list[ValidationIssue] = []
    if not schedule.is_valid():
        issues.append(
            ValidationIssue("error", "invalid_schedule", "PulseSchedule is invalid.")
        )
    if len(set(config.qubit_map.values())) != len(SYMBOLIC_QUBITS):
        issues.append(
            ValidationIssue(
                "error",
                "qubit_map_not_unique",
                "Symbolic-to-physical qubit mapping is not one-to-one.",
            )
        )

    cx_events = [event for event in timeline if event.kind == "CX_CR"]
    expected_per_round = [(edge.control, edge.target) for edge in REQUIRED_EDGES]
    for round_index in range(config.n_stabilizer_rounds):
        round_cx = [event for event in cx_events if event.round_index == round_index]
        actual = [
            cast(tuple[DataQubit, AncillaQubit], event.qubits) for event in round_cx
        ]
        if actual != expected_per_round:
            issues.append(
                ValidationIssue(
                    "error",
                    "edge_order",
                    f"Round {round_index} edge order is {actual}, "
                    f"expected {expected_per_round}.",
                )
            )
        a2_order = [event.qubits[0] for event in round_cx if event.qubits[1] == "A2"]
        if a2_order != ["D1", "D3", "D2", "D4"]:
            issues.append(
                ValidationIssue(
                    "error",
                    "a2_order",
                    f"Round {round_index} A2 order is {a2_order}.",
                )
            )
        for previous, current in pairwise(round_cx):
            if current.start_ns < previous.end_ns - 1e-9:
                issues.append(
                    ValidationIssue(
                        "error",
                        "cr_overlap",
                        f"{previous.name} overlaps {current.name}.",
                    )
                )
        if any(
            event.qubits[0] not in DATA_QUBITS or event.qubits[1] not in ANCILLA_QUBITS
            for event in round_cx
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "cr_direction",
                    f"Round {round_index} contains a non data-to-ancilla CX.",
                )
            )

        semantic = [
            (event.kind, event.qubits)
            for event in timeline
            if event.round_index == round_index
            and event.layer is not None
            and event.kind in ("H", "CX_CR")
        ]
        expected_semantic = [
            (operation.kind, operation.qubits)
            for operation in serial_round_operations()
        ]
        if semantic != expected_semantic:
            issues.append(
                ValidationIssue(
                    "error",
                    "layer_semantics",
                    f"Round {round_index} layer event sequence changed.",
                )
            )

        round_readouts = [
            event
            for event in timeline
            if event.kind == "READOUT"
            and event.round_index == round_index
            and event.qubits[0] in ANCILLA_QUBITS
        ]
        if len(round_readouts) != len(ANCILLA_QUBITS):
            issues.append(
                ValidationIssue(
                    "error",
                    "syndrome_readout_count",
                    f"Round {round_index} does not have three syndrome readouts.",
                )
            )
        elif round_cx:
            readout_start = min(event.start_ns for event in round_readouts)
            layer_end = max(
                event.end_ns
                for event in timeline
                if event.round_index == round_index
                and event.layer is not None
                and event.kind in ("H", "CX_CR")
            )
            if readout_start < layer_end - 1e-9:
                issues.append(
                    ValidationIssue(
                        "error",
                        "early_readout",
                        f"Round {round_index} readout starts before the L4 post-H "
                        "completes.",
                    )
                )

    for event in timeline:
        if not _is_grid_aligned(event.start_ns, sampling_period_ns):
            issues.append(
                ValidationIssue(
                    "error",
                    "start_off_grid",
                    f"{event.name} starts at {event.start_ns} ns.",
                )
            )
        if not _is_grid_aligned(event.end_ns, sampling_period_ns):
            issues.append(
                ValidationIssue(
                    "error",
                    "end_off_grid",
                    f"{event.name} ends at {event.end_ns} ns.",
                )
            )
        if event.kind == "READOUT":
            capture_start = float(event.metadata["capture_start_ns"])
            capture_duration = float(event.metadata["capture_duration_ns"])
            if not _is_grid_aligned(capture_start, capture_alignment_ns):
                issues.append(
                    ValidationIssue(
                        "error",
                        "capture_start_alignment",
                        f"{event.name} capture starts at {capture_start} ns.",
                    )
                )
            if not _is_grid_aligned(capture_duration, capture_alignment_ns):
                issues.append(
                    ValidationIssue(
                        "error",
                        "capture_duration_alignment",
                        f"{event.name} capture duration is {capture_duration} ns.",
                    )
                )

    ancilla_h = [
        event
        for event in timeline
        if event.kind == "H" and event.qubits[0] in ANCILLA_QUBITS
    ]
    if ancilla_h:
        issues.append(
            ValidationIssue(
                "error",
                "ancilla_h",
                "Ancilla Hadamards are forbidden in the data-control conversion.",
            )
        )

    initial_capture_count = (
        len(SYMBOLIC_QUBITS) if config.initial_ground_postselection else 0
    )
    expected_capture_count = (
        initial_capture_count
        + config.n_stabilizer_rounds * len(ANCILLA_QUBITS)
        + len(DATA_QUBITS)
    )
    if len(captures) != expected_capture_count:
        issues.append(
            ValidationIssue(
                "error",
                "capture_count",
                f"Capture map has {len(captures)} entries, "
                f"expected {expected_capture_count}.",
            )
        )
    initial_captures = [
        capture for capture in captures if capture.role == "initial_ground"
    ]
    if len(initial_captures) != initial_capture_count:
        issues.append(
            ValidationIssue(
                "error",
                "initial_readout_count",
                f"Initial ground readout has {len(initial_captures)} captures, "
                f"expected {initial_capture_count}.",
            )
        )
    initial_readouts = [
        event
        for event in timeline
        if event.kind == "READOUT" and event.metadata.get("role") == "initial_ground"
    ]
    if initial_readouts:
        if len({event.start_ns for event in initial_readouts}) != 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "initial_readout_not_simultaneous",
                    "All seven initial readout pulses must start together.",
                )
            )
        later_operations = [
            event
            for event in timeline
            if event.kind in ("RESET", "XPI", "H", "CX_CR", "BASIS_ROTATION")
        ]
        if (
            later_operations
            and max(event.end_ns for event in initial_readouts)
            > min(event.start_ns for event in later_operations) + 1e-9
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "initial_readout_order",
                    "Initial readout must complete before state preparation.",
                )
            )

    final_rotations = [event for event in timeline if event.kind == "BASIS_ROTATION"]
    expected_rotations = [
        (symbol, axis)
        for symbol, axis in zip(
            DATA_QUBITS,
            config.resolved_final_measurement_axes,
            strict=True,
        )
        if axis != "Z"
    ]
    actual_rotations = [
        (cast(DataQubit, event.qubits[0]), event.metadata.get("axis"))
        for event in final_rotations
    ]
    if actual_rotations != expected_rotations:
        issues.append(
            ValidationIssue(
                "error",
                "final_basis",
                f"Final basis rotations are {actual_rotations}, "
                f"expected {expected_rotations}.",
            )
        )
    if config.hadamard_decomposition == "Z180-Y90":
        issues.append(
            ValidationIssue(
                "warning",
                "virtual_z_hadamard",
                "Qubex marks Z180-Y90 Hadamard phase correction for CR targets "
                "as unresolved; validate the frame convention.",
            )
        )
    if not config.acknowledge_native_frame_calibration:
        issues.append(
            ValidationIssue(
                "warning",
                "frame_calibration_unacknowledged",
                "Hardware execution must remain disabled until serial-context "
                "frame/truth-table calibration is acknowledged.",
            )
        )
    if (
        not config.initial_ground_postselection
        and not config.acknowledge_initial_ground_state
    ):
        issues.append(
            ValidationIssue(
                "warning",
                "initial_ground_state_unacknowledged",
                "Without initial heralding, hardware execution must remain disabled "
                "until passive data/ancilla ground-state preparation is validated.",
            )
        )
    if config.initial_ground_postselection:
        if config.initial_readout_recovery_delay_ns is None:
            issues.append(
                ValidationIssue(
                    "warning",
                    "initial_readout_recovery_unspecified",
                    "Hardware execution requires a calibrated resonator-recovery "
                    "delay after the initial herald readout.",
                )
            )
        if not config.acknowledge_initial_readout_calibration:
            issues.append(
                ValidationIssue(
                    "warning",
                    "initial_readout_calibration_unacknowledged",
                    "Hardware execution must remain disabled until simultaneous "
                    "initial readout, assignment, and QND behavior are validated.",
                )
            )
    if config.shot_interval_ns is None:
        issues.append(
            ValidationIssue(
                "warning",
                "shot_interval_unspecified",
                "Hardware execution requires an explicit calibrated shot interval.",
            )
        )

    if config.max_waveform_amplitude is not None:
        sampled = schedule.get_sampled_sequences()
        for label, waveform in sampled.items():
            peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
            if peak > config.max_waveform_amplitude + 1e-12:
                issues.append(
                    ValidationIssue(
                        "error",
                        "waveform_peak",
                        f"{label} peak {peak} exceeds configured limit "
                        f"{config.max_waveform_amplitude}.",
                    )
                )

    return ValidationReport(tuple(issues))


def build_serialized_d2_surface_code_schedule(
    adapter: SurfaceCodePulseAdapter,
    config: SurfaceCodeConfig,
) -> BuiltSurfaceCodeExperiment:
    """
    Build and validate the device-independent serialized pulse schedule.

    This is the adapter boundary for tests and for sites that provide their own
    calibrated logical-CX or reset implementation.
    """
    return _SurfaceCodeSequenceBuilder(adapter, config).build()


def validate_compiled_capture_schedule(
    built: BuiltSurfaceCodeExperiment,
    measurement_schedule: MeasurementSchedule,
) -> tuple[ValidationReport, float]:
    """Compare semantic captures with backend-compiled non-workaround captures."""
    issues: list[ValidationIssue] = []
    expected_by_label: dict[str, list[CaptureRef]] = {}
    for reference in built.captures:
        expected_by_label.setdefault(reference.readout_label, []).append(reference)

    actual_by_label: dict[str, list[Any]] = {}
    for capture in measurement_schedule.capture_schedule.captures:
        if capture.is_workaround:
            continue
        for channel in capture.channels:
            actual_by_label.setdefault(channel, []).append(capture)

    unexpected = sorted(set(actual_by_label) - set(expected_by_label))
    if unexpected:
        issues.append(
            ValidationIssue(
                "error",
                "unexpected_compiled_capture",
                f"Compiled schedule has untracked readout captures on {unexpected}.",
            )
        )

    offsets: list[float] = []
    for readout_label, expected in expected_by_label.items():
        ordered_expected = sorted(expected, key=lambda item: item.capture_index)
        ordered_actual = sorted(
            actual_by_label.get(readout_label, []),
            key=lambda item: item.start_time,
        )
        if len(ordered_actual) != len(ordered_expected):
            issues.append(
                ValidationIssue(
                    "error",
                    "compiled_capture_count",
                    f"{readout_label} has {len(ordered_actual)} compiled captures; "
                    f"expected {len(ordered_expected)}.",
                )
            )
            continue
        for reference, capture in zip(ordered_expected, ordered_actual, strict=True):
            offsets.append(float(capture.start_time) - reference.capture_start_ns)
            if not np.isclose(
                float(capture.duration),
                reference.capture_duration_ns,
                atol=1e-9,
            ):
                issues.append(
                    ValidationIssue(
                        "error",
                        "compiled_capture_duration",
                        f"{reference.acquisition_label} compiled duration "
                        f"{capture.duration} ns differs from semantic duration "
                        f"{reference.capture_duration_ns} ns.",
                    )
                )

    compiled_offset = offsets[0] if offsets else 0.0
    if any(not np.isclose(offset, compiled_offset, atol=1e-9) for offset in offsets):
        issues.append(
            ValidationIssue(
                "error",
                "compiled_capture_offset",
                "Backend compilation did not preserve one common timeline offset.",
            )
        )
    if compiled_offset < -1e-9:
        issues.append(
            ValidationIssue(
                "error",
                "negative_compiled_offset",
                f"Backend compilation shifted captures backward by {compiled_offset} ns.",
            )
        )
    return ValidationReport(tuple(issues)), compiled_offset


def build_d2_surface_code_experiment(
    exp: Experiment,
    config: SurfaceCodeConfig,
    *,
    reset_schedule_factory: ResetScheduleFactory | None = None,
    compile_measurement_schedule: bool = True,
) -> BuiltSurfaceCodeExperiment:
    """
    Build the serialized d=2 schedule using calibrated Qubex operations.

    Parameters
    ----------
    exp
        Initialized Qubex experiment.  Hardware connection is not required for
        local schedule construction when configuration is already available.
    config
        Surface-code settings and symbolic-to-physical mapping.
    reset_schedule_factory
        Required for ``reset_strategy='custom'``.
    compile_measurement_schedule
        Whether to prebuild Qubex capture windows for validation/plotting.

    Returns
    -------
    BuiltSurfaceCodeExperiment
        Pulse schedule, semantic timeline, acquisition map, and validation.
    """
    adapter = QubexPulseAdapter(
        exp,
        config,
        reset_schedule_factory=reset_schedule_factory,
    )
    calibration = preflight_directed_cr_calibrations(adapter)
    calibration.require_ok()
    built = replace(
        build_serialized_d2_surface_code_schedule(adapter, config),
        execution_provenance="qubex_native_cnot",
    )
    if not compile_measurement_schedule:
        return built
    measurement_schedule = exp.build_measurement_schedule(
        pulse_schedule=built.schedule.copy(),
        final_measurement=False,
        capture_placement="pulse_aligned",
        plot=False,
    )
    compiled_validation, compiled_offset = validate_compiled_capture_schedule(
        built,
        measurement_schedule,
    )
    validation = ValidationReport(built.validation.issues + compiled_validation.issues)
    validation.require_ok()
    return replace(
        built,
        validation=validation,
        measurement_schedule=measurement_schedule,
        compiled_time_offset_ns=compiled_offset,
    )


def _capture_targets(
    built: BuiltSurfaceCodeExperiment,
    *,
    role: Literal["initial_ground", "syndrome", "final_data"],
) -> list[tuple[str, int]]:
    """Return result target/index pairs in semantic analysis order."""
    if role == "syndrome":
        references = sorted(
            (capture for capture in built.captures if capture.role == role),
            key=lambda capture: (
                cast(int, capture.round_index),
                ANCILLA_QUBITS.index(cast(AncillaQubit, capture.symbolic_qubit)),
            ),
        )
    elif role == "initial_ground":
        references = sorted(
            (capture for capture in built.captures if capture.role == role),
            key=lambda capture: SYMBOLIC_QUBITS.index(capture.symbolic_qubit),
        )
    else:
        references = sorted(
            (capture for capture in built.captures if capture.role == role),
            key=lambda capture: DATA_QUBITS.index(
                cast(DataQubit, capture.symbolic_qubit)
            ),
        )
    return [(capture.hardware_qubit, capture.capture_index) for capture in references]


def analyze_multiple_measure_result(
    result: MultipleMeasureResult,
    built: BuiltSurfaceCodeExperiment,
) -> SurfaceCodeAnalysis:
    """
    Classify and analyze the captures returned by ``Experiment.execute``.

    Low-confidence shots marked ``-1`` by Qubex are removed consistently from
    all syndrome and data captures before analysis.
    """
    initial_targets = _capture_targets(built, role="initial_ground")
    syndrome_targets = _capture_targets(built, role="syndrome")
    data_targets = _capture_targets(built, role="final_data")
    try:
        initial_columns = (
            result.get_classified_data(
                initial_targets,
                threshold=built.config.classifier_threshold,
            )
            if initial_targets
            else None
        )
        syndrome_columns = result.get_classified_data(
            syndrome_targets,
            threshold=built.config.classifier_threshold,
        )
        data_columns = result.get_classified_data(
            data_targets,
            threshold=built.config.classifier_threshold,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "Could not classify surface-code captures. Ensure a classifier is "
            "loaded for all seven physical qubits."
        ) from exc

    n_result_shots = syndrome_columns.shape[0]
    if data_columns.shape[0] != n_result_shots or (
        initial_columns is not None and initial_columns.shape[0] != n_result_shots
    ):
        raise ValueError("Surface-code captures have different shot counts.")
    valid = np.all(syndrome_columns >= 0, axis=1) & np.all(data_columns >= 0, axis=1)
    if initial_columns is not None:
        valid &= np.all(initial_columns >= 0, axis=1)
    valid_indices = np.flatnonzero(valid)
    initial = (
        None
        if initial_columns is None
        else initial_columns[valid].reshape(
            valid_indices.size,
            len(SYMBOLIC_QUBITS),
        )
    )
    syndrome = syndrome_columns[valid].reshape(
        valid_indices.size,
        built.config.n_stabilizer_rounds,
        len(ANCILLA_QUBITS),
    )
    data = data_columns[valid].reshape(valid_indices.size, len(DATA_QUBITS))
    analysis_config = replace(built.config, n_shots=int(valid_indices.size))
    return analyze_bit_arrays(
        config=analysis_config,
        initial_ground_bits=initial,
        syndrome_bits=syndrome,
        final_data_bits=data,
        source_shot_indices=valid_indices,
    )


@dataclass(frozen=True)
class ExecutedSurfaceCodeExperiment:
    """Hold raw Qubex captures together with decoded surface-code data."""

    built: BuiltSurfaceCodeExperiment
    raw_result: MultipleMeasureResult
    analysis: SurfaceCodeAnalysis


def execute_d2_surface_code_experiment(
    exp: Experiment,
    built: BuiltSurfaceCodeExperiment,
    *,
    plot: bool = False,
) -> ExecutedSurfaceCodeExperiment:
    """
    Execute an already-built surface-code schedule on connected hardware.

    Notes
    -----
    ``reset_awg_and_capunits`` resets electronics, not qubit states.  Qubit
    reset semantics are entirely determined by ``SurfaceCodeConfig``.
    """
    if not built.config.acknowledge_native_frame_calibration:
        raise ValueError(
            "Refusing hardware execution until "
            "acknowledge_native_frame_calibration=true. Validate the logical CX "
            "truth table and frame convention in every serial context first."
        )
    if built.execution_provenance != "qubex_native_cnot":
        raise ValueError(
            "Refusing hardware execution for an adapter-only schedule. Build through "
            "build_d2_surface_code_experiment so every CR primitive is wrapped by "
            "Qubex Experiment.cnot."
        )
    if (
        not built.config.initial_ground_postselection
        and not built.config.acknowledge_initial_ground_state
    ):
        raise ValueError(
            "Refusing hardware execution until "
            "acknowledge_initial_ground_state=true when initial heralding is "
            "disabled. Qubex does not emit a data-qubit reset in this sequence."
        )
    if built.config.initial_ground_postselection:
        if not built.config.acknowledge_initial_readout_calibration:
            raise ValueError(
                "Refusing hardware execution until "
                "acknowledge_initial_readout_calibration=true. Validate "
                "simultaneous assignment, QND behavior, and false acceptance."
            )
        if built.config.initial_readout_recovery_delay_ns is None:
            raise ValueError(
                "Hardware execution with initial ground post-selection requires "
                "an experimentally validated initial_readout_recovery_delay_ns."
            )
    if built.config.shot_interval_ns is None:
        raise ValueError(
            "Hardware execution requires an explicit, experimentally validated "
            "shot_interval_ns for sequence repetition and device recovery."
        )
    if built.measurement_schedule is None:
        raise ValueError(
            "Hardware execution requires compiled capture validation; rebuild with "
            "compile_measurement_schedule=True."
        )
    built.validation.require_ok()
    result = exp.execute(
        schedule=built.schedule.copy(),
        mode="single",
        n_shots=built.config.n_shots,
        shot_interval=built.config.shot_interval_ns,
        time_integration=True,
        add_last_measurement=False,
        enable_dsp_classification=False,
        reset_awg_and_capunits=True,
        plot=plot,
    )
    analysis = analyze_multiple_measure_result(result, built)
    return ExecutedSurfaceCodeExperiment(
        built=built,
        raw_result=result,
        analysis=analysis,
    )


def example_config_dict() -> dict[str, Any]:
    """
    Return a ready-to-edit workspace configuration template.

    The physical map is inferred from the seven directed calibration labels in
    the local ``144Qv2-quel3`` surface-code configuration.  It must be verified
    by the operator.  At the time this file was generated, ``D4_A2``
    (``Q091-Q066``) was absent and most listed CR durations were zero, so
    preflight intentionally blocks execution until calibration is completed.
    """
    config_root = Path("/home/ban/workspace_2026/qubex-config-Q28-surface/144Qv2-quel3")
    return {
        "notes": [
            "Verify the inferred symbolic qubit mapping before use.",
            "Do not increase calibration_valid_days merely to bypass stale data.",
            "Calibrate all eight directed data-to-ancilla CR edges first.",
            "Set acknowledge_native_frame_calibration only after serial-context validation.",
            "Set shot_interval_ns from repetition/recovery calibration. "
            "acknowledge_initial_ground_state is required only when initial "
            "heralding is disabled.",
            "Set initial_readout_recovery_delay_ns and "
            "acknowledge_initial_readout_calibration only after validating "
            "simultaneous herald readout and resonator recovery.",
        ],
        "qubex": {
            "system_id": "144Qv2-quel3",
            "config_dir": str(config_root / "config"),
            "params_dir": str(config_root / "params"),
            "calib_note_path": str(config_root / "calibration" / "calib_note.json"),
            "configuration_mode": "ge-cr-cr",
            "calibration_valid_days": None,
        },
        "surface_code": {
            "qubit_map": {
                "D1": "Q064",
                "D2": "Q067",
                "D3": "Q088",
                "D4": "Q091",
                "A1": "Q065",
                "A2": "Q066",
                "A3": "Q089",
            },
            "logical_state": "0L",
            "custom_bitstring": None,
            "initial_basis": None,
            "n_stabilizer_rounds": 1,
            "final_basis": "Z",
            "final_measurement_axes": None,
            "syndrome_mode": "postselect",
            "postselect_syndrome": [0, 0, 0],
            "n_shots": 1_024,
            "shot_interval_ns": None,
            "use_simultaneous_cr": False,
            "reset_strategy": "shot_boundary",
            "passive_reset_delay_ns": None,
            "hadamard_decomposition": "Y90-X180",
            "readout_inversion": {"A1": False, "A2": False, "A3": False},
            "data_readout_inversion": {
                "D1": False,
                "D2": False,
                "D3": False,
                "D4": False,
            },
            "dd": {
                "enabled": False,
                "pulse_axes": [],
                "repeat_count": 1,
                "during_syndrome_readout": True,
                "during_reset": True,
            },
            "cr_overrides": {},
            "classifier_threshold": None,
            "initial_ground_postselection": True,
            "initial_readout_recovery_delay_ns": None,
            "acknowledge_native_frame_calibration": False,
            "acknowledge_initial_ground_state": False,
            "acknowledge_initial_readout_calibration": False,
            "max_waveform_amplitude": None,
        },
    }


def write_config_template(path: Path | str) -> Path:
    """Write the editable JSON configuration template without overwriting."""
    destination = Path(path)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(
            example_config_dict(),
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")
    return destination


def _print_calibration_report(report: CalibrationReport) -> None:
    """Print a concise directed-edge preflight table."""
    print("edge      physical direction   status  detail")
    for edge in report.edges:
        status = "PASS" if edge.ok else "FAIL"
        detail = "ok" if edge.ok else "; ".join(edge.issues)
        print(
            f"{edge.edge_id:<9} {edge.control}->{edge.target:<10} {status:<6} {detail}"
        )
    for warning in report.warnings:
        print(f"WARNING: {warning}")


def _run_self_test() -> int:
    """Run offline semantic and ideal-circuit validation."""
    operations = serial_round_operations()
    a2_order = [
        operation.qubits[0]
        for operation in operations
        if operation.kind == "CX_CR" and operation.qubits[1] == "A2"
    ]
    checks = run_ideal_validations()
    print(f"serial A2 order: {' -> '.join(a2_order)}")
    for name, check in checks.items():
        print(f"{name}: {'PASS' if check.passed else 'FAIL'} ({check.detail})")
    return 0 if all(check.passed for check in checks.values()) else 1


def _load_and_create_experiment(
    path: Path | str,
) -> tuple[ExperimentFileConfig, Experiment]:
    """Load JSON and create an unconnected Qubex experiment."""
    file_config = ExperimentFileConfig.load(path)
    exp = file_config.qubex.create_experiment(file_config.surface_code.qubit_map)
    return file_config, exp


def _write_analysis(
    path: Path | str,
    executed: ExecutedSurfaceCodeExperiment,
) -> Path:
    """Write decoded experiment data without overwriting an existing file."""
    destination = Path(path)
    payload = {
        "config": asdict(executed.built.config),
        "analysis": executed.analysis.to_json_dict(),
        "capture_map": [asdict(capture) for capture in executed.built.captures],
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build or run a d=2 data-control surface-code experiment with "
            "strictly serialized CR pairs."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("self-test", help="run offline ideal-circuit checks")

    template = subparsers.add_parser(
        "write-template",
        help="write a ready-to-edit JSON configuration",
    )
    template.add_argument("path", type=Path)

    preflight = subparsers.add_parser(
        "preflight",
        help="validate all directed CR calibrations without hardware execution",
    )
    preflight.add_argument("config", type=Path)

    build = subparsers.add_parser(
        "build",
        help="build and validate the pulse/capture schedule without executing",
    )
    build.add_argument("config", type=Path)
    build.add_argument("--timeline", type=Path)
    build.add_argument("--plot", action="store_true")

    run = subparsers.add_parser("run", help="connect and execute on hardware")
    run.add_argument("config", type=Path)
    run.add_argument("--configure", action="store_true")
    run.add_argument("--timeline", type=Path)
    run.add_argument("--analysis", type=Path)
    run.add_argument("--plot", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run the standalone command-line workflow.

    ``run --configure`` is the only path that deploys configuration to hardware;
    normal ``build`` and ``preflight`` are read-only with respect to instruments.
    """
    args = _build_parser().parse_args(argv)
    if args.command == "self-test":
        return _run_self_test()
    if args.command == "write-template":
        destination = write_config_template(args.path)
        print(destination)
        return 0

    file_config, exp = _load_and_create_experiment(args.config)
    adapter = QubexPulseAdapter(exp, file_config.surface_code)
    calibration = preflight_directed_cr_calibrations(adapter)
    _print_calibration_report(calibration)
    if args.command == "preflight":
        return 0 if calibration.ok else 2
    calibration.require_ok()

    if args.command == "run":
        if not file_config.surface_code.acknowledge_native_frame_calibration:
            raise ValueError(
                "Refusing to connect: set acknowledge_native_frame_calibration "
                "only after serial-context CX validation."
            )
        if (
            not file_config.surface_code.initial_ground_postselection
            and not file_config.surface_code.acknowledge_initial_ground_state
        ):
            raise ValueError(
                "Refusing to connect: set acknowledge_initial_ground_state only "
                "after validating passive initialization when initial heralding "
                "is disabled."
            )
        if file_config.surface_code.initial_ground_postselection:
            if not file_config.surface_code.acknowledge_initial_readout_calibration:
                raise ValueError(
                    "Refusing to connect: set "
                    "acknowledge_initial_readout_calibration only after validating "
                    "simultaneous assignment and QND behavior."
                )
            if file_config.surface_code.initial_readout_recovery_delay_ns is None:
                raise ValueError(
                    "Refusing to connect: initial ground post-selection requires "
                    "a calibrated initial_readout_recovery_delay_ns."
                )
        if file_config.surface_code.shot_interval_ns is None:
            raise ValueError(
                "Refusing to connect: run requires an explicit calibrated "
                "shot_interval_ns."
            )
    if file_config.surface_code.reset_strategy == "custom":
        raise ValueError(
            "The standalone CLI cannot inject reset_schedule_factory. Use "
            "build_d2_surface_code_experiment(...) from Python for custom reset."
        )

    if args.command == "run":
        exp.connect()
        if args.configure:
            exp.configure()

    built = build_d2_surface_code_experiment(exp, file_config.surface_code)
    print(f"schedule duration: {built.schedule.duration:g} ns")
    print(f"captures: {len(built.captures)}")
    if args.timeline is not None:
        print(built.export_timeline(args.timeline))
    if args.plot:
        if built.measurement_schedule is None:
            built.schedule.plot()
        else:
            built.measurement_schedule.plot()

    if args.command == "run":
        executed = execute_d2_surface_code_experiment(
            exp,
            built,
            plot=args.plot,
        )
        accepted = int(np.count_nonzero(executed.analysis.accepted_shots))
        print(f"accepted classified shots: {accepted}")
        if args.analysis is not None:
            print(_write_analysis(args.analysis, executed))
    return 0


__all__ = [
    "ANCILLA_QUBITS",
    "DATA_QUBITS",
    "LAYERS",
    "REQUIRED_EDGES",
    "SYMBOLIC_QUBITS",
    "AbstractOperation",
    "Basis",
    "BuiltSurfaceCodeExperiment",
    "CalibrationReport",
    "CaptureRef",
    "DDConfig",
    "DirectedEdge",
    "ExecutedSurfaceCodeExperiment",
    "ExperimentFileConfig",
    "IdealCheck",
    "Layer",
    "MeasurementAxis",
    "QubexConnectionConfig",
    "QubexPulseAdapter",
    "SurfaceCodeAnalysis",
    "SurfaceCodeConfig",
    "SurfaceCodePulseAdapter",
    "TimelineEvent",
    "ValidationReport",
    "analyze_bit_arrays",
    "analyze_multiple_measure_result",
    "build_d2_surface_code_experiment",
    "build_serialized_d2_surface_code_schedule",
    "canonical_seed",
    "example_config_dict",
    "execute_d2_surface_code_experiment",
    "main",
    "preflight_directed_cr_calibrations",
    "run_ideal_validations",
    "serial_round_operations",
    "validate_built_schedule",
    "validate_compiled_capture_schedule",
    "write_config_template",
]


if __name__ == "__main__":
    raise SystemExit(main())

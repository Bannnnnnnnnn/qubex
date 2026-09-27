"""Tests for the standalone d=2 serial-CR surface-code experiment."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from qxpulse import Blank, PulseSchedule

from qubex.contrib.experiment.d2_surface_code_serial_cr import (
    DATA_QUBITS,
    REQUIRED_EDGES,
    SYMBOLIC_QUBITS,
    Basis,
    DDConfig,
    DirectedEdge,
    MeasurementAxis,
    SurfaceCodeConfig,
    analyze_bit_arrays,
    build_serialized_d2_surface_code_schedule,
    canonical_seed,
    run_ideal_validations,
    serial_round_operations,
    validate_compiled_capture_schedule,
)
from qubex.measurement import MeasurementSchedule
from qubex.measurement.models.capture_schedule import Capture, CaptureSchedule


def _qubit_map() -> dict[str, str]:
    return {
        "D1": "Q00",
        "D2": "Q01",
        "D3": "Q02",
        "D4": "Q03",
        "A1": "Q04",
        "A2": "Q05",
        "A3": "Q06",
    }


class _FakePulseAdapter:
    """Provide deterministic, calibration-free pulses for schedule tests."""

    @property
    def sampling_period_ns(self) -> float:
        """Return the fake control grid."""
        return 2.0

    @property
    def capture_alignment_ns(self) -> float:
        """Return the fake capture grid."""
        return 2.0

    @property
    def initial_labels(self) -> list[str]:
        """Return every fake drive, CR, and readout label."""
        qubit_map = _qubit_map()
        qubits = list(qubit_map.values())
        return [
            *qubits,
            *(f"R{qubit}" for qubit in qubits),
            *(
                f"{qubit_map[edge.control]}-{qubit_map[edge.target]}"
                for edge in REQUIRED_EDGES
            ),
        ]

    def qubit_label(self, symbol: str) -> str:
        """Resolve a symbolic test qubit."""
        return _qubit_map()[symbol]

    def xpi(self, symbol: str) -> Blank:
        """Return a four-nanosecond fake pi pulse."""
        return Blank(4.0)

    def hadamard(self, symbol: str) -> Blank:
        """Return a six-nanosecond fake Hadamard."""
        return Blank(6.0)

    def measurement_rotation(
        self,
        symbol: str,
        axis: MeasurementAxis,
    ) -> Blank | None:
        """Return deterministic X/Y measurement rotations."""
        if axis == "X":
            return Blank(6.0)
        if axis == "Y":
            return Blank(4.0)
        return None

    def cx_cr(self, edge: DirectedEdge, *, context: str) -> PulseSchedule:
        """Return a ten-nanosecond fake directed logical CX."""
        control = self.qubit_label(edge.control)
        target = self.qubit_label(edge.target)
        cr_label = f"{control}-{target}"
        with PulseSchedule([control, target, cr_label]) as schedule:
            for label in (control, target, cr_label):
                schedule.add(label, Blank(10.0))
        return schedule

    def readout(self, symbol: str) -> tuple[str, Blank, float, float]:
        """Return a fake readout pulse and capture window."""
        return f"R{self.qubit_label(symbol)}", Blank(12.0), 0.0, 8.0

    def dd_waveform(
        self,
        symbol: str,
        *,
        duration_ns: float,
        config: DDConfig,
    ) -> Blank:
        """Return a blank identity of the requested duration."""
        return Blank(duration_ns)

    def reset_schedule(self, round_index: int) -> PulseSchedule:
        """Reject the unused custom-reset path."""
        raise AssertionError("custom reset is not configured")


class _ReversePulseAdapter(_FakePulseAdapter):
    """Return a forbidden reverse-direction CR resource label."""

    def cx_cr(self, edge: DirectedEdge, *, context: str) -> PulseSchedule:
        """Return a fake macro whose physical CR direction is reversed."""
        control = self.qubit_label(edge.control)
        target = self.qubit_label(edge.target)
        reverse_cr_label = f"{target}-{control}"
        with PulseSchedule([control, target, reverse_cr_label]) as schedule:
            for label in (control, target, reverse_cr_label):
                schedule.add(label, Blank(10.0))
        return schedule


class _DataTouchingResetAdapter(_FakePulseAdapter):
    """Return a custom reset that incorrectly drives one data qubit."""

    def reset_schedule(self, round_index: int) -> PulseSchedule:
        """Return an invalid reset schedule on D1."""
        label = self.qubit_label("D1")
        with PulseSchedule([label]) as schedule:
            schedule.add(label, Blank(4.0))
        return schedule


def test_serial_round_preserves_layer_and_a2_edge_order() -> None:
    """One round should keep both layer slots serial and preserve the hook-sensitive A2 order."""
    operations = serial_round_operations()

    cx_operations = [operation for operation in operations if operation.kind == "CX_CR"]
    assert [
        (operation.layer, operation.slot, operation.qubits)
        for operation in cx_operations
    ] == [
        ("L1", 1, ("D1", "A2")),
        ("L1", 2, ("D3", "A3")),
        ("L2", 1, ("D3", "A2")),
        ("L2", 2, ("D4", "A3")),
        ("L3", 1, ("D1", "A1")),
        ("L3", 2, ("D2", "A2")),
        ("L4", 1, ("D2", "A1")),
        ("L4", 2, ("D4", "A2")),
    ]
    assert [
        operation.qubits[0]
        for operation in cx_operations
        if operation.qubits[1] == "A2"
    ] == ["D1", "D3", "D2", "D4"]


def test_serial_round_keeps_x_check_hadamards_outside_both_slots() -> None:
    """Each layer should wrap both serialized slots with the specified data Hadamard."""
    operations = serial_round_operations()

    by_layer = {
        layer: [operation for operation in operations if operation.layer == layer]
        for layer in ("L1", "L2", "L3", "L4")
    }
    assert [
        [(operation.kind, operation.qubits) for operation in by_layer[layer]]
        for layer in ("L1", "L2", "L3", "L4")
    ] == [
        [
            ("H", ("D1",)),
            ("CX_CR", ("D1", "A2")),
            ("CX_CR", ("D3", "A3")),
            ("H", ("D1",)),
        ],
        [
            ("H", ("D3",)),
            ("CX_CR", ("D3", "A2")),
            ("CX_CR", ("D4", "A3")),
            ("H", ("D3",)),
        ],
        [
            ("H", ("D2",)),
            ("CX_CR", ("D1", "A1")),
            ("CX_CR", ("D2", "A2")),
            ("H", ("D2",)),
        ],
        [
            ("H", ("D4",)),
            ("CX_CR", ("D2", "A1")),
            ("CX_CR", ("D4", "A2")),
            ("H", ("D4",)),
        ],
    ]


def test_built_schedule_serializes_every_logical_cx_globally() -> None:
    """Global barriers should make all eight logical-CX intervals disjoint."""
    built = build_serialized_d2_surface_code_schedule(
        _FakePulseAdapter(),
        SurfaceCodeConfig(qubit_map=_qubit_map(), n_shots=1),
    )

    cx_events = [event for event in built.timeline if event.kind == "CX_CR"]
    assert len(cx_events) == 8
    assert all(
        current.start_ns >= previous.end_ns for previous, current in pairwise(cx_events)
    )
    assert [event.end_ns - event.start_ns for event in cx_events] == [10.0] * 8
    assert built.validation.ok


def test_builder_rejects_reverse_physical_cr_label() -> None:
    """Reject a custom adapter that hides a reverse-direction CR macro."""
    with pytest.raises(ValueError, match="direct CR label"):
        build_serialized_d2_surface_code_schedule(
            _ReversePulseAdapter(),
            SurfaceCodeConfig(qubit_map=_qubit_map(), n_shots=1),
        )


def test_builder_rejects_custom_reset_on_data_drive() -> None:
    """Keep custom ancilla reset schedules off every data and CR resource."""
    with pytest.raises(ValueError, match="must not touch"):
        build_serialized_d2_surface_code_schedule(
            _DataTouchingResetAdapter(),
            SurfaceCodeConfig(
                qubit_map=_qubit_map(),
                reset_strategy="custom",
                n_shots=1,
            ),
        )


def test_compiled_capture_validation_rejects_untracked_readout() -> None:
    """Detect reset or helper readouts that would shift result capture indices."""
    built = build_serialized_d2_surface_code_schedule(
        _FakePulseAdapter(),
        SurfaceCodeConfig(qubit_map=_qubit_map(), n_shots=1),
    )
    captures = [
        Capture(
            channels=[reference.readout_label],
            start_time=reference.capture_start_ns,
            duration=reference.capture_duration_ns,
        )
        for reference in built.captures
    ]
    captures.append(
        Capture(
            channels=[built.captures[0].readout_label],
            start_time=1.0,
            duration=8.0,
        )
    )
    measurement_schedule = MeasurementSchedule(
        pulse_schedule=built.schedule.copy(),
        capture_schedule=CaptureSchedule(captures=captures),
    )

    report, _ = validate_compiled_capture_schedule(built, measurement_schedule)

    assert not report.ok
    assert any(issue.code == "compiled_capture_count" for issue in report.errors)


@pytest.mark.parametrize(
    ("logical_state", "expected"),
    [
        ("0L", ("0000", "Z")),
        ("1L", ("0011", "Z")),
        ("+L", ("0000", "X")),
        ("-L", ("0101", "X")),
    ],
)
def test_canonical_logical_seed_convention(
    logical_state: str,
    expected: tuple[str, str],
) -> None:
    """Canonical logical labels should resolve to the specified bitstring and basis."""
    assert canonical_seed(logical_state) == expected


def test_config_rejects_non_unique_physical_qubits() -> None:
    """A symbolic map should reject two roles assigned to one physical qubit."""
    qubit_map = _qubit_map()
    qubit_map["A3"] = qubit_map["D1"]

    with pytest.raises(ValueError, match="one-to-one"):
        SurfaceCodeConfig(qubit_map=qubit_map)


def test_config_rejects_simultaneous_cr() -> None:
    """Reject requests to restore simultaneous CR."""
    with pytest.raises(ValueError, match="serial"):
        SurfaceCodeConfig(qubit_map=_qubit_map(), use_simultaneous_cr=True)


@pytest.mark.parametrize(
    "field",
    [
        "use_simultaneous_cr",
        "initial_ground_postselection",
        "acknowledge_native_frame_calibration",
        "acknowledge_initial_ground_state",
        "acknowledge_initial_readout_calibration",
    ],
)
def test_json_config_rejects_truthy_strings_for_safety_booleans(field: str) -> None:
    """Reject string values instead of coercing them into safety acknowledgements."""
    with pytest.raises(TypeError, match="JSON boolean"):
        SurfaceCodeConfig.from_mapping(
            {
                "qubit_map": _qubit_map(),
                field: "false",
            }
        )


def test_config_rejects_unknown_cr_override_context() -> None:
    """Reject override keys that would otherwise be silently ignored."""
    with pytest.raises(ValueError, match="Unknown CR override target"):
        SurfaceCodeConfig(
            qubit_map=_qubit_map(),
            cr_overrides={"D1_A2@L1": {"cr_phase": 0.0}},
        )


def test_repeated_rounds_require_an_explicit_reset_strategy() -> None:
    """Repeated rounds should not silently treat an AWG reset as an ancilla reset."""
    with pytest.raises(ValueError, match="reset"):
        SurfaceCodeConfig(qubit_map=_qubit_map(), n_stabilizer_rounds=2)

    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_stabilizer_rounds=2,
        reset_strategy="passive",
        passive_reset_delay_ns=10_000.0,
    )
    assert config.n_stabilizer_rounds == 2


def test_bit_analysis_applies_inversion_detection_events_and_logical_parity() -> None:
    """Analysis should invert configured readout bits before syndrome and logical processing."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_stabilizer_rounds=2,
        reset_strategy="passive",
        passive_reset_delay_ns=10_000.0,
        n_shots=2,
        final_basis="Z",
        readout_inversion={"A1": True, "A2": False, "A3": True},
        data_readout_inversion={"D1": True, "D2": False, "D3": False, "D4": False},
    )
    raw_syndromes = np.array(
        [
            [[1, 0, 1], [0, 0, 1]],
            [[0, 1, 0], [0, 0, 1]],
        ],
        dtype=np.int8,
    )
    raw_data = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.int8,
    )

    analysis = analyze_bit_arrays(
        config=config,
        syndrome_bits=raw_syndromes,
        final_data_bits=raw_data,
    )

    np.testing.assert_array_equal(
        analysis.syndrome_bits,
        np.array(
            [
                [[0, 0, 0], [1, 0, 0]],
                [[1, 1, 1], [1, 0, 0]],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        analysis.detection_events,
        np.array([[[1, 0, 0]], [[0, 1, 1]]], dtype=np.int8),
    )
    np.testing.assert_array_equal(analysis.logical_bits, np.array([1, 1]))
    np.testing.assert_array_equal(analysis.source_shot_indices, np.array([0, 1]))
    assert analysis.frame_x.shape == (2, 2, len(DATA_QUBITS))
    assert analysis.frame_z.shape == (2, 2, len(DATA_QUBITS))
    assert not np.any(analysis.frame_x)
    assert not np.any(analysis.frame_z)


def test_pauli_frame_mode_records_the_configured_pure_error_frame() -> None:
    """Map corrected syndrome bits to the specified software Pauli frame."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_shots=1,
        syndrome_mode="pauli_frame",
    )

    analysis = analyze_bit_arrays(
        config=config,
        syndrome_bits=np.array([[[1, 1, 1]]], dtype=np.int8),
        final_data_bits=np.zeros((1, 4), dtype=np.int8),
    )

    np.testing.assert_array_equal(analysis.frame_x[0, 0], [0, 1, 0, 1])
    np.testing.assert_array_equal(analysis.frame_z[0, 0], [0, 0, 0, 1])


def test_postselection_marks_only_requested_syndrome_sector() -> None:
    """Postselection should retain shots satisfying every requested round syndrome."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_stabilizer_rounds=2,
        reset_strategy="passive",
        passive_reset_delay_ns=10_000.0,
        n_shots=2,
        syndrome_mode="postselect",
        postselect_syndrome=(0, 0, 0),
    )
    syndromes = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
        ],
        dtype=np.int8,
    )

    analysis = analyze_bit_arrays(
        config=config,
        syndrome_bits=syndromes,
        final_data_bits=np.zeros((2, 4), dtype=np.int8),
    )

    np.testing.assert_array_equal(analysis.accepted_shots, np.array([True, False]))


def test_initial_ground_postselection_requires_initial_bits() -> None:
    """Enabled ground-state heralding should reject analysis without herald bits."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_shots=1,
        initial_ground_postselection=True,
    )

    with pytest.raises(ValueError, match="initial_ground_bits"):
        analyze_bit_arrays(
            config=config,
            syndrome_bits=np.zeros((1, 1, 3), dtype=np.int8),
            final_data_bits=np.zeros((1, 4), dtype=np.int8),
        )


def test_initial_ground_and_syndrome_postselection_are_combined() -> None:
    """Apply inversion before AND-combining herald and syndrome acceptance."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_shots=3,
        syndrome_mode="postselect",
        initial_ground_postselection=True,
        readout_inversion={"A1": True},
        data_readout_inversion={"D2": True},
    )
    initial = np.array(
        [
            [0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0],
        ],
        dtype=np.int8,
    )
    syndrome = np.array(
        [
            [[1, 0, 0]],
            [[1, 0, 0]],
            [[1, 1, 0]],
        ],
        dtype=np.int8,
    )

    analysis = analyze_bit_arrays(
        config=config,
        initial_ground_bits=initial,
        syndrome_bits=syndrome,
        final_data_bits=np.zeros((3, 4), dtype=np.int8),
    )

    np.testing.assert_array_equal(
        analysis.initial_ground_bits,
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
    )
    np.testing.assert_array_equal(
        analysis.initial_ground_accepted_shots,
        np.array([True, False, True]),
    )
    np.testing.assert_array_equal(
        analysis.syndrome_accepted_shots,
        np.array([True, True, False]),
    )
    np.testing.assert_array_equal(
        analysis.accepted_shots,
        np.array([True, False, False]),
    )


def test_builder_adds_initial_ground_readout_before_the_surface_code() -> None:
    """Herald every qubit, recover, then use shifted capture indices."""
    recovery_delay_ns = 20.0
    built = build_serialized_d2_surface_code_schedule(
        _FakePulseAdapter(),
        SurfaceCodeConfig(
            qubit_map=_qubit_map(),
            n_shots=1,
            initial_ground_postselection=True,
            initial_readout_recovery_delay_ns=recovery_delay_ns,
        ),
    )

    initial = [
        capture for capture in built.captures if capture.role == "initial_ground"
    ]
    syndrome = [capture for capture in built.captures if capture.role == "syndrome"]
    final = [capture for capture in built.captures if capture.role == "final_data"]
    assert len(built.captures) == 14
    assert [capture.symbolic_qubit for capture in initial] == list(SYMBOLIC_QUBITS)
    assert all(capture.capture_index == 0 for capture in initial)
    assert all(capture.capture_index == 1 for capture in (*syndrome, *final))

    initial_labels = {capture.acquisition_label for capture in initial}
    initial_events = [
        event
        for event in built.timeline
        if event.kind == "READOUT" and event.acquisition_label in initial_labels
    ]
    recovery = [
        event
        for event in built.timeline
        if event.kind == "DELAY" and np.isclose(event.duration_ns, recovery_delay_ns)
    ]
    first_cx = next(event for event in built.timeline if event.kind == "CX_CR")
    assert len(initial_events) == len(SYMBOLIC_QUBITS)
    assert len(recovery) == 1
    assert max(event.end_ns for event in initial_events) <= recovery[0].start_ns
    assert recovery[0].end_ns <= first_cx.start_ns


def test_arbitrary_final_axes_rotate_only_x_and_y_data_qubits() -> None:
    """Mixed tomography axes should rotate X/Y only and disable logical parity."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_shots=1,
        final_measurement_axes=("X", "Y", "Z", "Z"),
    )
    built = build_serialized_d2_surface_code_schedule(_FakePulseAdapter(), config)
    rotations = [event for event in built.timeline if event.kind == "BASIS_ROTATION"]

    assert [
        (event.qubits, event.metadata["axis"], event.duration_ns) for event in rotations
    ] == [
        (("D1",), "X", 6.0),
        (("D2",), "Y", 4.0),
    ]
    analysis = analyze_bit_arrays(
        config=config,
        syndrome_bits=np.zeros((1, 1, 3), dtype=np.int8),
        final_data_bits=np.zeros((1, 4), dtype=np.int8),
    )
    np.testing.assert_array_equal(analysis.logical_bits, np.array([-1], dtype=np.int8))


@pytest.mark.parametrize(
    ("basis", "expected_axes", "expected_logical"),
    [
        ("Z", ("Z", "Z", "Z", "Z"), 1),
        ("X", ("X", "X", "X", "X"), 0),
    ],
)
def test_legacy_final_basis_controls_axes_and_logical_parity(
    basis: Basis,
    expected_axes: tuple[
        MeasurementAxis,
        MeasurementAxis,
        MeasurementAxis,
        MeasurementAxis,
    ],
    expected_logical: int,
) -> None:
    """Keep legacy uniform X/Z measurement and logical-parity behavior."""
    config = SurfaceCodeConfig(
        qubit_map=_qubit_map(),
        n_shots=1,
        final_basis=basis,
    )

    assert config.resolved_final_measurement_axes == expected_axes
    analysis = analyze_bit_arrays(
        config=config,
        syndrome_bits=np.zeros((1, 1, 3), dtype=np.int8),
        final_data_bits=np.array([[1, 1, 0, 0]], dtype=np.int8),
    )
    assert analysis.logical_bits[0] == expected_logical


def test_all_ideal_circuit_validations_pass() -> None:
    """The abstract serial circuit should pass parity, encoding, and error checks."""
    report = run_ideal_validations()

    assert report
    assert all(check.passed for check in report.values()), {
        name: check.detail for name, check in report.items() if not check.passed
    }

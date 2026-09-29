"""Tests for X90/Y90 Clifford sequence helpers."""

from __future__ import annotations

from qubex.clifford import Clifford, CliffordSequence
from qubex.contrib import create_xy_rb_sequences, generate_1q_xy_cliffords


def _compose_gate_names(gates: list[str]) -> CliffordSequence:
    """Compose a list of supported XY gate names."""
    sequence = CliffordSequence.I()
    for gate in gates:
        if gate == "X90":
            sequence = sequence.compose(Clifford.X90())
        elif gate == "Y90":
            sequence = sequence.compose(Clifford.Y90())
        else:
            raise AssertionError(f"Unexpected gate: {gate}")
    return sequence


def test_generate_1q_xy_cliffords_finds_all_24_without_virtual_z() -> None:
    """Given XY generators, then all representatives avoid Z90."""
    cliffords = generate_1q_xy_cliffords(max_gates=5)

    assert len(cliffords) == 24
    assert all(
        set(sequence.gate_sequence).issubset({"X90", "Y90"})
        for sequence in cliffords.values()
    )
    assert max(sequence.length for sequence in cliffords.values()) == 5
    assert all(clifford.inverse in cliffords for clifford in cliffords)


def test_create_xy_rb_sequences_appends_inverse_clifford() -> None:
    """Given seeded RB gates, then appending the inverse gives identity."""
    cliffords_a, inverse_a = create_xy_rb_sequences(20, seed=1234, max_gates=5)
    cliffords_b, inverse_b = create_xy_rb_sequences(20, seed=1234, max_gates=5)

    flattened = [gate for clifford in cliffords_a for gate in clifford] + inverse_a

    assert (cliffords_a, inverse_a) == (cliffords_b, inverse_b)
    assert set(flattened).issubset({"X90", "Y90"})
    assert _compose_gate_names(flattened).clifford.is_identity()

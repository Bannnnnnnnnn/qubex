"""Tests for simultaneous randomized benchmarking function APIs."""

from __future__ import annotations

from qubex.contrib import (
    create_xy_rb_sequences,
    generate_1q_xy_cliffords,
    simultaneous_randomized_benchmarking,
    simultaneous_xy_rb_sequence,
    xy_rb_sequence_1q,
)


def test_all_simultaneous_rb_functions_are_exported_from_contrib() -> None:
    """Given contrib package, then simultaneous RB helpers are available."""
    assert callable(generate_1q_xy_cliffords)
    assert callable(create_xy_rb_sequences)
    assert callable(xy_rb_sequence_1q)
    assert callable(simultaneous_xy_rb_sequence)
    assert callable(simultaneous_randomized_benchmarking)

"""Regression tests for pulse tomography semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from qxpulse import Arbitrary, PhaseShift, PulseArray

from qubex.experiment.services.measurement_service import MeasurementService


def test_partial_waveform_keeps_terminal_phase_shift_at_sample_boundary() -> None:
    """Partial waveform extraction should retain a terminal phase shift at the boundary."""
    service = cast(Any, object.__new__(MeasurementService))
    waveform = PulseArray([Arbitrary([1 + 0j]), PhaseShift(np.pi / 2)])

    partial = service.partial_waveform(waveform, 1)

    assert isinstance(partial, PulseArray)
    assert partial.values == pytest.approx([1 + 0j])
    assert partial.final_frame_shift == pytest.approx(np.pi / 2)

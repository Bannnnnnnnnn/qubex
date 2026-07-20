"""Tests for the JPA calibration functional API."""

from __future__ import annotations

from qubex.contrib import JPAConstraintError, calibrate_jpa
from qubex.contrib.experiment import (
    JPAConstraintError as ExperimentJPAConstraintError,
    calibrate_jpa as experiment_calibrate_jpa,
)
from qubex.experiment.models import Result


def test_calibrate_jpa_is_exported_from_contrib_namespaces() -> None:
    """The JPA calibration function should be available from both contrib namespaces."""
    assert callable(calibrate_jpa)
    assert experiment_calibrate_jpa is calibrate_jpa


def test_jpa_constraint_error_is_exported_and_retains_diagnostics() -> None:
    """The public constraint error should preserve completed scan diagnostics."""
    diagnostics = Result(data={"stage": "coarse"})

    error = JPAConstraintError("No acceptable JPA point.", diagnostics=diagnostics)

    assert ExperimentJPAConstraintError is JPAConstraintError
    assert isinstance(error, ValueError)
    assert error.diagnostics is diagnostics

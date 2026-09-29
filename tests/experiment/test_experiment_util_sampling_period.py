"""Tests for sampling-period resolution in experiment utilities."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from qubex.experiment.experiment_util import ExperimentUtil


def test_discretize_time_range_uses_backend_sampling_period_by_default(
    monkeypatch,
) -> None:
    """Given backend dt, when discretizing without dt, then utility uses backend sampling period."""
    backend_controller = type("_BackendController", (), {"sampling_period_ns": 0.4})()
    system_manager = type(
        "_SystemManager",
        (),
        {"backend_controller": backend_controller},
    )()
    monkeypatch.setattr(
        "qubex.experiment.experiment_util.SystemManager.shared",
        lambda: system_manager,
    )

    discretized = ExperimentUtil.discretize_time_range(
        np.array([0.21, 0.61], dtype=float),
    )

    assert np.allclose(discretized, np.array([0.4, 0.8]))


def test_create_qubit_subgroups_preserves_custom_target_aliases(monkeypatch) -> None:
    """Given a custom target alias, subgrouping should use physical qubit index but return the alias."""

    class _ExperimentSystem:
        @staticmethod
        def resolve_qubit_label(label: str) -> str:
            return "Q28" if label == "Q28_tmp" else label

        @staticmethod
        def get_qubit(label: str) -> SimpleNamespace:
            qubits = {
                "Q25": SimpleNamespace(label="Q25", index=25),
                "Q28": SimpleNamespace(label="Q28", index=28),
            }
            return qubits[label]

    system_manager = SimpleNamespace(experiment_system=_ExperimentSystem())
    monkeypatch.setattr(
        "qubex.experiment.experiment_util.SystemManager.shared",
        lambda: system_manager,
    )

    subgroups = ExperimentUtil.create_qubit_subgroups(["Q25", "Q28_tmp"])

    assert subgroups == [["Q28_tmp"], ["Q25"]]

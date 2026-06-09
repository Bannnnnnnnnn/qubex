"""Tests for the temporary S159A five-channel control profile."""

from __future__ import annotations

from qubex.system.control_system import Box, BoxType
from qubex.system.quantum_system import Qubit
from qubex.system.quel1.quel1_port_configurator import create_control_configuration


def _make_box(box_id: str) -> Box:
    return Box.new(
        id=box_id,
        name=box_id,
        type=BoxType.QUEL1SE_A,
        address="10.0.0.2",
        adapter="dummy",
        port_numbers=[2, 4],
    )


def _make_qubit(label: str, frequency: float) -> Qubit:
    qubit = Qubit(index=0, label=label, chip_id="chip", resonator=f"R{label[1:]}")
    qubit.control_frequency_ge_value = frequency
    qubit.anharmonicity_value = -0.3
    return qubit


def test_s159a_overrides_mxfe0_control_channel_counts() -> None:
    """Given S159A, when ports are initialized, then port 2 has five channels."""
    s159a = _make_box("S159A")
    other = _make_box("S160A")

    assert s159a.get_port(2).n_channels == 5
    assert s159a.get_port(4).n_channels == 1
    assert other.get_port(2).n_channels == 3
    assert other.get_port(4).n_channels == 3


def test_five_channel_ge_cr_cr_assigns_extra_cr_channels() -> None:
    """Given five control channels, when GE/CR mode is used, then all channels are configured."""
    qubit = _make_qubit("Q24", 7.0)
    spectators = [
        _make_qubit("Q25", 6.7),
        _make_qubit("Q26", 7.2),
        _make_qubit("Q27", 7.5),
    ]

    config = create_control_configuration(
        mode="ge-cr-cr",
        qubit=qubit,
        n_channels=5,
        get_spectator_qubits=lambda _: spectators,
        excluded_targets=(),
    )

    assert sorted(config["channels"]) == [0, 1, 2, 3, 4]
    assert config["channels"][0]["targets"] == ["Q24"]
    assert config["channels"][1]["targets"] == ["Q24-CR", "Q24-Q25"]
    assert config["channels"][2]["targets"] == ["Q24-Q26"]
    assert config["channels"][3]["targets"] == ["Q24-Q27"]
    assert config["channels"][4]["targets"] == []

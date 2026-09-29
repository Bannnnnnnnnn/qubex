"""Tests for control system option-driven channel mapping."""

from __future__ import annotations

import pytest

from qubex.system.control_system import Box, CapPort, GenPort


def _control_channel_counts(box: Box) -> list[int]:
    """Return channel counts for R8 control ports 6-9."""
    return [len(box.get_port(port_num).channels) for port_num in (6, 7, 8, 9)]


def _port_channel_counts(box: Box, port_numbers: tuple[int, ...]) -> tuple[int, ...]:
    """Return channel counts for selected ports."""
    return tuple(len(box.get_port(port_num).channels) for port_num in port_numbers)


def test_r8_box_uses_default_awg2222_when_options_omitted() -> None:
    """Given R8 box without options, when building ports, then control channels are 2-2-2-2."""
    box = Box.new(
        id="B0",
        name="R8",
        type="quel1se-riken8",
        address="192.0.2.10",
        adapter="A0",
    )

    assert _control_channel_counts(box) == [2, 2, 2, 2]


def test_r8_box_applies_awg1331_option_to_control_channels() -> None:
    """Given R8 box with awg1331 option, when building ports, then control channels are 1-3-3-1."""
    box = Box.new(
        id="B0",
        name="R8",
        type="quel1se-riken8",
        address="192.0.2.10",
        adapter="A0",
        options=("se8_mxfe1_awg1331",),
    )

    assert _control_channel_counts(box) == [1, 3, 3, 1]


@pytest.mark.parametrize(
    ("box_type", "group0_ports", "group1_ports"),
    [
        ("quel1-a", (0, 1, 2), (7, 8, 11)),
        ("quel1se-fujitsu11-a", (0, 1, 2), (7, 8, 9)),
        ("qube-riken-a", (1, 0, 5), (12, 13, 8)),
        ("qube-ou-a", (1, 0, 5), (12, 13, 8)),
    ],
)
def test_dual_readout_group0_updates_type_a_port_channels(
    box_type: str,
    group0_ports: tuple[int, int, int],
    group1_ports: tuple[int, int, int],
) -> None:
    """Given dual-readout group0 option, when building Type-A ports, then group0 channel counts change."""
    box = Box.new(
        id="B0",
        name="Box",
        type=box_type,
        address="192.0.2.10",
        adapter="A0",
        options=("dual_readout_group0",),
    )

    assert _port_channel_counts(box, group0_ports) == (5, 2, 2)
    assert _port_channel_counts(box, group1_ports) == (4, 1, 3)


@pytest.mark.parametrize(
    ("box_type", "group0_ports", "group1_ports"),
    [
        ("quel1-a", (0, 1, 2), (7, 8, 11)),
        ("quel1se-fujitsu11-a", (0, 1, 2), (7, 8, 9)),
        ("qube-riken-a", (1, 0, 5), (12, 13, 8)),
        ("qube-ou-a", (1, 0, 5), (12, 13, 8)),
    ],
)
def test_dual_readout_group1_updates_type_a_port_channels(
    box_type: str,
    group0_ports: tuple[int, int, int],
    group1_ports: tuple[int, int, int],
) -> None:
    """Given dual-readout group1 option, when building Type-A ports, then group1 channel counts change."""
    box = Box.new(
        id="B0",
        name="Box",
        type=box_type,
        address="192.0.2.10",
        adapter="A0",
        options=("dual_readout_group1",),
    )

    assert _port_channel_counts(box, group0_ports) == (4, 1, 3)
    assert _port_channel_counts(box, group1_ports) == (5, 2, 2)


@pytest.mark.parametrize(
    ("box_type", "group0_ports", "group1_ports"),
    [
        ("quel1-a", (0, 1, 2), (7, 8, 11)),
        ("quel1se-fujitsu11-a", (0, 1, 2), (7, 8, 9)),
        ("qube-riken-a", (1, 0, 5), (12, 13, 8)),
        ("qube-ou-a", (1, 0, 5), (12, 13, 8)),
    ],
)
def test_dual_readout_both_groups_update_type_a_port_channels(
    box_type: str,
    group0_ports: tuple[int, int, int],
    group1_ports: tuple[int, int, int],
) -> None:
    """Given both dual-readout options, when building Type-A ports, then both groups change."""
    box = Box.new(
        id="B0",
        name="Box",
        type=box_type,
        address="192.0.2.10",
        adapter="A0",
        options=("dual_readout_group0", "dual_readout_group1"),
    )

    assert _port_channel_counts(box, group0_ports) == (5, 2, 2)
    assert _port_channel_counts(box, group1_ports) == (5, 2, 2)


@pytest.mark.parametrize(
    "box_type",
    [
        "quel1-b",
        "qube-riken-b",
        "qube-ou-b",
        "quel1se-riken8",
        "quel1se-fujitsu11-b",
    ],
)
def test_dual_readout_options_reject_unsupported_box_types(box_type: str) -> None:
    """Given dual-readout option on unsupported box type, when building ports, then ValueError is raised."""
    with pytest.raises(ValueError, match="Dual-readout options are supported only"):
        Box.new(
            id="B0",
            name="Box",
            type=box_type,
            address="192.0.2.10",
            adapter="A0",
            options=("dual_readout_group0",),
        )


def test_r8_box_rejects_multiple_awg_options() -> None:
    """Given R8 box with conflicting awg options, when building ports, then ValueError is raised."""
    with pytest.raises(ValueError, match="Multiple AWG options are not allowed"):
        Box.new(
            id="B0",
            name="R8",
            type="quel1se-riken8",
            address="192.0.2.10",
            adapter="A0",
            options=("se8_mxfe1_awg1331", "se8_mxfe1_awg2222"),
        )


def test_box_traits_for_r8_reflect_direct_nco_control() -> None:
    """Given R8 box, when reading traits, then control and readout traits match R8 behavior."""
    box = Box.new(
        id="B0",
        name="R8",
        type="quel1se-riken8",
        address="192.0.2.10",
        adapter="A0",
    )

    assert box.traits.ctrl_ssb is None
    assert box.traits.readout_ssb == "L"
    assert box.traits.default_control_frequency_range == (3.0, 5.0, 0.005)


def test_box_traits_for_non_r8_keep_legacy_defaults() -> None:
    """Given non-R8 box, when reading traits, then legacy LO/SSB defaults are preserved."""
    box = Box.new(
        id="B1",
        name="Q1",
        type="quel1-a",
        address="192.0.2.11",
        adapter="A1",
    )

    assert box.traits.ctrl_ssb == "L"
    assert box.traits.readout_ssb == "U"
    assert box.traits.default_control_frequency_range == (6.5, 9.5, 0.005)


def test_new_ports_start_with_unset_lo_and_cnco() -> None:
    """Given a new box, when ports are initialized, then LO and CNCO start as unset."""
    box = Box.new(
        id="B2",
        name="Q2",
        type="quel1-a",
        address="192.0.2.12",
        adapter="A2",
        port_numbers=[0, 1, 2],
    )

    for port in box.ports:
        if isinstance(port, GenPort | CapPort):
            assert port.lo_freq is None
            assert port.cnco_freq is None
        if isinstance(port, CapPort):
            assert port.rfswitch is None
        if isinstance(port, GenPort):
            assert port.vatt is None
            assert port.fullscale_current is None
            assert port.rfswitch is None

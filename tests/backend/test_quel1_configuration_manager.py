"""Tests for QuEL-1 configuration manager behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest

from qubex.backend.quel1.managers.configuration_manager import Quel1ConfigurationManager
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

Port = int | tuple[int, int]


class _FakeBox:
    def __init__(
        self,
        *,
        boxtype: str = "quel1se-riken8",
        input_ports: tuple[Port, ...] = (),
    ) -> None:
        self.boxtype = boxtype
        self._input_ports = input_ports
        self.config_port_calls: list[dict[str, Any]] = []
        self.config_channel_calls: list[dict[str, Any]] = []
        self.config_runit_calls: list[dict[str, Any]] = []
        self.dump_port_calls: list[Port] = []

    def get_input_ports(self) -> tuple[Port, ...]:
        """Return fixed input ports."""
        return self._input_ports

    def config_port(self, **kwargs: Any) -> None:
        """Record config_port kwargs."""
        self.config_port_calls.append(kwargs)

    def config_channel(self, **kwargs: Any) -> None:
        """Record config_channel kwargs."""
        self.config_channel_calls.append(kwargs)

    def config_runit(self, **kwargs: Any) -> None:
        """Record config_runit kwargs."""
        self.config_runit_calls.append(kwargs)

    def dump_port(self, port: Port) -> dict[str, Any]:
        """Record dump_port calls."""
        self.dump_port_calls.append(port)
        return {}


class _FakeAd9082:
    def __init__(self) -> None:
        self.adc_cnco_calls: list[tuple[set[int], str]] = []

    def set_adc_cnco(self, adc_indices: set[int], ftw: str) -> None:
        """Record selected ADC CNCO writes."""
        self.adc_cnco_calls.append((set(adc_indices), ftw))


class _FakeDualReadoutCss:
    _DUAL_READOUT_SECONDARY_CDDC = 1

    def __init__(self) -> None:
        self.ad9082 = {0: _FakeAd9082()}

    def get_adc_idx(self, group: int, rline: str) -> tuple[int, int]:
        """Return the primary CDDC."""
        assert (group, rline) == (1, "r")
        return 0, 3

    def is_dual_readout_mode_enabled(self, group: int, rline: str) -> bool:
        """Return dual-readout status."""
        assert (group, rline) == (1, "r")
        return True

    def _validate_frequency_info(
        self,
        mxfe_idx: int,
        freq_type: str,
        freq_in_hz: int,
        ftw: None,
    ) -> tuple[int, str]:
        """Return a fake FTW for assertions."""
        assert mxfe_idx == 0
        assert freq_type == "adc_cnco"
        assert ftw is None
        return freq_in_hz, f"ftw:{freq_in_hz}"


class _FakeDualReadoutDev:
    def _get_rchannel_from_runit(self, group: int, rline: str, runit: int) -> int:
        """Map the 3+1 dual-readout runits to two rchannels."""
        assert (group, rline) == (1, "r")
        return 1 if runit == 4 else 0


class _FakeDualReadoutBox(_FakeBox):
    def __init__(self) -> None:
        super().__init__(boxtype="quel1-a")
        self.css = _FakeDualReadoutCss()
        self._dev = _FakeDualReadoutDev()

    def _convert_input_port(self, port: Port) -> tuple[int, str]:
        """Return group/rline for the dual-readout read-in port."""
        assert port == 7
        return 1, "r"


class _BoxPoolStub:
    def __init__(self, box: _FakeBox) -> None:
        self._boxes = {"B0": (box,)}


class _RuntimeContextStub:
    is_connected = True

    def __init__(self, box: _FakeBox) -> None:
        self.boxpool = _BoxPoolStub(box)
        self.validated_box_names: list[str] = []

    def validate_box_availability(self, box_name: str) -> None:
        """Record box availability checks."""
        self.validated_box_names.append(box_name)


def _configure_r8_port(
    *,
    port: Port,
    input_ports: tuple[Port, ...] = (),
) -> tuple[_RuntimeContextStub, _FakeBox]:
    box = _FakeBox(input_ports=input_ports)
    runtime_context = _RuntimeContextStub(box)
    manager = Quel1ConfigurationManager(
        runtime_context=cast(Quel1RuntimeContext, runtime_context)
    )

    manager.config_port(
        box_name="B0",
        port=port,
        lo_freq_hz=5_000_000_000,
        cnco_freq_hz=100_000_000,
        vatt=2048,
        sideband="U",
        fullscale_current=16383,
        rfswitch="pass",
    )

    return runtime_context, box


@pytest.mark.parametrize(
    (
        "port",
        "input_ports",
        "expected_lo_freq",
        "expected_vatt",
        "expected_sideband",
    ),
    [
        (1, (), 5_000_000_000, 2048, "U"),
        (0, (0,), 5_000_000_000, None, None),
        (3, (), None, None, None),
    ],
)
def test_r8_config_port_filters_only_unsupported_mixer_fields(
    port: Port,
    input_ports: tuple[Port, ...],
    expected_lo_freq: int | None,
    expected_vatt: int | None,
    expected_sideband: str | None,
) -> None:
    """Given R8 port traits, when configuring a port, then unsupported mixer fields are dropped."""
    runtime_context, box = _configure_r8_port(port=port, input_ports=input_ports)

    assert runtime_context.validated_box_names == ["B0"]
    assert box.dump_port_calls == []
    assert box.config_port_calls == [
        {
            "port": port,
            "lo_freq": expected_lo_freq,
            "cnco_freq": 100_000_000,
            "vatt": expected_vatt,
            "sideband": expected_sideband,
            "fullscale_current": 16383,
            "rfswitch": "pass",
        }
    ]


def test_config_runit_sets_dual_readout_secondary_adc_cnco() -> None:
    """Given a 3+1 dual-readout runit, when syncing CNCO, then CDDC1 is configured."""
    box = _FakeDualReadoutBox()
    runtime_context = _RuntimeContextStub(box)
    manager = Quel1ConfigurationManager(
        runtime_context=cast(Quel1RuntimeContext, runtime_context)
    )

    manager.config_runit(
        box_name="B0",
        port=7,
        runit=4,
        cnco_freq_hz=1_078_125_000,
        fnco_freq_hz=0,
    )

    assert box.css.ad9082[0].adc_cnco_calls == [
        ({1}, "ftw:1078125000"),
    ]
    assert box.config_runit_calls == [
        {"port": 7, "runit": 4, "fnco_freq": 0},
    ]

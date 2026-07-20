"""Tests for temporary DC voltage control cleanup."""

from __future__ import annotations

import pytest

from qubex.backend import dc_voltage_controller


class _FakeSupply:
    """Record DC supply operations and inject selected failures."""

    def __init__(self, original_voltages: dict[int, float]) -> None:
        self.original_voltages = original_voltages
        self.calls: list[tuple[object, ...]] = []
        self.get_errors: dict[int, BaseException] = {}
        self.restore_errors: dict[int, BaseException] = {}
        self.on_errors: dict[int, BaseException] = {}
        self.off_errors: dict[int, BaseException] = {}
        self.close_error: BaseException | None = None

    def get_voltage(self, channel: int) -> float:
        """Return one original voltage."""
        self.calls.append(("get_voltage", channel))
        error = self.get_errors.get(channel)
        if error is not None:
            raise error
        return self.original_voltages[channel]

    def set_voltage(self, channel: int, voltage: float) -> None:
        """Record one voltage update and optionally fail during restoration."""
        self.calls.append(("set_voltage", channel, voltage))
        error = self.restore_errors.get(channel)
        if error is not None and voltage == self.original_voltages[channel]:
            raise error

    def on(self, channel: int) -> None:
        """Record one output enable and optionally fail."""
        self.calls.append(("on", channel))
        error = self.on_errors.get(channel)
        if error is not None:
            raise error

    def off(self, channel: int) -> None:
        """Record one output disable and optionally fail."""
        self.calls.append(("off", channel))
        error = self.off_errors.get(channel)
        if error is not None:
            raise error

    def close(self) -> None:
        """Record connection cleanup and optionally fail."""
        self.calls.append(("close",))
        if self.close_error is not None:
            raise self.close_error


def _use_fake_supply(
    monkeypatch: pytest.MonkeyPatch,
    supply: _FakeSupply,
) -> None:
    """Install one fake ONS61797 constructor."""

    def create_supply(*, port: str) -> _FakeSupply:
        assert port == dc_voltage_controller.PORT
        return supply

    monkeypatch.setattr(dc_voltage_controller, "ONS61797", create_supply)


def test_dc_voltage_restores_disables_and_closes_every_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal context exit should restore, disable, and close all channels."""
    supply = _FakeSupply({1: 0.1, 2: 0.2})
    _use_fake_supply(monkeypatch, supply)

    with dc_voltage_controller.dc_voltage({1: 1.1, 2: 1.2}) as yielded:
        assert yielded is supply

    assert ("set_voltage", 1, 0.1) in supply.calls
    assert ("set_voltage", 2, 0.2) in supply.calls
    assert ("off", 1) in supply.calls
    assert ("off", 2) in supply.calls
    assert supply.calls[-1] == ("close",)


def test_dc_voltage_preserves_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A constructor failure should propagate without unbound-local masking."""
    connection_error = OSError("connection failed")

    def fail_to_connect(*, port: str) -> None:
        assert port == dc_voltage_controller.PORT
        raise connection_error

    monkeypatch.setattr(dc_voltage_controller, "ONS61797", fail_to_connect)

    with (
        pytest.raises(OSError, match="connection failed") as exc_info,
        dc_voltage_controller.dc_voltage({1: 1.0}),
    ):
        pass

    assert exc_info.value is connection_error


def test_dc_voltage_preserves_partial_setup_failure_and_runs_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A setup failure should win while restoration, disable, and close are attempted."""
    setup_error = RuntimeError("enable failed")
    supply = _FakeSupply({1: 0.1, 2: 0.2})
    supply.on_errors[1] = setup_error
    supply.restore_errors[1] = RuntimeError("restore failed")
    supply.off_errors[1] = RuntimeError("disable failed")
    supply.close_error = RuntimeError("close failed")
    _use_fake_supply(monkeypatch, supply)

    with (
        pytest.raises(RuntimeError, match="enable failed") as exc_info,
        dc_voltage_controller.dc_voltage({1: 1.1, 2: 1.2}),
    ):
        pass

    assert exc_info.value is setup_error
    assert ("set_voltage", 1, 0.1) in supply.calls
    assert ("off", 1) in supply.calls
    assert ("get_voltage", 2) not in supply.calls
    assert supply.calls[-1] == ("close",)


def test_dc_voltage_disables_touched_channel_when_voltage_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read failure should still disable the touched channel and close."""
    read_error = RuntimeError("read failed")
    supply = _FakeSupply({1: 0.1})
    supply.get_errors[1] = read_error
    _use_fake_supply(monkeypatch, supply)

    with (
        pytest.raises(RuntimeError, match="read failed") as exc_info,
        dc_voltage_controller.dc_voltage({1: 1.1}),
    ):
        pass

    assert exc_info.value is read_error
    assert ("set_voltage", 1, 0.1) not in supply.calls
    assert ("off", 1) in supply.calls
    assert supply.calls[-1] == ("close",)


def test_dc_voltage_preserves_body_failure_and_completes_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A body failure should win while every later cleanup operation is attempted."""
    body_error = RuntimeError("measurement failed")
    supply = _FakeSupply({1: 0.1, 2: 0.2})
    supply.restore_errors[1] = RuntimeError("restore failed")
    supply.off_errors[1] = RuntimeError("disable failed")
    supply.close_error = RuntimeError("close failed")
    _use_fake_supply(monkeypatch, supply)

    with (
        pytest.raises(RuntimeError, match="measurement failed") as exc_info,
        dc_voltage_controller.dc_voltage({1: 1.1, 2: 1.2}),
    ):
        raise body_error

    assert exc_info.value is body_error
    assert ("set_voltage", 2, 0.2) in supply.calls
    assert ("off", 1) in supply.calls
    assert ("off", 2) in supply.calls
    assert supply.calls[-1] == ("close",)


def test_dc_voltage_surfaces_first_cleanup_failure_after_normal_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal body should surface cleanup failure after all cleanup is attempted."""
    restore_error = RuntimeError("restore failed")
    supply = _FakeSupply({1: 0.1, 2: 0.2})
    supply.restore_errors[1] = restore_error
    supply.off_errors[1] = RuntimeError("disable failed")
    supply.close_error = RuntimeError("close failed")
    _use_fake_supply(monkeypatch, supply)

    with (
        pytest.raises(RuntimeError, match="restore failed") as exc_info,
        dc_voltage_controller.dc_voltage({1: 1.1, 2: 1.2}),
    ):
        pass

    assert exc_info.value is restore_error
    assert ("set_voltage", 2, 0.2) in supply.calls
    assert ("off", 1) in supply.calls
    assert ("off", 2) in supply.calls
    assert supply.calls[-1] == ("close",)

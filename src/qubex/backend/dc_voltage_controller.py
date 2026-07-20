"""DC voltage control helpers for the ONS61797 device."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Final

from qubex.third_party.ons61797 import ONS61797

# TODO: Make port configurable
PORT: Final = "/dev/ttyACM0"
logger = logging.getLogger(__name__)


@contextmanager
def dc_voltage(voltages: dict[int, float]) -> Iterator[ONS61797]:
    """
    Temporarily apply DC voltages and restore originals on exit.

    Notes
    -----
    Cleanup attempts voltage restoration, output disable, and connection close
    independently. A setup or context-body exception takes precedence over
    cleanup errors; otherwise, the first cleanup error is raised.
    """
    ons61797: ONS61797 | None = None
    original_voltages: dict[int, float] = {}
    touched_channels: list[int] = []
    primary_error: BaseException | None = None
    try:
        ons61797 = ONS61797(port=PORT)
        for channel, voltage in voltages.items():
            touched_channels.append(channel)
            original_voltages[channel] = ons61797.get_voltage(channel)
            ons61797.set_voltage(channel, voltage)
            ons61797.on(channel)
        yield ons61797
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[tuple[str, BaseException]] = []
        if ons61797 is not None:
            for channel in touched_channels:
                if channel in original_voltages:
                    try:
                        ons61797.set_voltage(
                            channel,
                            original_voltages[channel],
                        )
                    except BaseException as exc:
                        cleanup_errors.append(
                            (f"restoring channel {channel} voltage", exc)
                        )
                try:
                    ons61797.off(channel)
                except BaseException as exc:
                    cleanup_errors.append((f"disabling channel {channel}", exc))
            try:
                ons61797.close()
            except BaseException as exc:
                cleanup_errors.append(("closing the DC supply connection", exc))

        if cleanup_errors:
            if primary_error is not None:
                errors_to_log = cleanup_errors
            else:
                errors_to_log = cleanup_errors[1:]
            for operation, error in errors_to_log:
                logger.error(
                    "DC voltage cleanup failed while %s.",
                    operation,
                    exc_info=(type(error), error, error.__traceback__),
                )
            if primary_error is None:
                raise cleanup_errors[0][1]


def with_connection(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap calls with a temporary ONS61797 connection."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            if self._ons61797 is None:
                self._ons61797 = ONS61797(port=PORT)
            else:
                self._ons61797.connect(port=PORT)
            return func(self, *args, **kwargs)
        finally:
            if self._ons61797 is not None:
                self._ons61797.close()

    return wrapper


class DCVoltageController:
    """Singleton controller for DC voltage device access."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def shared(cls) -> DCVoltageController:
        """Return the shared controller instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the controller if not already initialized."""
        if self._initialized:
            return
        self._ons61797: ONS61797 | None = None
        self._initialized = True

    def __del__(self):
        """Close the device connection on deletion."""
        if self._ons61797 is not None:
            self._ons61797.close()

    @property
    def ons61797(self) -> ONS61797:
        """Return the active device connection."""
        if self._ons61797 is None:
            raise RuntimeError("No connection established.")
        return self._ons61797

    @with_connection
    def on(self, channel: int) -> None:
        """Turn on the specified output channel."""
        self.ons61797.on(channel=channel)

    @with_connection
    def off(self, channel: int) -> None:
        """Turn off the specified output channel."""
        self.ons61797.off(channel=channel)

    @with_connection
    def get_output_state(self, channel: int) -> int:
        """Get the output state of the specified channel."""
        return self.ons61797.get_output_state(channel=channel)

    @with_connection
    def set_voltage(self, channel: int, voltage: float) -> None:
        """Set the voltage for the specified channel."""
        self.ons61797.set_voltage(channel=channel, voltage=voltage)

    @with_connection
    def get_voltage(self, channel: int) -> float:
        """Get the voltage for the specified channel."""
        return self.ons61797.get_voltage(channel=channel)

    @with_connection
    def get_device_information(self) -> str:
        """Return device information from the controller."""
        return self.ons61797.get_device_information()

    @with_connection
    def reset(self) -> None:
        """Reset the device settings."""
        self.ons61797.reset()

    @contextmanager
    def connection(self) -> Iterator[ONS61797]:
        """Yield a connected device and close on exit."""
        try:
            if self._ons61797 is None:
                self._ons61797 = ONS61797(port=PORT)
            else:
                self._ons61797.connect(port=PORT)
            yield self._ons61797
        finally:
            if self._ons61797 is not None:
                self._ons61797.close()

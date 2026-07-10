# ruff: noqa: SLF001

"""Configuration and box-dump manager for QuEL-1 backend controller."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qubex.backend.quel1.quel1_backend_constants import (
    DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT,
)
from qubex.backend.quel1.quel1_runtime_context import Quel1RuntimeContext

from .option_resolver import resolve_config_options

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from qubex.backend.quel1.compat.qubecalib_protocols import (
        Quel1BoxCommonProtocol as Quel1Box,
        SequencerProtocol as Sequencer,
    )


class Quel1ConfigurationManager:
    """Handle configuration, define, and dump operations for QuEL-1."""

    def __init__(self, *, runtime_context: Quel1RuntimeContext) -> None:
        self._runtime_context = runtime_context

    def set_box_options(self, box_options: dict[str, tuple[str, ...]]) -> None:
        """Set per-box relink option labels."""
        self._runtime_context.set_box_options(box_options)

    def dump_box(self, *, box_name: str) -> dict:
        """Dump one box configuration and tolerate box-level errors."""
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
            )
            return box.dump_box()
        except Exception:
            logger.exception(f"Failed to dump box {box_name}.")
            return {}

    def dump_port(
        self,
        *,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> dict:
        """Dump one port configuration and tolerate box-level errors."""
        try:
            box = self._resolve_box(
                box_name=box_name,
                reconnect=True,
            )
            return box.dump_port(port_number)
        except Exception:
            logger.exception(f"Failed to dump port {port_number} of box {box_name}.")
            return {}

    def config_port(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        lo_freq_hz: int | None,
        cnco_freq_hz: int | None,
        vatt: int | None,
        sideband: str | None,
        fullscale_current: int | None,
        rfswitch: str | None,
    ) -> None:
        """Configure one box port."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        if box.boxtype == "quel1se-riken8":
            QUEL1SE_R8_MIXER_PORTS = frozenset({1, 2})
            input_ports = box.get_input_ports()
            if port not in (set(input_ports) | QUEL1SE_R8_MIXER_PORTS):
                lo_freq_hz = None
            if port not in QUEL1SE_R8_MIXER_PORTS:
                vatt = None
                sideband = None
        box.config_port(
            port=port,
            lo_freq=lo_freq_hz,
            cnco_freq=cnco_freq_hz,
            vatt=vatt,
            sideband=sideband,
            fullscale_current=fullscale_current,
            rfswitch=rfswitch,
        )

    def config_channel(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        channel: int,
        cnco_freq_hz: int | None,
        fnco_freq_hz: int | None,
    ) -> None:
        """Configure one box channel."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        if cnco_freq_hz is not None:
            self._config_channel_cnco(
                box=box,
                port=port,
                channel=channel,
                cnco_freq_hz=cnco_freq_hz,
            )
        box.config_channel(
            port=port,
            channel=channel,
            fnco_freq=fnco_freq_hz,
        )

    def config_runit(
        self,
        *,
        box_name: str,
        port: int | tuple[int, int],
        runit: int,
        cnco_freq_hz: int | None,
        fnco_freq_hz: int | None,
    ) -> None:
        """Configure one box runit."""
        box = self._resolve_box(
            box_name=box_name,
            reconnect=True,
        )
        if cnco_freq_hz is not None:
            self._config_runit_cnco(
                box=box,
                port=port,
                runit=runit,
                cnco_freq_hz=cnco_freq_hz,
            )
        box.config_runit(
            port=port,
            runit=runit,
            fnco_freq=fnco_freq_hz,
        )

    def define_clockmaster(
        self,
        *,
        ipaddr: str,
    ) -> None:
        """Define clockmaster in qubecalib."""
        self._runtime_context.qubecalib.define_clockmaster(
            ipaddr=ipaddr,
            reset=False,
        )

    def define_box(
        self,
        *,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
    ) -> None:
        """Define one box in qubecalib."""
        option_map = self._runtime_context.driver.Quel1ConfigOption._value2member_map_
        config_options = resolve_config_options(
            option_map=option_map,
            box_name=box_name,
            option_labels=self._runtime_context.box_options.get(box_name, ()),
        )
        define_box_kwargs: dict[str, object] = {
            "box_name": box_name,
            "ipaddr_wss": ipaddr_wss,
            "boxtype": boxtype,
        }
        if config_options is not None:
            define_box_kwargs["config_options"] = config_options
        self._runtime_context.qubecalib.define_box(**define_box_kwargs)

    def define_port(
        self,
        *,
        port_name: str,
        box_name: str,
        port_number: int | tuple[int, int],
    ) -> None:
        """Define one port in qubecalib."""
        self._runtime_context.qubecalib.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
        )

    def define_channel(
        self,
        *,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        """Define one channel in qubecalib."""
        self._runtime_context.qubecalib.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def add_channel_target_relation(
        self,
        *,
        channel_name: str,
        target_name: str,
    ) -> None:
        """Add one channel-target relation if missing."""
        relation = (channel_name, target_name)
        sysdb = self._runtime_context.qubecalib.sysdb
        if relation not in sysdb._relation_channel_target:
            sysdb._relation_channel_target.append(relation)

    def define_target(
        self,
        *,
        target_name: str,
        channel_name: str,
        target_frequency_ghz: float | None = None,
    ) -> None:
        """Define one target in qubecalib."""
        self._runtime_context.qubecalib.define_target(
            target_name=target_name,
            channel_name=channel_name,
            target_frequency=target_frequency_ghz,
        )

    def modify_target_frequency(
        self,
        *,
        target: str,
        frequency_ghz: float,
    ) -> None:
        """Modify one target frequency in qubecalib."""
        self._runtime_context.qubecalib.modify_target_frequency(
            target,
            frequency_ghz,
        )

    def modify_target_frequencies(
        self,
        *,
        target_frequencies_ghz: dict[str, float],
    ) -> None:
        """Modify multiple target frequencies in qubecalib."""
        for target, frequency_ghz in target_frequencies_ghz.items():
            self.modify_target_frequency(
                target=target,
                frequency_ghz=frequency_ghz,
            )

    def add_sequencer(self, *, sequencer: Sequencer) -> None:
        """Add one sequencer into qubecalib executor queue."""
        self._runtime_context.qubecalib._executor.add_command(sequencer)

    def show_command_queue(self) -> str:
        """Return current qubecalib command queue string."""
        return self._runtime_context.qubecalib.show_command_queue()

    def clear_command_queue(self) -> None:
        """Clear qubecalib command queue."""
        self._runtime_context.qubecalib.clear_command_queue()

    def _config_channel_cnco(
        self,
        *,
        box: Quel1Box,
        port: int | tuple[int, int],
        channel: int,
        cnco_freq_hz: int,
    ) -> None:
        """Apply channel CNCO when the underlying DAC CNCO is uniquely addressable."""
        css = getattr(box, "css", None)
        convert_output_port = getattr(box, "_convert_output_port", None)
        if css is None or not callable(convert_output_port):
            return
        try:
            group, line = convert_output_port(port)
            mxfe_idx, fduc_idx = css.get_fduc_idx(group, line, channel)
            dac_idx = self._find_dac_for_fduc(css=css, mxfe_idx=mxfe_idx, fduc_idx=fduc_idx)
        except Exception:
            logger.debug(
                "Could not resolve DAC CNCO for port %s channel %s; skipping channel CNCO.",
                port,
                channel,
                exc_info=True,
            )
            return

        fducs_on_dac = tuple(css.ad9082[mxfe_idx].get_fduc_of_dac(dac_idx))
        if len(fducs_on_dac) > 1:
            logger.debug(
                "Skipping channel CNCO for port %s channel %s because MxFE%s DAC-CNCO%s "
                "is shared by FDUCs %s.",
                port,
                channel,
                mxfe_idx,
                dac_idx,
                fducs_on_dac,
            )
            return

        self._set_dac_cnco_indices(
            css=css,
            mxfe_idx=mxfe_idx,
            dac_indices={dac_idx},
            cnco_freq_hz=cnco_freq_hz,
        )

    def _config_runit_cnco(
        self,
        *,
        box: Quel1Box,
        port: int | tuple[int, int],
        runit: int,
        cnco_freq_hz: int,
    ) -> None:
        """Apply runit CNCO, including the secondary CDDC in dual-readout mode."""
        css = getattr(box, "css", None)
        dev = getattr(box, "_dev", None)
        convert_input_port = getattr(box, "_convert_input_port", None)
        if css is None or dev is None or not callable(convert_input_port):
            return
        try:
            group, rline = convert_input_port(port)
            rchannel = dev._get_rchannel_from_runit(group, rline, runit)
            mxfe_idx, adc_idx = self._resolve_adc_cnco_index(
                css=css,
                group=group,
                rline=rline,
                rchannel=rchannel,
            )
        except Exception:
            logger.debug(
                "Could not resolve ADC CNCO for port %s runit %s; skipping runit CNCO.",
                port,
                runit,
                exc_info=True,
            )
            return

        self._set_adc_cnco_indices(
            css=css,
            mxfe_idx=mxfe_idx,
            adc_indices={adc_idx},
            cnco_freq_hz=cnco_freq_hz,
        )

    @staticmethod
    def _find_dac_for_fduc(*, css: Any, mxfe_idx: int, fduc_idx: int) -> int:
        """Return the DAC/CDUC containing one FDUC."""
        for dac_idx in range(4):
            if fduc_idx in css.ad9082[mxfe_idx].get_fduc_of_dac(dac_idx):
                return dac_idx
        raise ValueError(f"FDUC{fduc_idx} is not assigned to any DAC on MxFE{mxfe_idx}.")

    @staticmethod
    def _resolve_adc_cnco_index(
        *,
        css: Any,
        group: int,
        rline: str,
        rchannel: int,
    ) -> tuple[int, int]:
        """Return the CDDC index for one capture rchannel."""
        mxfe_idx, primary_adc_idx = css.get_adc_idx(group, rline)
        if not css.is_dual_readout_mode_enabled(group, rline):
            return mxfe_idx, primary_adc_idx
        if rchannel == 0:
            return mxfe_idx, primary_adc_idx
        if rchannel == 1:
            return mxfe_idx, int(css._DUAL_READOUT_SECONDARY_CDDC)
        return mxfe_idx, primary_adc_idx

    @staticmethod
    def _set_dac_cnco_indices(
        *,
        css: Any,
        mxfe_idx: int,
        dac_indices: set[int],
        cnco_freq_hz: int,
    ) -> None:
        """Set selected DAC/CDUC CNCO indices."""
        freq_hz, ftw = css._validate_frequency_info(
            mxfe_idx,
            "dac_cnco",
            cnco_freq_hz,
            None,
        )
        logger.info(
            "DAC-CNCO%s of MxFE%s is set to %sHz via channel override.",
            sorted(dac_indices),
            mxfe_idx,
            freq_hz,
        )
        css.ad9082[mxfe_idx].set_dac_cnco(dac_indices, ftw)

    @staticmethod
    def _set_adc_cnco_indices(
        *,
        css: Any,
        mxfe_idx: int,
        adc_indices: set[int],
        cnco_freq_hz: int,
    ) -> None:
        """Set selected ADC/CDDC CNCO indices."""
        freq_hz, ftw = css._validate_frequency_info(
            mxfe_idx,
            "adc_cnco",
            cnco_freq_hz,
            None,
        )
        logger.info(
            "ADC-CNCO%s of MxFE%s is set to %sHz via runit override.",
            sorted(adc_indices),
            mxfe_idx,
            freq_hz,
        )
        css.ad9082[mxfe_idx].set_adc_cnco(adc_indices, ftw)

    def get_resource_map(self, *, targets: list[str]) -> dict[str, list[dict]]:
        """Build a resource map for selected targets from system config database."""
        db = self._runtime_context.qubecalib.system_config_database
        target_settings = db._target_settings
        box_settings = db._box_settings
        port_settings = db._port_settings
        result: dict[str, list[dict]] = {}
        for target in targets:
            if target not in target_settings:
                raise ValueError(f"Target {target} not in available targets.")
            channels = db.get_channels_by_target(target)
            bpc_list = [db.get_channel(channel) for channel in channels]
            result[target] = [
                {
                    "box": box_settings[box_name],
                    "port": port_settings[port_name],
                    "channel_number": channel_number,
                    "target": target_settings[target],
                }
                for box_name, port_name, channel_number in bpc_list
            ]
        return result

    def _resolve_box(
        self,
        *,
        box_name: str,
        reconnect: bool,
    ) -> Quel1Box:
        """Resolve a box from runtime context or create it lazily."""
        self._runtime_context.validate_box_availability(box_name)
        if not self._runtime_context.is_connected:
            db = self._runtime_context.qubecalib.system_config_database
            box = db.create_box(box_name, reconnect=False)
            if reconnect:
                box.reconnect(
                    background_noise_threshold=DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
                )
            return box
        boxpool = self._runtime_context.boxpool
        if box_name in boxpool._boxes:
            return boxpool._boxes[box_name][0]
        db = self._runtime_context.qubecalib.system_config_database
        box = db.create_box(box_name, reconnect=False)
        if reconnect:
            box.reconnect(
                background_noise_threshold=DEFAULT_BACKGROUND_NOISE_THRESHOLD_AT_RECONNECT
            )
        return box

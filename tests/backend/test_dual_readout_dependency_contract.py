"""Contract tests for the pinned dual-readout Quelware dependency."""

from quel_ic_config import Quel1Box
from quel_ic_config.ad9082 import Ad9082Mixin
from quel_ic_config.quel1_box_intrinsic import Quel1BoxIntrinsic


def test_pinned_quelware_exposes_dual_readout_contract() -> None:
    """Given the locked environment, Quelware exposes all required dual-readout APIs."""
    assert callable(getattr(Quel1Box, "enable_dual_readout", None))
    assert callable(getattr(Quel1Box, "is_dual_readout_enabled", None))
    assert callable(getattr(Quel1BoxIntrinsic, "_load_config_parameter", None))
    assert callable(getattr(Ad9082Mixin, "get_fduc_of_dac", None))
    assert callable(getattr(Ad9082Mixin, "get_virtual_adc_select", None))
    assert hasattr(Ad9082Mixin, "_DUAL_READOUT_PRIMARY_FDDC")
    assert hasattr(Ad9082Mixin, "_DUAL_READOUT_SECONDARY_FDDC")
    assert hasattr(Ad9082Mixin, "_DUAL_READOUT_PRIMARY_VC_PAIR")
    assert hasattr(Ad9082Mixin, "_DUAL_READOUT_SECONDARY_VC_PAIR")

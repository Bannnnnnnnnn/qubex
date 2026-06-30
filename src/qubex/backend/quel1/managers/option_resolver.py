"""Helpers for resolving QuEL-1 box option labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DUAL_READOUT_GROUP_OPTION_PREFIX = "dual_readout_group"
DUAL_READOUT_OUTPUT_OPTION_PREFIX = "dual_readout_output_mxfe"


def resolve_dual_readout_groups_from_labels(
    option_labels: Sequence[str],
) -> frozenset[int]:
    """Resolve dual-readout group indices from qubex/quelware option labels."""
    groups: set[int] = set()
    for option_label in option_labels:
        if option_label.startswith(DUAL_READOUT_GROUP_OPTION_PREFIX):
            group_text = option_label.removeprefix(DUAL_READOUT_GROUP_OPTION_PREFIX)
        elif option_label.startswith(DUAL_READOUT_OUTPUT_OPTION_PREFIX):
            group_text = option_label.removeprefix(DUAL_READOUT_OUTPUT_OPTION_PREFIX)
        else:
            continue
        try:
            groups.add(int(group_text))
        except ValueError:
            continue
    return frozenset(groups)


def expand_config_option_labels(option_labels: Sequence[str]) -> list[str]:
    """Expand qubex-only box options into quelware config option labels."""
    expanded: list[str] = []
    for option_label in option_labels:
        if option_label.startswith(DUAL_READOUT_GROUP_OPTION_PREFIX):
            group_text = option_label.removeprefix(DUAL_READOUT_GROUP_OPTION_PREFIX)
            try:
                group = int(group_text)
            except ValueError:
                expanded.append(option_label)
                continue
            expanded.append(f"{DUAL_READOUT_OUTPUT_OPTION_PREFIX}{group}")
            continue
        expanded.append(option_label)
    return list(dict.fromkeys(expanded))


def resolve_config_options(
    *,
    option_map: Mapping[str, Any],
    box_name: str,
    option_labels: Sequence[str],
) -> list[Any] | None:
    """Resolve box option labels to driver-specific Quel1ConfigOption members."""
    expanded_labels = expand_config_option_labels(option_labels)
    if not expanded_labels:
        return None

    config_options: list[Any] = []
    for option_label in expanded_labels:
        option = option_map.get(option_label)
        if option is None:
            if option_label.startswith(DUAL_READOUT_OUTPUT_OPTION_PREFIX):
                config_options.append(option_label)
                continue
            raise ValueError(
                f"Unknown Quel1 config option `{option_label}` for box `{box_name}`."
            )
        config_options.append(option)
    return config_options

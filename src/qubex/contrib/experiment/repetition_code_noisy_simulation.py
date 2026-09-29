"""Stim-based noisy simulation for one-dimensional repetition-code results."""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray

from qubex.experiment.models import Result

Basis = Literal["bit", "phase"]


class GateType(IntEnum):
    """qec_camp-compatible gate type identifiers."""

    INIT0 = 0
    INIT1 = 1
    CNOT = 2
    IDLE_CNOT = 3
    IDLE_MEAS = 4
    MEAS = 5
    CLOCK = 6
    H = 7


@dataclass(frozen=True)
class _Layout:
    qubits: list[str]
    data_qubits: list[str]
    measure_qubits: list[str]
    distance: int
    layer1_pairs: list[tuple[int, int]]
    layer2_pairs: list[tuple[int, int]]
    cnot_pairs: list[tuple[str, str]]


@dataclass(frozen=True)
class _QubitCalibration:
    qubit: str
    index: int
    t1_us: float
    t1_source: str
    t2_us: float
    t2_source: str
    init0_error: float
    init1_error: float
    meas_error: float
    h_error: float
    average_readout_fidelity: float
    thermal_excitation_probability: float | None


@dataclass(frozen=True)
class _NoiseModel:
    init0: dict[int, float]
    init1: dict[int, float]
    meas: dict[int, float]
    h: dict[int, float]
    cnot: dict[tuple[int, int], float]
    qubit_calibrations: dict[int, _QubitCalibration]
    layer1_duration_ns: float
    layer2_duration_ns: float
    readout_duration_ns: float
    idle_error_scale: float
    uniform_idle_error_rate: float | None
    noise_table: list[dict[str, object]]
    calibration_summary: list[dict[str, object]]
    cnot_duration_table: list[dict[str, object]]
    sources: dict[str, object]


@dataclass(frozen=True)
class _ParameterStore:
    params_dir: Path | None
    cr_params: Mapping[str, object]

    def load_param(self, name: str) -> Mapping[str, object]:
        if self.params_dir is None:
            return {}
        path = self.params_dir / f"{name}.yaml"
        if not path.exists():
            return {}
        obj = _load_yaml_file(path)
        if isinstance(obj, Mapping) and isinstance(obj.get("data"), Mapping):
            return obj["data"]
        if isinstance(obj, Mapping):
            return obj
        return {}

    def get_param(
        self, name: str, key: str, default: float | None = None
    ) -> float | None:
        return _finite_float(self.load_param(name).get(key), default=default)

    def get_first_param(
        self,
        names: Sequence[str],
        key: str,
        *,
        default: float | None,
    ) -> tuple[float | None, str]:
        for name in names:
            value = self.get_param(name, key, default=None)
            if value is not None:
                return value, name
        return default, "fallback"


def repetition_code_noisy_simulation(
    exp: Any | None,
    qubits: Sequence[str],
    rounds: int | Sequence[int],
    *,
    basis: Basis = "bit",
    initial: str | None = None,
    n_shots: int = 10_000,
    seed: int = 42,
    config_root: Path | str | None = None,
    chip_id: str | None = None,
    params_dir: Path | str | None = None,
    calibration_note_path: Path | str | None = None,
    measured_results: Any | None = None,
    uniform_error_rate: float | None = None,
    readout_duration_ns: float | None = None,
    default_init0_error: float = 0.01,
    default_x180_fidelity: float = 0.99,
    default_readout_fidelity: float = 0.90,
    default_zx90_fidelity: float = 0.95,
    default_t1_us: float = 30.0,
    default_t2_us: float = 20.0,
    default_readout_duration_ns: float = 1024.0,
    default_cnot_duration_ns: float = 512.0,
    init_error_scale: float = 1.0,
    meas_error_scale: float = 1.0,
    cnot_error_scale: float = 1.0,
    idle_error_scale: float = 1.0,
    h_error_scale: float = 1.0,
    t2_param_priority: Sequence[str] = ("t2_echo", "t2_echo_average", "t2_star"),
    run_correlation: bool = True,
    return_records: bool = False,
    return_stim_circuits: bool = False,
    plot: bool = False,
) -> Result:
    """
    Simulate a repetition-code experiment with stim and compare it to measured data.

    ``stim`` is treated as an optional runtime dependency. qubex itself does not
    depend on stim; this function raises a clear ImportError when stim is absent.

    Parameters
    ----------
    exp
        Optional :class:`qubex.Experiment`. When present, ``chip_id``,
        ``params_dir``, and CR calibration data are inferred from it when not
        explicitly provided. Pass ``None`` when using only files.
    qubits
        Physical chain ordered as data, measure, data, measure, ..., data.
    rounds
        One round count or a sequence of round counts to simulate.
    basis, initial
        Same logical convention as ``contrib.repetition_code``:
        ``basis="bit"`` defaults to ``initial="0"`` and ``basis="phase"``
        defaults to ``initial="+"``.
    n_shots, seed
        Stim sampling size and base random seed. Each round uses ``seed + i``.
    config_root, chip_id, params_dir, calibration_note_path
        Sources for qubex-config-local style params and calibration notes.
        If ``uniform_error_rate`` is provided, calibration files are optional.
    measured_results
        A ``Result``, ``result.data`` mapping, path, or sequence of those from
        ``contrib.repetition_code``. Logical error rates and correlation
        matrices are compared when available.
    uniform_error_rate
        If provided, use a uniform depolarizing rate instead of calibration
        parameters for all modeled operations.
    run_correlation
        If True, return simulated detector two-point correlation matrices and
        measured-minus-simulated differences when measured matrices are present.
    return_records, return_stim_circuits
        Include raw sampled measurement records or stim circuit text in
        ``Result.data``. These can be large.
    plot
        If True, create a measured-vs-simulated logical-error plot.

    Returns
    -------
    Result
        Payload containing noise tables, simulation payloads, measured rows,
        comparison rows, and optional figures.
    """
    stim = _require_stim()
    layout = _build_layout(qubits)
    round_list = _normalize_rounds(rounds)
    basis = _normalize_basis(basis)
    initial = _normalize_initial(initial, basis=basis)
    n_shots = _validate_positive_int(n_shots, name="n_shots")

    params_path, cr_params, source_info = _resolve_parameter_sources(
        exp,
        config_root=config_root,
        chip_id=chip_id,
        params_dir=params_dir,
        calibration_note_path=calibration_note_path,
    )
    store = _ParameterStore(params_dir=params_path, cr_params=cr_params)
    noise_model = _build_noise_model(
        layout,
        store,
        uniform_error_rate=uniform_error_rate,
        readout_duration_ns=readout_duration_ns,
        default_init0_error=default_init0_error,
        default_x180_fidelity=default_x180_fidelity,
        default_readout_fidelity=default_readout_fidelity,
        default_zx90_fidelity=default_zx90_fidelity,
        default_t1_us=default_t1_us,
        default_t2_us=default_t2_us,
        default_readout_duration_ns=default_readout_duration_ns,
        default_cnot_duration_ns=default_cnot_duration_ns,
        init_error_scale=init_error_scale,
        meas_error_scale=meas_error_scale,
        cnot_error_scale=cnot_error_scale,
        idle_error_scale=idle_error_scale,
        h_error_scale=h_error_scale,
        t2_param_priority=t2_param_priority,
        source_info=source_info,
    )

    measured_payloads = _load_measured_payloads(measured_results)
    measured_rows = [_measured_summary_row(payload) for payload in measured_payloads]

    simulations: list[dict[str, object]] = []
    for offset, num_round in enumerate(round_list):
        simulations.append(
            _simulate_one_round_count(
                stim,
                layout=layout,
                num_round=num_round,
                basis=basis,
                initial=initial,
                noise_model=noise_model,
                n_shots=n_shots,
                seed=seed + offset,
                run_correlation=run_correlation,
                return_records=return_records,
                return_stim_circuit=return_stim_circuits,
            )
        )

    simulation_rows = [_simulation_summary_row(payload) for payload in simulations]
    comparison_rows = _comparison_rows(measured_rows, simulation_rows)
    correlation_comparisons = (
        _correlation_comparisons(measured_payloads, simulations)
        if run_correlation
        else []
    )

    payload: dict[str, object] = {
        "qubex_has_stim_dependency": False,
        "stim_available": True,
        "qubits": list(layout.qubits),
        "data_qubits": list(layout.data_qubits),
        "measure_qubits": list(layout.measure_qubits),
        "cnot_pairs": list(layout.cnot_pairs),
        "distance": layout.distance,
        "rounds": round_list,
        "basis": basis,
        "initial": initial,
        "n_shots": n_shots,
        "seed": seed,
        "uniform_error_rate": uniform_error_rate,
        "source_info": noise_model.sources,
        "noise_table": noise_model.noise_table,
        "calibration_summary": noise_model.calibration_summary,
        "cnot_duration_table": noise_model.cnot_duration_table,
        "simulations": simulations,
        "simulation_rows": simulation_rows,
        "measured_rows": measured_rows,
        "comparison_rows": comparison_rows,
        "correlation_comparisons": correlation_comparisons,
        "notes": [
            "qubex does not declare stim as a dependency; stim is imported only when this function runs.",
            "Correlation matrices from calibration are model predictions under independent local noise, not measured correlated-noise estimates.",
            "t1_logical_error_rate is an independent-data-qubit majority-vote approximation, separate from the stim detector-decoder simulation.",
        ],
    }

    result = Result(data=payload)
    if plot:
        result.figures = {
            "logical_error": plot_repetition_code_noisy_simulation(result)
        }
    return result


def is_stim_available() -> bool:
    """Return whether the optional stim runtime dependency can be imported."""
    try:
        import stim  # noqa: F401
    except ImportError:
        return False
    return True


def plot_repetition_code_noisy_simulation(
    result: Result | Mapping[str, object],
    *,
    show: bool = True,
) -> Any:
    """Plot measured, stim-simulated, and T1-only logical error rates."""
    payload = result.data if isinstance(result, Result) else result
    rows = list(payload.get("comparison_rows", []))
    if not rows:
        rows = list(payload.get("simulation_rows", []))
    if not rows:
        raise ValueError("No simulation or comparison rows are available to plot.")

    plt = _matplotlib_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    labels = sorted(
        {
            (str(row.get("basis", "")), str(row.get("initial", "")))
            for row in rows
            if isinstance(row, Mapping)
        }
    )
    cmap = plt.get_cmap("tab10")
    for label_index, (basis, initial) in enumerate(labels):
        color = cmap(label_index % 10)
        group = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and str(row.get("basis", "")) == basis
            and str(row.get("initial", "")) == initial
        ]
        group = sorted(group, key=lambda row: int(row.get("num_round", 0)))
        rounds = [int(row.get("num_round", 0)) for row in group]
        sim_rates = [
            _optional_float(row.get("simulation_logical_error_rate")) for row in group
        ]
        if any(rate is not None for rate in sim_rates):
            ax.plot(
                rounds,
                [np.nan if rate is None else rate for rate in sim_rates],
                marker="x",
                linestyle="--",
                color=color,
                label=f"stim {basis}:{initial}",
            )
        measured_rates = [
            _optional_float(row.get("measured_logical_error_rate")) for row in group
        ]
        if any(rate is not None for rate in measured_rates):
            ax.plot(
                rounds,
                [np.nan if rate is None else rate for rate in measured_rates],
                marker="o",
                linestyle="-",
                color=color,
                label=f"meas {basis}:{initial}",
            )
        t1_rates = [_optional_float(row.get("t1_logical_error_rate")) for row in group]
        if any(rate is not None for rate in t1_rates):
            ax.plot(
                rounds,
                [np.nan if rate is None else rate for rate in t1_rates],
                marker=".",
                linestyle=":",
                color=color,
                label=f"T1 {basis}:{initial}",
            )
    ax.set_xlabel("num_round")
    ax.set_ylabel("logical error rate")
    ax.set_title("repetition code measured vs noisy simulation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def _require_stim() -> Any:
    try:
        import stim
    except ImportError as exc:
        raise ImportError(
            "stim is not installed in this qubex environment. Install stim in the "
            "active environment to run repetition_code_noisy_simulation."
        ) from exc
    return stim


def _build_layout(qubits: Sequence[str]) -> _Layout:
    qubit_list = [str(qubit) for qubit in qubits]
    if len(qubit_list) < 5 or len(qubit_list) % 2 == 0:
        raise ValueError("qubits must be an odd-length chain with at least 5 entries.")
    data_qubits = qubit_list[0::2]
    measure_qubits = qubit_list[1::2]
    distance = len(data_qubits)
    layer1_pairs = [
        (2 * stabilizer, 2 * stabilizer + 1) for stabilizer in range(distance - 1)
    ]
    layer2_pairs = [
        (2 * stabilizer + 2, 2 * stabilizer + 1) for stabilizer in range(distance - 1)
    ]
    cnot_pairs: list[tuple[str, str]] = []
    for stabilizer, measure_qubit in enumerate(measure_qubits):
        cnot_pairs.append((data_qubits[stabilizer], measure_qubit))
        cnot_pairs.append((data_qubits[stabilizer + 1], measure_qubit))
    return _Layout(
        qubits=qubit_list,
        data_qubits=data_qubits,
        measure_qubits=measure_qubits,
        distance=distance,
        layer1_pairs=layer1_pairs,
        layer2_pairs=layer2_pairs,
        cnot_pairs=cnot_pairs,
    )


def _normalize_rounds(rounds: int | Sequence[int]) -> list[int]:
    if isinstance(rounds, int):
        round_list = [rounds]
    else:
        round_list = [int(num_round) for num_round in rounds]
    if not round_list:
        raise ValueError("rounds must not be empty.")
    for num_round in round_list:
        _validate_positive_int(num_round, name="rounds")
    return round_list


def _validate_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _normalize_basis(basis: str) -> Basis:
    normalized = basis.strip().lower()
    if normalized not in ("bit", "phase"):
        raise ValueError("basis must be 'bit' or 'phase'.")
    return normalized  # type: ignore[return-value]


def _normalize_initial(initial: str | None, *, basis: Basis) -> str:
    if initial is None:
        return "0" if basis == "bit" else "+"
    normalized = str(initial).strip().lower().replace(" ", "")
    if normalized == "plus":
        normalized = "+"
    elif normalized == "minus":
        normalized = "-"
    allowed = {"0", "1"} if basis == "bit" else {"+", "-"}
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"initial must be one of {allowed_text} for basis={basis!r}.")
    return normalized


def _resolve_parameter_sources(
    exp: Any | None,
    *,
    config_root: Path | str | None,
    chip_id: str | None,
    params_dir: Path | str | None,
    calibration_note_path: Path | str | None,
) -> tuple[Path | None, Mapping[str, object], dict[str, object]]:
    resolved_chip_id = chip_id or _exp_string_attr(exp, "chip_id")
    resolved_params = Path(params_dir).expanduser() if params_dir is not None else None
    if resolved_params is None:
        ctx = getattr(exp, "ctx", None)
        ctx_params = getattr(ctx, "params_path", None)
        if ctx_params is not None:
            resolved_params = Path(str(ctx_params)).expanduser()
    if (
        resolved_params is None
        and config_root is not None
        and resolved_chip_id is not None
    ):
        resolved_params = Path(config_root).expanduser() / resolved_chip_id / "params"

    cr_params = _exp_cr_params(exp)
    resolved_calib = (
        Path(calibration_note_path).expanduser()
        if calibration_note_path is not None
        else None
    )
    if (
        resolved_calib is None
        and config_root is not None
        and resolved_chip_id is not None
    ):
        resolved_calib = (
            Path(config_root).expanduser()
            / resolved_chip_id
            / "calibration"
            / "calib_note.json"
        )
    if not cr_params and resolved_calib is not None and resolved_calib.exists():
        cr_params = _load_cr_params(resolved_calib)

    source_info: dict[str, object] = {
        "chip_id": resolved_chip_id,
        "params_dir": str(resolved_params) if resolved_params is not None else None,
        "calibration_note_path": str(resolved_calib)
        if resolved_calib is not None
        else None,
        "cr_params_source": "exp.calib_note"
        if _exp_cr_params(exp)
        else "calibration_note_path"
        if cr_params
        else "none",
    }
    return resolved_params, cr_params, source_info


def _exp_string_attr(exp: Any | None, name: str) -> str | None:
    if exp is None:
        return None
    value = getattr(exp, name, None)
    if value is None:
        return None
    return str(value)


def _exp_cr_params(exp: Any | None) -> Mapping[str, object]:
    if exp is None:
        return {}
    calib_note = getattr(exp, "calib_note", None)
    cr_params = getattr(calib_note, "cr_params", None)
    return cr_params if isinstance(cr_params, Mapping) else {}


def _load_cr_params(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    cr_params = obj.get("cr_params") if isinstance(obj, Mapping) else None
    return cr_params if isinstance(cr_params, Mapping) else {}


def _build_noise_model(
    layout: _Layout,
    store: _ParameterStore,
    *,
    uniform_error_rate: float | None,
    readout_duration_ns: float | None,
    default_init0_error: float,
    default_x180_fidelity: float,
    default_readout_fidelity: float,
    default_zx90_fidelity: float,
    default_t1_us: float,
    default_t2_us: float,
    default_readout_duration_ns: float,
    default_cnot_duration_ns: float,
    init_error_scale: float,
    meas_error_scale: float,
    cnot_error_scale: float,
    idle_error_scale: float,
    h_error_scale: float,
    t2_param_priority: Sequence[str],
    source_info: Mapping[str, object],
) -> _NoiseModel:
    if uniform_error_rate is not None:
        return _build_uniform_noise_model(
            layout,
            uniform_error_rate=uniform_error_rate,
            readout_duration_ns=readout_duration_ns or default_readout_duration_ns,
            cnot_duration_ns=default_cnot_duration_ns,
            default_t1_us=default_t1_us,
            default_t2_us=default_t2_us,
            source_info=source_info,
        )

    resolved_readout_duration = _resolve_readout_duration_ns(
        store,
        readout_duration_ns=readout_duration_ns,
        default_readout_duration_ns=default_readout_duration_ns,
    )
    layer1_duration, layer1_rows = _layer_duration_ns(
        layout,
        store,
        layout.layer1_pairs,
        default_cnot_duration_ns=default_cnot_duration_ns,
    )
    layer2_duration, layer2_rows = _layer_duration_ns(
        layout,
        store,
        layout.layer2_pairs,
        default_cnot_duration_ns=default_cnot_duration_ns,
    )

    init0: dict[int, float] = {}
    init1: dict[int, float] = {}
    meas: dict[int, float] = {}
    h: dict[int, float] = {}
    cnot: dict[tuple[int, int], float] = {}
    calibrations: dict[int, _QubitCalibration] = {}
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []

    for index, qubit in enumerate(layout.qubits):
        t1, t1_source = store.get_first_param(
            ("t1", "t1_average"),
            qubit,
            default=default_t1_us,
        )
        if t1 is None or t1 <= 0:
            t1 = default_t1_us
            t1_source = "fallback_DEFAULT_T1_US"
        t2, t2_source = store.get_first_param(
            t2_param_priority, qubit, default=default_t2_us
        )
        if t2 is None or t2 <= 0:
            t2 = default_t2_us
            t2_source = "fallback_DEFAULT_T2_US"

        p_init0, init0_source, init0_raw = _init0_error(
            store,
            qubit,
            default_init0_error=default_init0_error,
            scale=init_error_scale,
        )
        p_init1, init1_source, init1_raw = _init1_error(
            store,
            qubit,
            p_init0=p_init0,
            default_x180_fidelity=default_x180_fidelity,
            scale=init_error_scale,
        )
        p_meas, meas_source, readout_fidelity = _fidelity_error(
            store.get_param("average_readout_fidelity", qubit, default=None),
            default_fidelity=default_readout_fidelity,
            scale=meas_error_scale,
            config_source="average_readout_fidelity",
            fallback_source="fallback_DEFAULT_READOUT_FIDELITY",
        )
        p_h, h_source, h_raw = _h_error(
            store,
            qubit,
            default_x180_fidelity=default_x180_fidelity,
            scale=h_error_scale,
        )
        thermal = store.get_param("thermal_excitation_probability", qubit, default=None)

        init0[index] = p_init0
        init1[index] = p_init1
        meas[index] = p_meas
        h[index] = p_h
        calibrations[index] = _QubitCalibration(
            qubit=qubit,
            index=index,
            t1_us=t1,
            t1_source=t1_source,
            t2_us=t2,
            t2_source=t2_source,
            init0_error=p_init0,
            init1_error=p_init1,
            meas_error=p_meas,
            h_error=p_h,
            average_readout_fidelity=readout_fidelity,
            thermal_excitation_probability=thermal,
        )
        rows.extend(
            [
                _noise_row(
                    GateType.INIT0, (index,), qubit, p_init0, init0_source, init0_raw
                ),
                _noise_row(
                    GateType.INIT1, (index,), qubit, p_init1, init1_source, init1_raw
                ),
                _noise_row(
                    GateType.MEAS,
                    (index,),
                    qubit,
                    p_meas,
                    meas_source,
                    readout_fidelity,
                ),
                _noise_row(GateType.H, (index,), qubit, p_h, h_source, h_raw),
            ]
        )
        idle_meas_error, idle_meas_raw = _idle_error_from_calibration(
            calibrations[index],
            resolved_readout_duration,
            scale=idle_error_scale,
        )
        rows.append(
            _noise_row(
                GateType.IDLE_MEAS,
                (index,),
                qubit,
                idle_meas_error,
                f"{t1_source}+{t2_source}",
                idle_meas_raw,
                duration_ns=resolved_readout_duration,
            )
        )
        representative_idle_cnot_duration = max(layer1_duration, layer2_duration)
        idle_cnot_error, idle_cnot_raw = _idle_error_from_calibration(
            calibrations[index],
            representative_idle_cnot_duration,
            scale=idle_error_scale,
        )
        rows.append(
            _noise_row(
                GateType.IDLE_CNOT,
                (index,),
                qubit,
                idle_cnot_error,
                f"{t1_source}+{t2_source}",
                idle_cnot_raw,
                duration_ns=representative_idle_cnot_duration,
            )
        )
        summary.append(
            {
                "qubit": qubit,
                "index": index,
                "t1_us": t1,
                "t1_source": t1_source,
                "t2_us": t2,
                "t2_source": t2_source,
                "init0_error": p_init0,
                "init1_error": p_init1,
                "meas_error": p_meas,
                "h_error": p_h,
                "average_readout_fidelity": readout_fidelity,
                "thermal_excitation_probability": thermal,
            }
        )

    for control, target in layout.layer1_pairs + layout.layer2_pairs:
        physical = f"{layout.qubits[control]}->{layout.qubits[target]}"
        p_cnot, source, raw = _cnot_error(
            store,
            layout.qubits[control],
            layout.qubits[target],
            default_zx90_fidelity=default_zx90_fidelity,
            scale=cnot_error_scale,
        )
        cnot[(control, target)] = p_cnot
        rows.append(
            _noise_row(GateType.CNOT, (control, target), physical, p_cnot, source, raw)
        )

    cnot_duration_table = layer1_rows + layer2_rows
    return _NoiseModel(
        init0=init0,
        init1=init1,
        meas=meas,
        h=h,
        cnot=cnot,
        qubit_calibrations=calibrations,
        layer1_duration_ns=layer1_duration,
        layer2_duration_ns=layer2_duration,
        readout_duration_ns=resolved_readout_duration,
        idle_error_scale=idle_error_scale,
        uniform_idle_error_rate=None,
        noise_table=rows,
        calibration_summary=summary,
        cnot_duration_table=cnot_duration_table,
        sources=dict(source_info),
    )


def _build_uniform_noise_model(
    layout: _Layout,
    *,
    uniform_error_rate: float,
    readout_duration_ns: float,
    cnot_duration_ns: float,
    default_t1_us: float,
    default_t2_us: float,
    source_info: Mapping[str, object],
) -> _NoiseModel:
    p = _clip_probability(uniform_error_rate)
    init0 = dict.fromkeys(range(len(layout.qubits)), p)
    init1 = dict(init0)
    meas = dict(init0)
    h = dict(init0)
    cnot = dict.fromkeys(layout.layer1_pairs + layout.layer2_pairs, p)
    calibrations = {
        index: _QubitCalibration(
            qubit=qubit,
            index=index,
            t1_us=default_t1_us,
            t1_source="uniform_default",
            t2_us=default_t2_us,
            t2_source="uniform_default",
            init0_error=p,
            init1_error=p,
            meas_error=p,
            h_error=p,
            average_readout_fidelity=1.0 - p,
            thermal_excitation_probability=None,
        )
        for index, qubit in enumerate(layout.qubits)
    }
    rows = [
        _noise_row(gate_type, (index,), qubit, p, "uniform_error_rate", p)
        for index, qubit in enumerate(layout.qubits)
        for gate_type in (GateType.INIT0, GateType.INIT1, GateType.MEAS, GateType.H)
    ]
    for index, qubit in enumerate(layout.qubits):
        rows.append(
            _noise_row(
                GateType.IDLE_MEAS,
                (index,),
                qubit,
                p,
                "uniform_error_rate",
                p,
                duration_ns=readout_duration_ns,
            )
        )
        rows.append(
            _noise_row(
                GateType.IDLE_CNOT,
                (index,),
                qubit,
                p,
                "uniform_error_rate",
                p,
                duration_ns=cnot_duration_ns,
            )
        )
    rows.extend(
        _noise_row(
            GateType.CNOT,
            pair,
            f"{layout.qubits[pair[0]]}->{layout.qubits[pair[1]]}",
            p,
            "uniform_error_rate",
            p,
        )
        for pair in layout.layer1_pairs + layout.layer2_pairs
    )
    summary = [
        {
            "qubit": cal.qubit,
            "index": cal.index,
            "t1_us": cal.t1_us,
            "t2_us": cal.t2_us,
            "init0_error": p,
            "init1_error": p,
            "meas_error": p,
            "h_error": p,
        }
        for cal in calibrations.values()
    ]
    duration_table = [
        {
            "layer": layer,
            "pair": f"{layout.qubits[pair[0]]}->{layout.qubits[pair[1]]}",
            "duration_ns": cnot_duration_ns,
            "source": "uniform_default",
            "raw_duration_ns": cnot_duration_ns,
        }
        for layer, pairs in (
            ("layer1", layout.layer1_pairs),
            ("layer2", layout.layer2_pairs),
        )
        for pair in pairs
    ]
    return _NoiseModel(
        init0=init0,
        init1=init1,
        meas=meas,
        h=h,
        cnot=cnot,
        qubit_calibrations=calibrations,
        layer1_duration_ns=cnot_duration_ns,
        layer2_duration_ns=cnot_duration_ns,
        readout_duration_ns=readout_duration_ns,
        idle_error_scale=1.0,
        uniform_idle_error_rate=p,
        noise_table=rows,
        calibration_summary=summary,
        cnot_duration_table=duration_table,
        sources={**dict(source_info), "noise_model": "uniform"},
    )


def _resolve_readout_duration_ns(
    store: _ParameterStore,
    *,
    readout_duration_ns: float | None,
    default_readout_duration_ns: float,
) -> float:
    if readout_duration_ns is not None and readout_duration_ns > 0:
        return float(readout_duration_ns)
    defaults = store.load_param("measurement_defaults")
    readout = defaults.get("readout") if isinstance(defaults, Mapping) else None
    value = (
        _finite_float(readout.get("duration_ns"), default=None)
        if isinstance(readout, Mapping)
        else None
    )
    if value is None or value <= 0:
        return float(default_readout_duration_ns)
    return value


def _layer_duration_ns(
    layout: _Layout,
    store: _ParameterStore,
    pairs: Sequence[tuple[int, int]],
    *,
    default_cnot_duration_ns: float,
) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    durations: list[float] = []
    for control, target in pairs:
        control_label = layout.qubits[control]
        target_label = layout.qubits[target]
        duration, source, raw = _cr_duration_ns(
            store,
            control_label,
            target_label,
            default_cnot_duration_ns=default_cnot_duration_ns,
        )
        durations.append(duration)
        rows.append(
            {
                "layer": "layer1" if pairs is layout.layer1_pairs else "layer2",
                "pair": f"{control_label}->{target_label}",
                "duration_ns": duration,
                "source": source,
                "raw_duration_ns": raw,
            }
        )
    return max(durations) if durations else float(default_cnot_duration_ns), rows


def _cr_duration_ns(
    store: _ParameterStore,
    control: str,
    target: str,
    *,
    default_cnot_duration_ns: float,
) -> tuple[float, str, float | None]:
    key = f"{control}-{target}"
    params = store.cr_params.get(key)
    source = f"cr_params:{key}"
    if not isinstance(params, Mapping):
        reverse_key = f"{target}-{control}"
        params = store.cr_params.get(reverse_key)
        source = f"cr_params_reverse:{reverse_key}"
    duration = (
        _finite_float(params.get("duration"), default=None)
        if isinstance(params, Mapping)
        else None
    )
    if duration is None or duration <= 0:
        return (
            float(default_cnot_duration_ns),
            f"fallback_DEFAULT_CNOT_DURATION_NS:{source}",
            duration,
        )
    return duration, source, duration


def _init0_error(
    store: _ParameterStore,
    qubit: str,
    *,
    default_init0_error: float,
    scale: float,
) -> tuple[float, str, float]:
    thermal = store.get_param("thermal_excitation_probability", qubit, default=None)
    if thermal is None:
        return (
            _clip_probability(default_init0_error * scale),
            "fallback_DEFAULT_INIT0_ERROR",
            default_init0_error,
        )
    return (
        _clip_probability(thermal * scale),
        "thermal_excitation_probability",
        thermal,
    )


def _init1_error(
    store: _ParameterStore,
    qubit: str,
    *,
    p_init0: float,
    default_x180_fidelity: float,
    scale: float,
) -> tuple[float, str, dict[str, float]]:
    x180_fid = store.get_param("x180_gate_fidelity", qubit, default=None)
    source = "x180_gate_fidelity"
    if x180_fid is None or x180_fid <= 0 or x180_fid > 1:
        x180_fid = store.get_param(
            "average_gate_fidelity",
            qubit,
            default=default_x180_fidelity,
        )
        source = "average_gate_fidelity"
    if x180_fid is None or x180_fid <= 0 or x180_fid > 1:
        x180_fid = default_x180_fidelity
        source = "fallback_DEFAULT_X180_FIDELITY"
    p_x = 1.0 - x180_fid
    p = 1.0 - (1.0 - p_init0) * (1.0 - p_x)
    return (
        _clip_probability(p * scale),
        source,
        {"p_init0": p_init0, "x180_fidelity": x180_fid},
    )


def _h_error(
    store: _ParameterStore,
    qubit: str,
    *,
    default_x180_fidelity: float,
    scale: float,
) -> tuple[float, str, float]:
    x90_fid = store.get_param("x90_gate_fidelity", qubit, default=None)
    if x90_fid is not None and 0 < x90_fid <= 1:
        return _clip_probability((1.0 - x90_fid) * scale), "x90_gate_fidelity", x90_fid
    avg_fid = store.get_param("average_gate_fidelity", qubit, default=None)
    if avg_fid is not None and 0 < avg_fid <= 1:
        return (
            _clip_probability((1.0 - avg_fid) * scale),
            "average_gate_fidelity",
            avg_fid,
        )
    return (
        _clip_probability((1.0 - default_x180_fidelity) * scale),
        "fallback_DEFAULT_X180_FIDELITY",
        default_x180_fidelity,
    )


def _fidelity_error(
    fidelity: float | None,
    *,
    default_fidelity: float,
    scale: float,
    config_source: str,
    fallback_source: str,
) -> tuple[float, str, float]:
    source = config_source
    if fidelity is None or fidelity <= 0 or fidelity > 1:
        fidelity = default_fidelity
        source = fallback_source
    return _clip_probability((1.0 - fidelity) * scale), source, fidelity


def _cnot_error(
    store: _ParameterStore,
    control: str,
    target: str,
    *,
    default_zx90_fidelity: float,
    scale: float,
) -> tuple[float, str, float]:
    key = f"{control}-{target}"
    fid = store.get_param("zx90_gate_fidelity", key, default=None)
    source = "zx90_gate_fidelity"
    if fid is None:
        reverse_key = f"{target}-{control}"
        fid = store.get_param("zx90_gate_fidelity", reverse_key, default=None)
        if fid is not None:
            source = f"zx90_gate_fidelity_reverse:{reverse_key}"
    return _fidelity_error(
        fid,
        default_fidelity=default_zx90_fidelity,
        scale=scale,
        config_source=source,
        fallback_source="fallback_DEFAULT_ZX90_FIDELITY",
    )


def _idle_error_from_calibration(
    calibration: _QubitCalibration,
    duration_ns: float,
    *,
    scale: float,
) -> tuple[float, dict[str, float]]:
    duration_us = duration_ns / 1000.0
    p_t1 = 1.0 - math.exp(-duration_us / calibration.t1_us)
    inv_tphi = max(0.0, 1.0 / calibration.t2_us - 1.0 / (2.0 * calibration.t1_us))
    if inv_tphi > 0:
        tphi = 1.0 / inv_tphi
        p_phi = 0.5 * (1.0 - math.exp(-duration_us / tphi))
    else:
        tphi = math.inf
        p_phi = 0.0
    p = 1.0 - (1.0 - p_t1) * (1.0 - p_phi)
    return (
        _clip_probability(p * scale),
        {
            "duration_ns": duration_ns,
            "t1_us": calibration.t1_us,
            "t2_us": calibration.t2_us,
            "tphi_us": tphi,
            "p_t1": p_t1,
            "p_phi": p_phi,
        },
    )


def _noise_row(
    gate_type: GateType,
    target_qubit_list: tuple[int, ...],
    physical_target: str,
    error_rate: float,
    source: str,
    raw_value: object,
    *,
    duration_ns: float | None = None,
) -> dict[str, object]:
    return {
        "gate_type": gate_type.name,
        "gate_type_value": int(gate_type),
        "target_qubit_list": target_qubit_list,
        "physical_target": physical_target,
        "error_rate": float(error_rate),
        "source": source,
        "raw_value": _jsonable(raw_value),
        "duration_ns": duration_ns,
    }


def _simulate_one_round_count(
    stim: Any,
    *,
    layout: _Layout,
    num_round: int,
    basis: Basis,
    initial: str,
    noise_model: _NoiseModel,
    n_shots: int,
    seed: int,
    run_correlation: bool,
    return_records: bool,
    return_stim_circuit: bool,
) -> dict[str, object]:
    circuit = _build_stim_circuit(
        stim,
        layout=layout,
        num_round=num_round,
        basis=basis,
        initial=initial,
        noise_model=noise_model,
    )
    records = np.asarray(
        circuit.compile_sampler(seed=seed).sample(n_shots), dtype=np.int8
    )
    expected_records = num_round * len(layout.measure_qubits) + len(layout.data_qubits)
    if records.shape[1] != expected_records:
        raise RuntimeError(
            f"stim produced {records.shape[1]} records, expected {expected_records}."
        )
    analysis = _analyze_record_array(
        records,
        distance=layout.distance,
        rounds=num_round,
        basis=basis,
        initial=initial,
        run_correlation=run_correlation,
    )
    t1_theory = _t1_logical_error_rate(
        layout,
        noise_model,
        rounds=num_round,
        basis=basis,
        initial=initial,
    )
    payload: dict[str, object] = {
        "source": "simulation",
        "qubits": list(layout.qubits),
        "data_qubits": list(layout.data_qubits),
        "measure_qubits": list(layout.measure_qubits),
        "distance": layout.distance,
        "rounds": num_round,
        "num_round": num_round,
        "basis": basis,
        "initial": initial,
        "n_shots": n_shots,
        "seed": seed,
        **analysis,
        **t1_theory,
    }
    if return_records:
        payload["record_array"] = records
    if return_stim_circuit:
        payload["stim_circuit"] = str(circuit)
    return payload


def _build_stim_circuit(
    stim: Any,
    *,
    layout: _Layout,
    num_round: int,
    basis: Basis,
    initial: str,
    noise_model: _NoiseModel,
) -> Any:
    circuit = stim.Circuit()
    _append_initialization(
        circuit, layout=layout, basis=basis, initial=initial, noise_model=noise_model
    )
    for _round_index in range(num_round):
        if basis == "phase":
            _append_h_layer(
                circuit,
                data_indices=range(0, len(layout.qubits), 2),
                noise_model=noise_model,
            )
        _append_cnot_layer(
            circuit,
            layout=layout,
            pairs=layout.layer1_pairs,
            layer_duration_ns=noise_model.layer1_duration_ns,
            noise_model=noise_model,
        )
        circuit.append("TICK")
        _append_cnot_layer(
            circuit,
            layout=layout,
            pairs=layout.layer2_pairs,
            layer_duration_ns=noise_model.layer2_duration_ns,
            noise_model=noise_model,
        )
        circuit.append("TICK")
        if basis == "phase":
            _append_h_layer(
                circuit,
                data_indices=range(0, len(layout.qubits), 2),
                noise_model=noise_model,
            )
        _append_measurement_layer(
            circuit,
            targets=range(1, len(layout.qubits), 2),
            noise_model=noise_model,
        )
        _append_idle_layer(
            circuit,
            targets=range(0, len(layout.qubits), 2),
            duration_ns=noise_model.readout_duration_ns,
            noise_model=noise_model,
        )
        circuit.append("TICK")
    if basis == "phase":
        _append_h_layer(
            circuit,
            data_indices=range(0, len(layout.qubits), 2),
            noise_model=noise_model,
        )
    _append_measurement_layer(
        circuit,
        targets=range(0, len(layout.qubits), 2),
        noise_model=noise_model,
    )
    return circuit


def _append_initialization(
    circuit: Any,
    *,
    layout: _Layout,
    basis: Basis,
    initial: str,
    noise_model: _NoiseModel,
) -> None:
    data_indices = set(range(0, len(layout.qubits), 2))
    for index in range(len(layout.qubits)):
        circuit.append("R", index)
        if (index in data_indices and basis == "bit" and initial == "1") or (
            index in data_indices and basis == "phase" and initial == "-"
        ):
            circuit.append("X", index)
            circuit.append("DEPOLARIZE1", index, noise_model.init1[index])
        else:
            circuit.append("DEPOLARIZE1", index, noise_model.init0[index])
    if basis == "phase":
        _append_h_layer(circuit, data_indices=data_indices, noise_model=noise_model)
    circuit.append("TICK")


def _append_h_layer(
    circuit: Any,
    *,
    data_indices: Sequence[int] | range | set[int],
    noise_model: _NoiseModel,
) -> None:
    for index in data_indices:
        circuit.append("H", int(index))
        circuit.append("DEPOLARIZE1", int(index), noise_model.h[int(index)])
    circuit.append("TICK")


def _append_cnot_layer(
    circuit: Any,
    *,
    layout: _Layout,
    pairs: Sequence[tuple[int, int]],
    layer_duration_ns: float,
    noise_model: _NoiseModel,
) -> None:
    touched: set[int] = set()
    for control, target in pairs:
        circuit.append("CX", [control, target])
        circuit.append(
            "DEPOLARIZE2", [control, target], noise_model.cnot[(control, target)]
        )
        touched.add(control)
        touched.add(target)
    idle_targets = [
        index for index in range(len(layout.qubits)) if index not in touched
    ]
    _append_idle_layer(
        circuit,
        targets=idle_targets,
        duration_ns=layer_duration_ns,
        noise_model=noise_model,
    )


def _append_idle_layer(
    circuit: Any,
    *,
    targets: Sequence[int] | range,
    duration_ns: float,
    noise_model: _NoiseModel,
) -> None:
    for index in targets:
        if noise_model.uniform_idle_error_rate is None:
            error_rate, _raw = _idle_error_from_calibration(
                noise_model.qubit_calibrations[int(index)],
                duration_ns,
                scale=noise_model.idle_error_scale,
            )
        else:
            error_rate = noise_model.uniform_idle_error_rate
        circuit.append("DEPOLARIZE1", int(index), error_rate)


def _append_measurement_layer(
    circuit: Any,
    *,
    targets: Sequence[int] | range,
    noise_model: _NoiseModel,
) -> None:
    for index in targets:
        p = noise_model.meas[int(index)]
        circuit.append("DEPOLARIZE1", int(index), p / 2.0)
        circuit.append("M", int(index), p / 2.0)


def _analyze_record_array(
    record_array: NDArray[np.int8],
    *,
    distance: int,
    rounds: int,
    basis: Basis,
    initial: str,
    run_correlation: bool,
) -> dict[str, object]:
    detector_array = _detectors_from_records(
        record_array, distance=distance, rounds=rounds
    )
    detector_flat = detector_array.reshape(detector_array.shape[0], -1)
    data_bits = record_array[:, rounds * (distance - 1) :]
    reference = _logical_reference_bit(initial, basis=basis)
    measured_observable = np.bitwise_xor.reduce(data_bits ^ reference, axis=1)
    predicted_observable = _decode_observables(
        detector_flat,
        distance=distance,
        rounds=rounds,
    )
    logical_errors = predicted_observable != measured_observable
    num_logical_errors = int(np.count_nonzero(logical_errors))
    logical_error_rate = float(num_logical_errors / len(record_array))
    payload: dict[str, object] = {
        "detector_array": detector_array,
        "detector_labels": _detector_labels(distance, rounds),
        "predicted_observable": predicted_observable,
        "measured_observable": measured_observable,
        "logical_errors": logical_errors,
        "num_logical_errors": num_logical_errors,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_per_round_linear": float(logical_error_rate / rounds),
        "logical_error_rate_per_round_iid": float(
            1.0 - (1.0 - logical_error_rate) ** (1.0 / rounds)
        ),
        "stderr": float(
            math.sqrt(
                logical_error_rate * (1.0 - logical_error_rate) / len(record_array)
            )
        ),
        "detector_hit_rates": _detector_hit_rates(detector_array),
    }
    if run_correlation:
        payload["correlation_matrix"] = _two_point_correlation(detector_flat)
    return payload


def _detectors_from_records(
    record_array: NDArray[np.int8],
    *,
    distance: int,
    rounds: int,
) -> NDArray[np.int8]:
    stabilizers = distance - 1
    measure = record_array[:, : rounds * stabilizers].reshape(
        record_array.shape[0],
        rounds,
        stabilizers,
    )
    data = record_array[:, rounds * stabilizers :]
    rows = np.zeros((record_array.shape[0], rounds + 3, stabilizers), dtype=np.int8)
    rows[:, 2 : 2 + rounds, :] = measure
    final = np.empty((record_array.shape[0], stabilizers), dtype=np.int8)
    for stabilizer in range(stabilizers):
        final[:, stabilizer] = (
            data[:, stabilizer]
            ^ data[:, stabilizer + 1]
            ^ rows[:, 1 + rounds, stabilizer]
        )
    rows[:, 2 + rounds, :] = final
    return rows[:, : rounds + 1, :] ^ rows[:, 2 : rounds + 3, :]


def _logical_reference_bit(initial: str, *, basis: Basis) -> int:
    if basis == "phase":
        return 0 if initial == "+" else 1
    return int(initial)


def _detector_labels(distance: int, rounds: int) -> list[str]:
    labels: list[str] = []
    for detector_row in range(rounds + 1):
        if detector_row == 0:
            row_label = "round0_boundary"
        elif detector_row == 1 and rounds > 1:
            row_label = "round1_boundary"
        elif detector_row < rounds:
            row_label = f"round{detector_row}_time"
        else:
            row_label = "final_data_boundary"
        labels.extend(
            f"{row_label}:S{stabilizer}" for stabilizer in range(distance - 1)
        )
    return labels


def _detector_hit_rates(detector_array: NDArray[np.int8]) -> list[dict[str, object]]:
    rates = detector_array.mean(axis=0)
    rows: list[dict[str, object]] = []
    for detector_row in range(rates.shape[0]):
        rows.extend(
            {
                "detector_row": detector_row,
                "stabilizer": f"S{stabilizer}",
                "hit_rate": float(rates[detector_row, stabilizer]),
            }
            for stabilizer in range(rates.shape[1])
        )
    return rows


def _two_point_correlation(
    detector_flat: NDArray[np.int8],
    *,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    detector_values = detector_flat.astype(float)
    means = detector_values.mean(axis=0)
    matrix = np.full((detector_values.shape[1], detector_values.shape[1]), np.nan)
    for i in range(detector_values.shape[1]):
        xi = means[i]
        for j in range(i + 1, detector_values.shape[1]):
            xj = means[j]
            xij = float(np.mean(detector_values[:, i] * detector_values[:, j]))
            denom = 1.0 - 2.0 * xi - 2.0 * xj + 4.0 * xij
            if abs(denom) < eps:
                denom = eps if denom >= 0 else -eps
            radicand = 1.0 - 4.0 * (xij - xi * xj) / denom
            value = 0.5 - 0.5 * np.sqrt(max(0.0, radicand))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _decode_observables(
    detector_flat: NDArray[np.int8],
    *,
    distance: int,
    rounds: int,
) -> NDArray[np.int8]:
    decoder = _DecoderGraph(distance=distance, rounds=rounds)
    return np.asarray([decoder.decode(row) for row in detector_flat], dtype=np.int8)


class _DecoderGraph:
    def __init__(self, *, distance: int, rounds: int) -> None:
        self.distance = int(distance)
        self.rounds = int(rounds)
        self.num_rows = self.rounds + 1
        self.num_stabilizers = self.distance - 1
        self.num_detectors = self.num_rows * self.num_stabilizers
        self.left_boundary = self.num_detectors
        self.right_boundary = self.num_detectors + 1
        self.graph = self._build_graph()
        self.shortest_paths = {
            node: self._shortest_paths_from(node)
            for node in range(self.num_detectors + 2)
        }
        self._match_cache: dict[tuple[int, ...], tuple[int, int]] = {}

    def decode(self, detector_row: NDArray[np.int8]) -> int:
        defects = tuple(int(index) for index in np.flatnonzero(detector_row))
        _cost, observable = self._match(defects)
        return int(observable)

    def _detector_index(self, row: int, stabilizer: int) -> int:
        return row * self.num_stabilizers + stabilizer

    def _build_graph(self) -> list[list[tuple[int, int]]]:
        graph: list[list[tuple[int, int]]] = [[] for _ in range(self.num_detectors + 2)]

        def add_edge(left: int, right: int, observable: int = 0) -> None:
            graph[left].append((right, observable))
            graph[right].append((left, observable))

        for row in range(self.num_rows):
            for stabilizer in range(self.num_stabilizers):
                node = self._detector_index(row, stabilizer)
                if row + 1 < self.num_rows:
                    add_edge(node, self._detector_index(row + 1, stabilizer))
                if stabilizer == 0:
                    add_edge(node, self.left_boundary)
                if stabilizer == self.num_stabilizers - 1:
                    add_edge(node, self.right_boundary, observable=1)
                if stabilizer + 1 < self.num_stabilizers:
                    add_edge(node, self._detector_index(row, stabilizer + 1))
        add_edge(self.left_boundary, self.right_boundary, observable=1)
        return graph

    def _shortest_paths_from(self, start: int) -> dict[int, tuple[int, int]]:
        best: dict[int, tuple[int, int]] = {start: (0, 0)}
        queue = [start]
        while queue:
            node = queue.pop(0)
            distance, observable = best[node]
            for neighbor, edge_observable in self.graph[node]:
                candidate = (distance + 1, observable ^ edge_observable)
                if neighbor not in best or candidate[0] < best[neighbor][0]:
                    best[neighbor] = candidate
                    queue.append(neighbor)
        return best

    def _pair_cost(self, left: int, right: int) -> tuple[int, int]:
        return self.shortest_paths[left][right]

    def _match(self, defects: tuple[int, ...]) -> tuple[int, int]:
        defects = tuple(sorted(defects))
        if defects in self._match_cache:
            return self._match_cache[defects]
        result = self._match_uncached(defects)
        self._match_cache[defects] = result
        return result

    def _match_uncached(self, defects: tuple[int, ...]) -> tuple[int, int]:
        if not defects:
            return 0, 0
        first = defects[0]
        remaining = defects[1:]
        boundary_options = [self.left_boundary, self.right_boundary]
        candidates: list[tuple[int, int]] = []
        for boundary in boundary_options:
            pair_cost, pair_obs = self._pair_cost(first, boundary)
            rem_cost, rem_obs = self._match(remaining)
            candidates.append((pair_cost + rem_cost, pair_obs ^ rem_obs))
        for index, other in enumerate(remaining):
            rest = remaining[:index] + remaining[index + 1 :]
            pair_cost, pair_obs = self._pair_cost(first, other)
            rem_cost, rem_obs = self._match(rest)
            candidates.append((pair_cost + rem_cost, pair_obs ^ rem_obs))
        return min(candidates, key=lambda item: item[0])


def _t1_logical_error_rate(
    layout: _Layout,
    noise_model: _NoiseModel,
    *,
    rounds: int,
    basis: Basis,
    initial: str,
) -> dict[str, object]:
    duration_us = (
        rounds
        * (
            noise_model.layer1_duration_ns
            + noise_model.layer2_duration_ns
            + noise_model.readout_duration_ns
        )
        + noise_model.readout_duration_ns
    ) / 1000.0
    probabilities: list[float] = []
    for data_index in range(0, len(layout.qubits), 2):
        t1 = noise_model.qubit_calibrations[data_index].t1_us
        if basis == "bit" and initial == "0":
            p = 0.0
        elif basis == "bit":
            p = 1.0 - math.exp(-duration_us / t1)
        else:
            p = 0.5 * (1.0 - math.exp(-duration_us / (2.0 * t1)))
        probabilities.append(_clip_probability(p))
    threshold = layout.distance // 2 + 1
    return {
        "t1_total_duration_us": duration_us,
        "t1_data_error_probabilities": probabilities,
        "t1_logical_error_rate": _poisson_binomial_tail(probabilities, threshold),
    }


def _poisson_binomial_tail(probabilities: Sequence[float], threshold: int) -> float:
    distribution = np.zeros(len(probabilities) + 1, dtype=float)
    distribution[0] = 1.0
    for probability in probabilities:
        next_distribution = np.zeros_like(distribution)
        for count in range(len(probabilities)):
            next_distribution[count] += distribution[count] * (1.0 - probability)
            next_distribution[count + 1] += distribution[count] * probability
        distribution = next_distribution
    return float(np.sum(distribution[threshold:]))


def _load_measured_payloads(measured_results: Any | None) -> list[Mapping[str, object]]:
    if measured_results is None:
        return []
    if isinstance(measured_results, (str, Path)):
        return [_load_measured_payload_path(Path(measured_results))]
    if isinstance(measured_results, Result):
        return [measured_results.data]
    if isinstance(measured_results, Mapping):
        return [measured_results]
    return [
        payload
        for item in measured_results
        for payload in _load_measured_payloads(item)
    ]


def _load_measured_payload_path(path: Path) -> Mapping[str, object]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, Mapping) and isinstance(obj.get("data"), Mapping):
            return obj["data"]
        if isinstance(obj, Mapping):
            return obj
    if suffix in {".pickle", ".pkl"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)  # noqa: S301
        if isinstance(obj, Result):
            return obj.data
        if isinstance(obj, Mapping) and isinstance(obj.get("data"), Mapping):
            return obj["data"]
        if isinstance(obj, Mapping):
            return obj
    raise ValueError(f"Unsupported measured payload file: {path}")


def _measured_summary_row(payload: Mapping[str, object]) -> dict[str, object]:
    logical_error_rate = _optional_float(payload.get("logical_error_rate"))
    shots = _optional_int(payload.get("shots_valid"))
    if shots is None:
        shots = _optional_int(payload.get("num_shot"))
    if shots is None:
        logical_errors = payload.get("logical_errors")
        if logical_errors is not None:
            shots = len(np.asarray(logical_errors))
    stderr = (
        math.sqrt(logical_error_rate * (1.0 - logical_error_rate) / shots)
        if logical_error_rate is not None and shots and shots > 0
        else None
    )
    return {
        "source": "measured",
        "basis": str(payload.get("basis", "")),
        "initial": str(payload.get("initial", "")),
        "num_round": _optional_int(payload.get("rounds"))
        or _optional_int(payload.get("num_round")),
        "distance": _optional_int(payload.get("distance")),
        "num_shot": shots,
        "logical_error_rate": logical_error_rate,
        "stderr": stderr,
    }


def _simulation_summary_row(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "source": "simulation",
        "basis": str(payload.get("basis", "")),
        "initial": str(payload.get("initial", "")),
        "num_round": _optional_int(payload.get("rounds")),
        "distance": _optional_int(payload.get("distance")),
        "num_shot": _optional_int(payload.get("n_shots")),
        "logical_error_rate": _optional_float(payload.get("logical_error_rate")),
        "stderr": _optional_float(payload.get("stderr")),
        "t1_logical_error_rate": _optional_float(payload.get("t1_logical_error_rate")),
    }


def _comparison_rows(
    measured_rows: Sequence[Mapping[str, object]],
    simulation_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    measured_by_key = {
        (row.get("basis"), row.get("initial"), row.get("num_round")): row
        for row in measured_rows
    }
    rows: list[dict[str, object]] = []
    for sim_row in simulation_rows:
        key = (sim_row.get("basis"), sim_row.get("initial"), sim_row.get("num_round"))
        measured = measured_by_key.get(key, {})
        measured_rate = _optional_float(measured.get("logical_error_rate"))
        simulation_rate = _optional_float(sim_row.get("logical_error_rate"))
        rows.append(
            {
                "basis": sim_row.get("basis"),
                "initial": sim_row.get("initial"),
                "num_round": sim_row.get("num_round"),
                "distance": sim_row.get("distance"),
                "measured_logical_error_rate": measured_rate,
                "measured_stderr": _optional_float(measured.get("stderr")),
                "measured_num_shot": measured.get("num_shot"),
                "simulation_logical_error_rate": simulation_rate,
                "simulation_stderr": _optional_float(sim_row.get("stderr")),
                "simulation_num_shot": sim_row.get("num_shot"),
                "t1_logical_error_rate": _optional_float(
                    sim_row.get("t1_logical_error_rate")
                ),
                "simulation_minus_measured": None
                if measured_rate is None or simulation_rate is None
                else simulation_rate - measured_rate,
            }
        )
    return rows


def _correlation_comparisons(
    measured_payloads: Sequence[Mapping[str, object]],
    simulations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    measured_by_key = {
        (
            payload.get("basis"),
            payload.get("initial"),
            payload.get("rounds") or payload.get("num_round"),
        ): payload
        for payload in measured_payloads
        if payload.get("correlation_matrix") is not None
    }
    rows: list[dict[str, object]] = []
    for simulation in simulations:
        key = (
            simulation.get("basis"),
            simulation.get("initial"),
            simulation.get("rounds") or simulation.get("num_round"),
        )
        measured = measured_by_key.get(key)
        if measured is None or simulation.get("correlation_matrix") is None:
            continue
        measured_matrix = np.asarray(measured["correlation_matrix"], dtype=float)
        simulation_matrix = np.asarray(simulation["correlation_matrix"], dtype=float)
        if measured_matrix.shape != simulation_matrix.shape:
            rows.append(
                {
                    "basis": simulation.get("basis"),
                    "initial": simulation.get("initial"),
                    "num_round": simulation.get("rounds"),
                    "matched": False,
                    "reason": "matrix shape mismatch",
                    "measured_shape": measured_matrix.shape,
                    "simulation_shape": simulation_matrix.shape,
                }
            )
            continue
        difference = simulation_matrix - measured_matrix
        rows.append(
            {
                "basis": simulation.get("basis"),
                "initial": simulation.get("initial"),
                "num_round": simulation.get("rounds"),
                "matched": True,
                "mean_abs_difference": float(np.nanmean(np.abs(difference))),
                "max_abs_difference": float(np.nanmax(np.abs(difference))),
                "difference_matrix": difference,
            }
        )
    return rows


def _load_yaml_file(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _finite_float(value: object, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _optional_float(value: object) -> float | None:
    return _finite_float(value, default=None)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clip_probability(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _matplotlib_pyplot() -> Any:
    import matplotlib.pyplot as plt

    return plt


__all__ = [
    "is_stim_available",
    "plot_repetition_code_noisy_simulation",
    "repetition_code_noisy_simulation",
]

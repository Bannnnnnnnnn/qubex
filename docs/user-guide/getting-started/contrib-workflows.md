# Community-contributed workflows

Some additional workflows in Qubex are provided as community-contributed
functions under `qubex.contrib` rather than as core `Experiment` methods.

This page is mainly a migration note for existing users. If an older notebook
or script calls an `Experiment` helper that is no longer available there, use
the corresponding contrib function instead and pass `exp` as the first
argument.

```python
from qubex import contrib
```

## Moved APIs

Use this mapping when updating older notebooks or scripts:

| Old call on `exp` | New contrib function |
| --- | --- |
| `exp.measure_cr_crosstalk(...)` | `contrib.measure_cr_crosstalk(exp, ...)` |
| `exp.cr_crosstalk_hamiltonian_tomography(...)` | `contrib.cr_crosstalk_hamiltonian_tomography(exp, ...)` |
| `exp._simultaneous_measurement_coherence(...)` | `contrib.simultaneous_coherence_measurement(exp, ...)` |
| `exp._stark_t1_experiment(...)` | `contrib.stark_t1_experiment(exp, ...)` |
| `exp._stark_ramsey_experiment(...)` | `contrib.stark_ramsey_experiment(exp, ...)` |
| `exp.purity_benchmarking(...)` | `contrib.purity_benchmarking(exp, ...)` |
| `exp.interleaved_purity_benchmarking(...)` | `contrib.interleaved_purity_benchmarking(exp, ...)` |

## JPA calibration

### Basic call

The public API needs only an experiment and one qubit label:

```python
from qubex import contrib

result = contrib.calibrate_jpa(exp, "Q22")
print(result["optimal_parameters"])
```

`"Q22"` is an anchor used to identify its readout MUX. The calibration is not
limited to that qubit: it measures every active, valid qubit on the same MUX
and accepts a setting only when all of those peers satisfy the constraints.
The default parameter ranges are derived from the MUX configuration, followed
by a coarse scan and, when applicable, a local fine scan.

### OFF baseline and selection rules

Before scanning candidate settings, the workflow measures a completely OFF
baseline with DC voltage `0.0` V and pump amplitude `0.0`. Pump frequency has
no effect while the amplitude is zero. The baseline uses exactly the same
qubits, readout-amplitude choice, readout duration, shot count, and shot
interval as the candidate scan, so candidate and baseline values can be
compared directly.

For each peer qubit and grid point, the workflow records two raw quantities:

- `score`: noise-normalized separation between the measured g- and e-state IQ
  clouds; larger is better.
- `flatness`: the larger principal-axis standard-deviation ratio from the g
  and e IQ clouds; `1.0` is circular, and larger values are more elongated.

It also calculates these baseline-relative quantities for each peer:

```text
score_gain      = candidate score / OFF-baseline score
flatness_ratio  = candidate flatness / OFF-baseline flatness
```

By default, a point is valid only if every peer has `score_gain >= 1.05` and
`flatness_ratio <= 1.1`. Among valid points, calibration maximizes the worst
peer's score gain. In other words, one qubit's large improvement cannot hide a
degradation of another qubit on the shared MUX.

The constraints can be made explicit in a measurement notebook:

```python
result = contrib.calibrate_jpa(
    exp,
    "Q22",
    n_shots=512,
    minimum_score_gain=1.05,
    maximum_flatness_ratio=1.1,
    flatness_threshold=None,
)
```

- `minimum_score_gain` is the minimum candidate/OFF score ratio required for
  every peer. For example, `1.05` requires at least a 5% improvement for each
  qubit.
- `maximum_flatness_ratio` limits how much each peer's flatness may increase
  relative to its own OFF baseline. For example, `1.1` permits at most a 10%
  increase.
- `flatness_threshold` is an optional absolute flatness cap in addition to the
  relative constraint. Its default is `None`, which disables the absolute
  cap. Set it, for example to `1.5`, only when an independently justified
  absolute limit is needed. A fixed value such as `1.2` can reject every point
  when an OFF cloud is naturally anisotropic, even though the JPA did not cause
  that anisotropy.

If `readout_amplitude` is omitted, every peer uses its configured amplitude.
Passing one value overrides the amplitude for all evaluated peers. Always
compare ON and OFF behavior under the same choice; the automatic baseline does
this for you.

### Inspecting the result

The returned `Result` contains both the selected point and the evidence used
to select it:

```python
print(result["baseline"])
print(result["score"], result["score_gain"])
print(result["scores_by_qubit"])
print(result["score_gains_by_qubit"])
print(result["flatness_by_qubit"])
print(result["flatness_ratios_by_qubit"])

scan = result["fine_scan"] or result["coarse_scan"]
raw_q22 = scan["scores"]["Q22"]
gain_q22 = scan["score_gains"]["Q22"]
flatness_q22 = scan["flatness"]["Q22"]
flatness_ratio_q22 = scan["flatness_ratios"]["Q22"]
valid = scan["valid_mask"]
```

The raw scan arrays retain the physical score and flatness values. Their
relative arrays retain the ratios to the one OFF baseline, while
`aggregate_score_gain` is the worst-peer gain at each grid point.
`valid_mask` shows which points met every enabled constraint.

### When no point passes

Completing a scan does not imply that an acceptable JPA setting exists. If no
point passes all peer constraints, `calibrate_jpa` raises
`JPAConstraintError`. The completed raw and relative scans remain available in
`diagnostics`, rather than being discarded:

```python
from qubex import contrib

try:
    result = contrib.calibrate_jpa(
        exp,
        "Q22",
        n_shots=512,
        minimum_score_gain=1.05,
        maximum_flatness_ratio=1.1,
    )
except contrib.JPAConstraintError as exc:
    diagnostics = exc.diagnostics
    print(diagnostics["failure_stage"])
    print(diagnostics["baseline"])

    coarse = diagnostics["coarse_scan"]
    print(coarse["aggregate_score_gain"])
    print(coarse["valid_mask"])
```

Use those diagnostics to determine whether the scan range missed a useful
region, a particular peer was degraded, or a constraint was intentionally too
strict. Do not treat the numerically largest invalid point as a calibrated
setting.

The workflow is deliberately measurement-only. On success it returns a
candidate, but does **not** leave the selected DC/pump setting applied and does
**not** write `jpa_params.yaml` (or any other parameter file). On both success
and failure, the DC context restores the previous voltage and disables the
touched output. Review and validate the returned candidate before applying or saving
it through the laboratory's normal configuration procedure.

## Simultaneous coherence

```python
import numpy as np
from qubex import contrib

results = contrib.simultaneous_coherence_measurement(
    exp,
    targets=[Q0, Q1],
    time_range=np.arange(0, 20_001, 1000),
    n_shots=1024,
)

t1_result = results["T1"]
t1_result.plot()
```

## Stark-driven characterization

```python
from qubex import contrib

stark_result = contrib.stark_t1_experiment(
    exp,
    targets=[Q0],
    stark_detuning=0.05,
    stark_amplitude=0.1,
    n_shots=1024,
)

stark_result.plot()
```

## Purity benchmarking

```python
from qubex import contrib

pb_result = contrib.purity_benchmarking(
    exp,
    targets=[Q0],
    n_shots=1024,
)

print(pb_result)
```

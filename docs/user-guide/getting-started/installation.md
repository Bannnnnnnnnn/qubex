# Installation

Qubex runs on Python 3.10+. When you are not using hardware backends, it is not tied to a specific operating system. Hardware backends require additional dependencies and typically assume a Linux host.
During the `v1.5.0 beta` period and until the official release, installing Qubex requires `uv`.
We plan to publish Qubex to PyPI for the official release, which will make `pip install` available.

## Prepare Python environment

We recommend using [uv](https://docs.astral.sh/uv/) to manage Python and virtual environments. `uv` is a unified tool for installing Python, creating virtual environments, and managing dependencies.

## Create virtual environment

It is best practice to install Qubex in a dedicated virtual environment to prevent conflicts with system packages or other projects.

Run one of the following commands in your project directory.

=== "uv"
    ```bash
    uv venv
    ```

=== "venv"
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

## Install Qubex

This section introduces installation options for standard use, backend-enabled setups, pinned versions, and development workflows.

### Install latest version

Use this option when you want the newest Qubex features from the repository.

```bash
uv pip install "qubex @ git+https://github.com/amachino/qubex.git"
```

### Install with QuEL-1 backend support for real hardware use (Linux)

Use this variant when you plan to use Qubex with real hardware on a Linux host.
If you only want to use `QuantumSimulator`, you do not need the `backend` or
`quel1` extra.

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git"
```

This installs the public QuEL-1 backend libraries required for hardware-backed
execution. `qubex[quel1]` is available as the equivalent explicit extra.

### Install with QuEL-3 support

Use this variant when you plan to use Qubex with QuEL-3 runtime support.

```bash
uv pip install "qubex[quel3] @ git+https://github.com/amachino/qubex.git"
```

`quel3` depends on `quelware-client`. If that package is not published on
PyPI, install it from the package source or index specified by your
environment.

### Install specific version

Use this option when you need a pinned version for reproducibility.

```bash
uv pip install "qubex[backend] @ git+https://github.com/amachino/qubex.git@<version>"
```

### Install for development

Use this setup to prepare a local environment for developing and testing Qubex.
Development commands in this repository assume a `uv`-managed environment.

```bash
git clone -b develop https://github.com/amachino/qubex.git
cd qubex
make sync
```

### Reproduce the dual-readout development branch

The `feature/dual-readout` branch uses pinned Git submodules for the
dual-readout-enabled `qxdriver-quel1` and `quelware-internal` sources. Your
GitHub account must be able to read `toshisumida/quelware-internal`.

For a new checkout, initialize the pinned submodules during cloning and install
the locked environment:

```bash
git clone --recurse-submodules \
  --branch feature/dual-readout \
  https://github.com/Bannnnnnnnnn/qubex.git
cd qubex
make sync
```

For an existing checkout, synchronize the changed submodule URLs before
installing:

```bash
git pull
git submodule sync --recursive
make sync
```

The resulting environment uses editable sources inside the checkout:

- `packages/qxdriver-quel1`
- `packages/quelware-internal/e7awghal`
- `packages/quelware-internal/quel_ic_config`

Avoid installing copies from unrelated local directories afterward, because
that overrides the locked source locations.

## Next steps

- Start with [Choose where to start](choose-where-to-start.md) to pick the entry point that matches your goal.
- If you are moving on to hardware-backed `Experiment` or `Low-level APIs`, continue with [System configuration](system-configuration.md).

# Environment & Hardware Record (Phase 0)

This document records the exact hardware, OS, and dependency-installation
findings from Phase 0, per `plan.md` Phase 0 deliverables ("documented
hardware spec committed to repo") and NFR3 ("document CPU/RAM ahead of
time"). All figures below are measured on the actual development machine
on 2026-09-10, not estimated.

## Host hardware

| Item | Value |
|---|---|
| CPU | Intel(R) Core(TM) i5-9500 @ 3.00GHz |
| Physical cores / logical processors | 6 / 6 |
| Total RAM | 7.95 GB |
| Host OS | Windows 11 Pro N, 64-bit, build 10.0.22621 |
| Host Python | 3.12.0 |

## Concrete-ML platform-compatibility finding (highest-risk unknown, tested first)

**Finding: Concrete-ML cannot be installed on native Windows.**

`pip install concrete-ml` on native Windows (Python 3.12.0, `.venv`)
fails during dependency resolution:

```
ERROR: Could not find a version that satisfies the requirement
concrete-ml-extensions==0.1.9 (from concrete-ml) (from versions: none)
ERROR: No matching distribution found for concrete-ml-extensions==0.1.9
```

This was confirmed to be a genuine, permanent platform gap (not a
transient resolver issue) by querying the PyPI JSON API for
`concrete-ml-extensions` directly: every published version (0.1.9, 0.2.0)
ships wheels only for `macosx_*` and `manylinux_*` (Linux) targets, for
Python 3.8-3.12. **No `win32` or `win_amd64` wheel exists for any
version.** `concrete-ml-extensions` is a compiled Rust/C++ extension
(the low-level FHE backend); there is no sdist fallback that builds on
Windows either. This matches Zama's documented supported-platform list
(Linux and macOS only); Windows requires WSL2.

**Decision (user-approved, 2026-09-10)**: set up the FHE toolchain inside
the WSL2 Ubuntu distribution already present on this machine, rather than
provisioning a cloud VM. The native Windows `.venv` remains the
environment for editor tooling and the non-FHE pipeline stages (data,
features, train, benchmark orchestration, analysis) that run identically
on both platforms.

## WSL2 environment

| Item | Value |
|---|---|
| Distro | Ubuntu 24.04.4 LTS (noble), WSL version 2 |
| WSL Python | 3.12.3 |
| WSL logical processors | 6 (matches host) |
| WSL total RAM | 3.8 GiB |
| WSL swap | 1.0 GiB |

**Important resource constraint**: WSL2's default memory allocation caps
at roughly half the host's physical RAM (no `.wslconfig` override is
present on this machine), so the FHE toolchain has **3.8 GB**, not the
host's 7.95 GB, to work with. This directly bounds feasible model size,
feature-tier width, and batch sizes for later FHE compilation/benchmark
phases (see `prd.md` NFR3) and must be treated as the operative memory
ceiling for all FHE work, not the host figure. If a later phase hits an
out-of-memory condition during compilation, increasing the WSL2 memory
limit via `.wslconfig` (up to the host's physical RAM) is the first
thing to try before concluding a configuration is infeasible.

### Concrete-ML installation under WSL2

`pip install concrete-ml==1.9.0` succeeds under WSL2 Ubuntu 24.04
(Python 3.12.3), after two build-toolchain prerequisites were installed:

- `cmake` + `build-essential` (`onnxoptimizer` builds a C++ extension
  from source and requires `cmake`; not present in a fresh Ubuntu WSL2
  image).
- `python3-dev` / `python3.12-dev` (the same `onnxoptimizer` build needs
  `Python.h`, provided by the Python development headers package, also
  not present by default).

With both installed, `pip install concrete-ml==1.9.0` completes cleanly
(`concrete-ml==1.9.0`, `concrete-python==2.10.0`,
`concrete-ml-extensions==0.1.9`, plus ~70 transitive dependencies
including `torch==2.3.1`). `import concrete.ml` and
`from concrete.ml.sklearn import LogisticRegression` both succeed.

### Second finding: native compilation fails when the venv lives under a Windows path with spaces

The first install attempt put the WSL venv inside the project directory
itself, at `/mnt/c/Users/Rupak/Downloads/ML PROJECTS/Fraud Detection
Using FHE/.venv-wsl` (the Windows filesystem, mounted into WSL2). Import
worked, but calling `.compile()` on a trained `concrete.ml.sklearn`
model failed:

```
RuntimeError: Can't emit artifacts: Command failed:
ld --shared -o /tmp/.../sharedlib.so /tmp/.../program.module-0.mlir.o
  /mnt/c/Users/Rupak/Downloads/ML PROJECTS/Fraud Detection Using FHE/.venv-wsl/lib/python3.12/site-packages/concrete_python.libs/libConcretelangRuntime-a373b010.so
  -rpath=... --disable-new-dtags
ld: cannot find /mnt/c/Users/Rupak/Downloads/ML: No such file or directory
ld: cannot find PROJECTS/Fraud: No such file or directory
...
```

Concrete-Python's native compiler backend invokes the system linker
(`ld`) with the venv's site-packages path unquoted; because the project
directory name contains spaces (`ML PROJECTS`, `Fraud Detection Using
FHE`), the linker command line gets split on those spaces and every
subsequent path fragment fails to resolve. This is a genuine bug in how
Concrete-Python constructs its linker invocation, not a Concrete-ML
install failure -- `import` and `.fit()` both worked fine; only native
`.compile()` broke, and only because of where the venv's `.so` files
happened to live.

**Fix**: move the WSL2 venv onto the Linux-native filesystem, at
`~/.venvs/fhe-fraud-detection` (no spaces anywhere in the path), leaving
the project source itself on the Windows-mounted `/mnt/c` path (accessed
from WSL2 as normal). This is also the standard WSL2 performance
recommendation independent of this bug (cross-filesystem `/mnt/c` I/O is
slower than native ext4). Re-running the identical install there, then a
compile smoke test, both succeeded:

```
COMPILE OK
CLEAR PREDICT OK: [1 1 1 1 0]
FHE-SIMULATE PREDICT OK: [1 1]
```

(4-feature toy Logistic Regression, `n_bits=4`, compiled and run through
both clear and FHE-simulate prediction paths -- this is a basic
compile/predict smoke test only, not the held-out correctness validation
required before any model is benchmarked; that is Phase 5's job.)

**Practical implication for every later WSL2/FHE phase**: any Concrete-ML
venv, and by extension anything that ends up inside a compiled circuit's
build path, must live under a space-free path on the Linux filesystem
(e.g. under `~/`), never directly under a Windows path mounted at
`/mnt/c` if that path contains spaces.

## Two-environment dependency layout

Because Concrete-ML only runs under WSL2/Linux, this project pins two
separate, non-overlapping requirement sets:

- `requirements.txt` -- core ML/data/config/logging/test dependencies,
  installable and runnable identically on Windows and WSL2/Linux. Used by
  the Windows `.venv`.
- `requirements-fhe.txt` -- the full, self-contained `pip freeze` of the
  working WSL2 FHE venv (Concrete-ML plus every transitive dependency,
  including its own numpy/pandas/scikit-learn/xgboost versions). Used
  *instead of* `requirements.txt` by the WSL2 venv -- never install both
  in the same environment.

**Known version conflict (documented, not silently resolved)**:
`concrete-ml==1.9.0` requires exactly `numpy==1.26.4`, and constrains
`scikit-learn` and `xgboost` to older major versions (`1.5.0` /
`1.6.2`) than what resolves freely on Windows (`1.9.0` / `3.4.1` here).
`pandas` happens to resolve to the same version (`3.0.5`) in both
environments, coincidentally. This means: **a plaintext model trained
under the Windows venv's package versions is not guaranteed to
deserialize/behave identically when loaded under the WSL2 FHE venv's
older scikit-learn/xgboost/numpy.** This must be handled explicitly
starting in Phase 5 (FHE PoC) -- e.g. by training models intended for FHE
compilation directly inside the WSL2 venv, or by re-verifying
deserialization compatibility before compiling a Windows-trained model.
Not resolved in Phase 0; flagged here for Phase 5 to address, per
`CLAUDE.md` §17 (handling uncertainty empirically, not guessing).

This two-lock-file split is itself a variance from the single generic
"pin dependencies" instruction in `plan.md` Phase 0, made necessary by
the Windows platform gap documented above; recorded here per the
"document deviations" rule in `CLAUDE.md` §2.

## How to reproduce this environment

```bash
# Windows side (non-FHE tooling, tests, editor)
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# WSL2 side (FHE compilation/inference work) -- venv MUST be on the
# Linux-native filesystem (no spaces in path), not under /mnt/c, per the
# linker bug documented above.
wsl -d Ubuntu
sudo apt-get update
sudo apt-get install -y cmake build-essential python3-dev python3.12-dev python3.12-venv
python3 -m venv ~/.venvs/fhe-fraud-detection
~/.venvs/fhe-fraud-detection/bin/python -m pip install -r requirements-fhe.txt
# Run project code/tests from the Windows-mounted project directory:
cd "/mnt/c/Users/Rupak/Downloads/ML PROJECTS/Fraud Detection Using FHE"
~/.venvs/fhe-fraud-detection/bin/python -m pytest
```

## Phase 0 verification summary

| Check | Environment | Result |
|---|---|---|
| Core deps import + version-pin match | Windows `.venv` | Pass (16 passed, 1 skipped for concrete-ml) |
| Core deps import + version-pin match | WSL2 `~/.venvs/fhe-fraud-detection` | Pass (17 passed) |
| `import concrete.ml` | WSL2 | Pass |
| `LogisticRegression(...).compile()` + clear predict + FHE-simulate predict | WSL2 | Pass |
| `import concrete-ml` (native Windows) | Windows | **Fails** -- no wheel exists for any platform tag; documented above |

# Changelog

All notable changes to XAD's Python bindings will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The package version follows the [XAD](https://github.com/auto-differentiation/xad)
C++ core it is built on.


## [unreleased]

### Added

- XAD as a shared library in the package, with headers and CMake config, for building
  extensions sharing the tape (`xad.get_include()`, `xad.get_cmake_dir()`,
  `xad.get_library_dir()`).
- Complex numbers: `Complex` in `adj_1st` and `fwd_1st` with active real and imaginary parts,
  operators with `Real`, float and `complex`, and `xad.cmath` mirroring `cmath`
  ([#2](https://github.com/auto-differentiation/xad-py/issues/2)).
- Second-order derivatives: `fwd_adj_2nd` and `fwd_fwd_2nd` modes, with `Complex` and `xad.math`;
  `xad.value`/`xad.derivative` nest as in C++; `xad.jacobian` and `xad.hessian`
  ([#3](https://github.com/auto-differentiation/xad-py/issues/3)).
- Checkpointing in `adj_1st` and `fwd_adj_2nd`: `CheckpointCallback`, `Tape.insertCallback`,
  `pushCallback`, `clearCallbacks`, `getAndResetOutputAdjoint` and `incrementAdjoint`,
  `Real.getSlot`, `x.value = v` (`adj_1st`) and `ScopedNestedRecording`
  ([#4](https://github.com/auto-differentiation/xad-py/issues/4)).
- External functions in `adj_1st` and `fwd_adj_2nd`: functions computed outside XAD with a
  hand-written adjoint through `CheckpointCallback`, with
  [samples/external_function.py](samples/external_function.py)
  ([#5](https://github.com/auto-differentiation/xad-py/issues/5)).
- NumPy arrays: `Array` in `adj_1st` and `fwd_1st`, n-dimensional arrays of active values in
  C++ with elementwise operators and ufuncs, broadcasting, `sum`/`prod`/`mean`, `@`/`dot`,
  `where`, `concatenate`, views for indexing, slicing and `reshape`, and
  `Tape.registerInputs`/`registerOutputs` on arrays, with
  [samples/numpy_arrays.py](samples/numpy_arrays.py)
  ([#6](https://github.com/auto-differentiation/xad-py/issues/6)).

### Changed

- Thread-local tapes: each thread has its own active tape.
- Dynamic MSVC runtime on Windows.
- Assigning an unknown attribute to a `Real` or `Tape` raises `AttributeError`; it was stored
  silently, so a typo such as `x.derivatve = 1.0` was lost.

### Fixed

- An XAD active type of any mode or order is no longer converted implicitly to a float argument.
- `import xad` alone sets up every mode's `Real` (properties such as `fwd_1st.Real.derivative`).
- An input registered on an earlier tape and registered again on a new tape gets its adjoints
  there (it was skipped, giving 0 or, with many inputs, a crash); also for `Array` inputs
  ([#54](https://github.com/auto-differentiation/xad-py/pull/54)).

## [2.1.0] - 2026-10-24

### Added

- `float(x)` for `Real` (`__float__`).
- Wheels for Linux aarch64 and macOS arm64.
- Type stubs and `py.typed` marker in the package.

### Changed

- Built on XAD 2.1.0 and pybind11 3.1.
- Python 3.10 to 3.14 supported; 3.8 and 3.9 dropped.
- Build switched to scikit-build-core; CI modernised (current runners, cibuildwheel 4,
  lint and test workflow).

### Fixed

- Pickling `Real` works with all protocols (0/1 crashed, 2+ lost the value).
  An unpickled adjoint `Real` is passive (not on a tape).

## [1.5.2] - 2024-04-04

Version aligned with XAD 1.5.2 (bindings moved to their own repository).

- Thread-local tape disabled in the Python builds.

## [1.5.1.1] - 2024-03-29

Minor fixes to allow linking against QuantLib-Risks without duplicate symbols.

## [1.5.1] - 2024-03-28

Initial release of Python bindings for XAD.


[2.1.0]: https://github.com/auto-differentiation/xad-py/compare/v1.5.2...v2.1.0
[1.5.2]: https://github.com/auto-differentiation/xad-py/compare/v1.5.1.1...v1.5.2
[1.5.1.1]: https://github.com/auto-differentiation/xad-py/compare/v1.5.1...v1.5.1.1
[1.5.1]: https://github.com/auto-differentiation/xad-py/releases/tag/v1.5.1

# Changelog

All notable changes to XAD's Python bindings will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The package version follows the [XAD](https://github.com/auto-differentiation/xad)
C++ core it is built on.


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

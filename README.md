[![Python](https://img.shields.io/pypi/pyversions/xad.svg)](https://auto-differentiation.github.io/tutorials/python)

> Part of the [XAD ecosystem](https://github.com/auto-differentiation) — the fastest automatic differentiation library for C++.
> If this is useful to you, please star the [core XAD repository](https://github.com/auto-differentiation/xad).

XAD is a library designed for
[automatic differentiation](https://auto-differentiation.github.io/tutorials/aad/),
aimed at both beginners and advanced users. It is intended for use in
production environments, emphasizing performance and ease of use. The library
facilitates the computation of derivatives within computer programs, making
the process efficient and straightforward for a wide range of mathematical
functions, from simple arithmetic to complex calculations, ensuring accurate
and automatic derivative computations.

The Python bindings for XAD offer the following features:

- Support for both forward and adjoint modes at the first order.
- Strong exception-safety guarantees.
- High performance, as demonstrated in extensive production use.

For more details and to integrate XAD into your projects, consult the
comprehensive [documentation](https://auto-differentiation.github.io/tutorials/python).

## Application Areas

Automatic differentiation has many application areas, for example:

-   **Machine Learning and Deep Learning:** Training neural networks or other
    machine learning models.
-   **Optimization:** Solving optimization problems in engineering and finance.
-   **Numerical Analysis:** Enhancing numerical solution methods for
    differential equations.
-   **Scientific Computing:** Simulating physical systems and processes.
-   **Risk Management and Quantitative Finance:** Assessing and hedging risk in
    financial models.
-   **Computer Graphics:** Optimizing rendering algorithms.
-   **Robotics:** Improving control and simulation of robotic systems.
-   **Meteorology:** Enhancing weather prediction models.
-   **Biotechnology:** Modeling biological processes and systems.

## Try it in your browser

Run these notebooks on Google Colab, with nothing to install:

| Notebook | |
|---|---|
| [Quickstart](notebooks/01_quickstart.ipynb): gradients in adjoint and forward mode, checked against finite differences | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auto-differentiation/xad-py/blob/main/notebooks/01_quickstart.ipynb) |
| [Option Greeks](notebooks/02_option_greeks.ipynb): Black-Scholes and Monte Carlo price with all Greeks in one adjoint pass | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auto-differentiation/xad-py/blob/main/notebooks/02_option_greeks.ipynb) |
| [Adjoint vs finite differences](notebooks/03_adjoint_vs_finite_differences.ipynb): gradient of a diffusion model as the number of parameters grows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/auto-differentiation/xad-py/blob/main/notebooks/03_adjoint_vs_finite_differences.ipynb) |

## Getting Started

Install:

```text
pip install xad
```

Supports Python 3.10 to 3.14. The package version follows the
[XAD C++ core](https://github.com/auto-differentiation/xad) it is built on.

Calculate first-order derivatives in adjoint mode:

```python
import xad.adj_1st as xadj


# set independent variables
x0_ad = xadj.Real(1.0)
x1_ad = xadj.Real(1.5)
x2_ad = xadj.Real(1.3)
x3_ad = xadj.Real(1.2)

with xadj.Tape() as tape:
    # and register them
    tape.registerInput(x0_ad)
    tape.registerInput(x1_ad)
    tape.registerInput(x2_ad)
    tape.registerInput(x3_ad)

    # start recording derivatives
    tape.newRecording()

    # calculate the output
    y = x0_ad + x1_ad - x2_ad * x3_ad

    # register and seed adjoint of output
    tape.registerOutput(y)
    y.derivative = 1.0

    # compute all other adjoints
    tape.computeAdjoints()

    # output results
    print(f"y = {y}")
    print(f"first order derivatives:\n")
    print(f"dy/dx0 = {x0_ad.derivative}")
    print(f"dy/dx1 = {x1_ad.derivative}")
    print(f"dy/dx2 = {x2_ad.derivative}")
    print(f"dy/dx3 = {x3_ad.derivative}")
```


Complex numbers work the same way: `Complex` has active real and imaginary parts, and
`xad.cmath` mirrors Python's `cmath` (see [samples/complex.py](samples/complex.py)):

```python
import xad.adj_1st as xadj
import xad.cmath

x, y = xadj.Real(0.5), xadj.Real(1.2)
with xadj.Tape() as tape:
    tape.registerInput(x)
    tape.registerInput(y)
    tape.newRecording()

    z = xadj.Complex(x, y)
    w = z * xad.cmath.exp(z)

    tape.registerOutput(w.real)
    w.real.derivative = 1.0
    tape.computeAdjoints()
    print(f"dRe(w)/dx = {x.derivative}, dRe(w)/dy = {y.derivative}")
```

Second-order derivatives use nested modes, as in XAD's C++: `xad.fwd_adj_2nd` (forward over
adjoint) and `xad.fwd_fwd_2nd`. `xad.value(x)` and `xad.derivative(x)` return the inner first-order
parts, so `xad.derivative(xad.derivative(x))` is a second derivative (see
[samples/fwd_adj_2nd.py](samples/fwd_adj_2nd.py)). For whole matrices, `xad.jacobian` and
`xad.hessian` take a function of a list of `Real`s:

```python
import xad
from xad.math import sin, cos


def f(x):
    return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0])


print(xad.hessian(f, [1.0, 1.5, 1.3, 1.2]))
```

Checkpointing (in `adj_1st` and `fwd_adj_2nd`) replaces part of the recording with a
`CheckpointCallback`, whose `computeAdjoint(tape)` is called during `computeAdjoints`, for example
to recompute that part in a nested recording and save tape memory
(see [samples/checkpointing.py](samples/checkpointing.py)). The tape keeps inserted callbacks
alive, and `ScopedNestedRecording` is a `with` block:

```python
class Cb(xadj.CheckpointCallback):
    def computeAdjoint(self, tape):
        ybar = tape.getAndResetOutputAdjoint(self.y_slot)
        x = xadj.Real(self.x_value)
        tape.registerInput(x)
        with xadj.ScopedNestedRecording(tape) as nested:
            y = f(x)  # recompute actively
            tape.registerOutput(y)
            y.derivative = ybar
            nested.computeAdjoints()
            nested.incrementAdjoint(self.x_slot, x.derivative)

# forward run: compute y passively, store x_value and the slots, then tape.insertCallback(cb)
```

For more information, see the [Documentation](https://auto-differentiation.github.io/tutorials/python).

## Building Extensions Against XAD

The package ships XAD as a shared library, with its C++ headers and a CMake package config.
C++ extension modules linking it share the active tape with `xad.adj_1st.Tape`
(per thread):

```cmake
find_package(XAD CONFIG REQUIRED)
target_link_libraries(my_ext PRIVATE XAD::xad)
```

```bash
cmake -B build -DXAD_DIR=$(python -c "import xad; print(xad.get_cmake_dir())")
```

`xad.get_include()` and `xad.get_library_dir()` give the header and library directories.
Import `xad` before the extension so the library is loaded.

## Related Projects

- XAD Comprehensive automatic differentiation in [Python](https://github.com/auto-differentiation/xad-py) and [C++](https://github.com/auto-differentiation/xad)
- QuantLib-Risks: Fast risk evaluations in [Python](https://github.com/auto-differentiation/QuantLib-Risks-Py) and [C++](https://github.com/auto-differentiation/QuantLib-Risks-Cpp)

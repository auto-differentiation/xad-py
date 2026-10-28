/*******************************************************************************

   Exports the XAD complex math functions to Python (mirrors cmath).

   This file is part of XAD's Python bindings, a comprehensive library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <XAD/XAD.hpp>
#include <pybind11/pybind11.h>
#include <complex>
#include "complex.hpp"
#include "double_caster.hpp"

namespace py = pybind11;

using AReal = xad::AReal<double>;
using FReal = xad::FReal<double>;

template <class T>
void add_cmath_functions(py::module_& m)
{
    using C = std::complex<T>;
    m.def("exp", [](const C& z) { return C(std::exp(z)); }, "exponential function");
    m.def("log", [](const C& z) { return C(std::log(z)); }, "natural logarithm");
    m.def("log10", [](const C& z) { return C(std::log10(z)); }, "base 10 logarithm");
    m.def("sqrt", [](const C& z) { return C(std::sqrt(z)); }, "square root");
    m.def("sin", [](const C& z) { return C(std::sin(z)); }, "sine");
    m.def("cos", [](const C& z) { return C(std::cos(z)); }, "cosine");
    m.def("tan", [](const C& z) { return C(std::tan(z)); }, "tangent");
    m.def("asin", [](const C& z) { return C(std::asin(z)); }, "inverse sine");
    m.def("acos", [](const C& z) { return C(std::acos(z)); }, "inverse cosine");
    m.def("atan", [](const C& z) { return C(std::atan(z)); }, "inverse tangent");
    m.def("sinh", [](const C& z) { return C(std::sinh(z)); }, "hyperbolic sine");
    m.def("cosh", [](const C& z) { return C(std::cosh(z)); }, "hyperbolic cosine");
    m.def("tanh", [](const C& z) { return C(std::tanh(z)); }, "hyperbolic tangent");
    m.def("asinh", [](const C& z) { return C(std::asinh(z)); }, "inverse hyperbolic sine");
    m.def("acosh", [](const C& z) { return C(std::acosh(z)); }, "inverse hyperbolic cosine");
    m.def("atanh", [](const C& z) { return C(std::atanh(z)); }, "inverse hyperbolic tangent");
    m.def(
        "log", [](const C& z, double base) { return C(std::log(z) / std::log(base)); },
        "logarithm to the given base");
    m.def(
        "log", [](const C& z, const C& base) { return C(std::log(z) / std::log(base)); },
        "logarithm to the given base");
    m.def("phase", [](const C& z) { return T(std::arg(z)); }, "phase (argument) of z");
    m.def(
        "polar", [](const C& z) { return std::make_pair(T(std::abs(z)), T(std::arg(z))); },
        "polar coordinates (r, phi) of z");
    m.def(
        "rect", [](const T& r, const T& phi) { return C(std::polar(r, phi)); },
        "complex number from polar coordinates");
    m.def(
        "rect", [](const T& r, double phi) { return C(std::polar(r, T(phi))); },
        "complex number from polar coordinates");
    m.def(
        "rect", [](double r, const T& phi) { return C(std::polar(T(r), phi)); },
        "complex number from polar coordinates");
}

void py_cmath(py::module_& m)
{
    py::module_ m1 = m.def_submodule("cmath");

    add_cmath_functions<AReal>(m1);
    add_cmath_functions<FReal>(m1);
}

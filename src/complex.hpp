/*******************************************************************************

   Defines the bindings for the XAD complex active types.

   This file is part of XAD's Python bindings, a comprehensive library for
   automatic differentiation.

   Copyright (C) 2010-2024 Xcelerit Computing Ltd.

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
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <complex>
#include <optional>
#include <string>
#include "double_caster.hpp"

namespace py = pybind11;

// the value of a Python complex, nothing for any other type
inline std::optional<std::complex<double>> py_complex_value(const py::object& o)
{
    if (!PyComplex_Check(o.ptr()))
        return std::nullopt;
    return std::complex<double>(PyComplex_RealAsDouble(o.ptr()), PyComplex_ImagAsDouble(o.ptr()));
}

// the operator and its reflected form with a Complex, Real, float/int or Python complex operand
template <class T, class Op>
void py_complex_binary_op(py::class_<std::complex<T>>& c, const char* name, const char* rname,
                          Op op)
{
    using C = std::complex<T>;
    // a Python complex is passive: take it as std::complex<double>, else defer to the other type
    auto with_py_complex = [op](const C& z, const py::object& o, bool reflected) -> py::object
    {
        auto w = py_complex_value(o);
        if (!w)
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        return py::cast(reflected ? op(*w, z) : op(z, *w));
    };

    c.def(
         name, [op](const C& a, const C& b) { return op(a, b); }, py::is_operator())
        .def(
            name, [op](const C& a, const T& b) { return op(a, b); }, py::is_operator())
        .def(
            name, [op](const C& a, double b) { return op(a, b); }, py::is_operator())
        .def(
            name, [with_py_complex](const C& a, const py::object& b)
            { return with_py_complex(a, b, false); }, py::is_operator())
        .def(
            rname, [op](const C& a, const T& b) { return op(b, a); }, py::is_operator())
        .def(
            rname, [op](const C& a, double b) { return op(b, a); }, py::is_operator())
        .def(
            rname, [with_py_complex](const C& a, const py::object& b)
            { return with_py_complex(a, b, true); }, py::is_operator());
}

// z == o (or z != o when negated) for a Python complex o, NotImplemented for other types
template <class C>
py::object py_complex_eq(const C& z, const py::object& o, bool negated)
{
    auto w = py_complex_value(o);
    if (!w)
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    bool eq = z == C(*w);
    return py::bool_(eq != negated);
}

template <class T>
void py_complex(py::module_& m)
{
    using C = std::complex<T>;
    auto c = py::class_<C>(m, "Complex", "complex number with active real and imaginary parts");

    c.def(py::init<>())
        .def(py::init<double, double>(), py::arg("re"), py::arg("im"))
        .def(py::init<const T&, const T&>(), py::arg("re"), py::arg("im"))
        .def(py::init<double>(), py::arg("re"))
        .def(py::init<const T&>(), py::arg("re"))
        // from a Python complex: no pybind11/complex.h, its caster would also claim std::complex<T>
        .def(py::init(
                 [](const py::object& z)
                 {
                     auto w = py_complex_value(z);
                     if (!w)
                         throw py::type_error("Complex() argument must be a complex number");
                     return C(*w);
                 }),
             py::arg("z"))
        .def("__neg__", [](const C& z) { return C(-z); })
        .def("__pos__", [](const C& z) { return z; })
        .def("__abs__", [](const C& z) { return T(std::abs(z)); })
        .def(
            "conjugate", [](const C& z) { return C(std::conj(z)); }, "complex conjugate")
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self == T())
        .def(py::self != T())
        .def(py::self == double())
        .def(py::self != double())
        // a Python complex: compare as passive Complex, else defer to the other type
        .def(
            "__eq__", [](const C& a, const py::object& b) { return py_complex_eq(a, b, false); },
            py::is_operator())
        .def(
            "__ne__", [](const C& a, const py::object& b) { return py_complex_eq(a, b, true); },
            py::is_operator())
        .def("__complex__",
             [](const C& z)
             {
                 return py::reinterpret_steal<py::object>(
                     PyComplex_FromDoubles(z.real().getValue(), z.imag().getValue()));
             })
        .def("__repr__",
             [](const C& z)
             {
                 double im = z.imag().getValue();
                 return "(" + std::to_string(z.real().getValue()) + (std::signbit(im) ? "-" : "+") +
                        std::to_string(std::abs(im)) + "j)";
             })
        // pickle the values only: an unpickled adjoint Complex is passive (not on tape)
        .def(py::pickle([](const C& z)
                        { return py::make_tuple(z.real().getValue(), z.imag().getValue()); },
                        [](const py::tuple& t)
                        { return C(t[0].cast<double>(), t[1].cast<double>()); }))
        // py::pickle needs protocol 2+: use that reduction for protocols 0/1 too
        .def(
            "__reduce_ex__",
            [](const py::object& self, const py::object&)
            {
                return py::module_::import("builtins")
                    .attr("object")
                    .attr("__reduce_ex__")(self, 2);
            },
            py::arg("protocol"))
        // the parts themselves (not copies), keeping the Complex alive while they are referenced
        .def_property_readonly(
            "real", [](C& z) -> T& { return z.real(); }, py::return_value_policy::reference_internal,
            "real part")
        .def_property_readonly(
            "imag", [](C& z) -> T& { return z.imag(); }, py::return_value_policy::reference_internal,
            "imaginary part");

    py_complex_binary_op<T>(
        c, "__add__", "__radd__", [](const auto& a, const auto& b) { return C(a + b); });
    py_complex_binary_op<T>(
        c, "__sub__", "__rsub__", [](const auto& a, const auto& b) { return C(a - b); });
    py_complex_binary_op<T>(
        c, "__mul__", "__rmul__", [](const auto& a, const auto& b) { return C(a * b); });
    py_complex_binary_op<T>(
        c, "__truediv__", "__rtruediv__", [](const auto& a, const auto& b) { return C(a / b); });
    py_complex_binary_op<T>(
        c, "__pow__", "__rpow__", [](const auto& a, const auto& b) { return C(std::pow(a, b)); });
}

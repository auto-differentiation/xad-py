/*******************************************************************************

   Exports XAD's Jacobian and Hessian drivers to Python.

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

#include <XAD/Hessian.hpp>
#include <XAD/Jacobian.hpp>
#include <XAD/XAD.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "double_caster.hpp"
#include <functional>
#include <vector>

namespace py = pybind11;

// func, raising a TypeError with the given message when its result is not of the type expected
template <class R, class Arg>
std::function<R(Arg&)> py_checked(const std::function<R(Arg&)>& func, const char* msg)
{
    return [func, msg](Arg& x)
    {
        try
        {
            return func(x);
        }
        catch (const py::cast_error&)
        {
            throw py::type_error(msg);
        }
    };
}

void py_jacobian(py::module_& m)
{
    using AReal = xad::AReal<double>;
    using Func = std::function<std::vector<AReal>(std::vector<AReal>&)>;
    m.def(
        "jacobian",
        [](const Func& func, const std::vector<double>& x)
        {
            return xad::computeJacobian(
                std::vector<AReal>(x.begin(), x.end()),
                py_checked(func, "jacobian: func must return a list of adj_1st.Real"));
        },
        py::arg("func"), py::arg("x"),
        "Jacobian of func (list of adj_1st.Real -> list of adj_1st.Real) at x, in adjoint mode: "
        "rows are outputs, columns inputs. Records on the active tape, or a new one if none is");
}

void py_hessian(py::module_& m)
{
    using AReal = xad::fwd_adj<double>::active_type;
    using Func = std::function<AReal(std::vector<AReal>&)>;
    m.def(
        "hessian",
        [](const Func& func, const std::vector<double>& x)
        {
            return xad::computeHessian(
                std::vector<AReal>(x.begin(), x.end()),
                py_checked(func, "hessian: func must return one fwd_adj_2nd.Real"));
        },
        py::arg("func"), py::arg("x"),
        "Hessian of func (list of fwd_adj_2nd.Real -> fwd_adj_2nd.Real) at x, in forward over "
        "adjoint mode. Records on the active tape, or a new one if none is");
}

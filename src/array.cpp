/*******************************************************************************

   Instantiates the XAD active array bindings for each mode.

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

#include "double_caster.hpp"  // first: must precede any double caster use

#include "array.hpp"

void py_array_adj_1st(py::module_& m)
{
    using T = xad::AReal<double>;
    using Tape = xad::Tape<double>;
    py_array<T>(m)
        .def(
            "_register_inputs",
            [](Array<T>& a, Tape& t) { a.for_each([&](T& x) { t.registerInput(x); }); },
            py::arg("tape"), "register the elements on tape as inputs, in C order")
        .def(
            "_register_outputs",
            [](Array<T>& a, Tape& t) { a.for_each([&](T& x) { t.registerOutput(x); }); },
            py::arg("tape"), "register the elements on tape as outputs, in C order");
}

void py_array_fwd_1st(py::module_& m) { py_array<xad::FReal<double>>(m); }

/*******************************************************************************

   Main pybind module definition for the extension module.

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

#include <XAD/XAD.hpp>
#include <pybind11/pybind11.h>
#include "double_caster.hpp"  // first: must precede any double caster use
#include "cmath.hpp"
#include "complex.hpp"
#include "exceptions.hpp"
#include "jacobian_hessian.hpp"
#include "math.hpp"
#include "real.hpp"
#include "tape.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

// in array.cpp: pybind11/numpy.h brings pybind11/complex.h, whose caster would claim the
// std::complex<T> bound as Complex here
void py_array_adj_1st(py::module_& m);

void py_adj_1st(py::module_ &m)
{
    py::module_ adj = m.def_submodule("adj_1st");
    py_real<AReal>(adj);
    py_complex<AReal>(adj);
    py_array_adj_1st(adj);
    py_checkpoint_callback<xad::Tape<double>>(adj);
    py_tape<xad::Tape<double>>(adj);
    py_scoped_nested_recording<xad::Tape<double>>(adj);
}

void py_fwd_adj_2nd(py::module_ &m)
{
    using mode = xad::fwd_adj<double>;
    py::module_ fwd_adj = m.def_submodule("fwd_adj_2nd");
    py_real<mode::active_type>(fwd_adj);
    py_complex<mode::active_type>(fwd_adj);
    py_checkpoint_callback<mode::tape_type>(fwd_adj);
    py_tape<mode::tape_type>(fwd_adj);
    py_scoped_nested_recording<mode::tape_type>(fwd_adj);
}

void py_fwd_fwd_2nd(py::module_ &m)
{
    py::module_ fwd_fwd = m.def_submodule("fwd_fwd_2nd");
    py_real<xad::fwd_fwd<double>::active_type>(fwd_fwd);
    py_complex<xad::fwd_fwd<double>::active_type>(fwd_fwd);
}

void py_fwd_1st(py::module_ &m)
{
    py::module_ fwd = m.def_submodule("fwd_1st");
    py_real<FReal>(fwd);
    py_complex<FReal>(fwd);
}

PYBIND11_MODULE(_xad, m)
{
    py_adj_1st(m);
    py_fwd_1st(m);
    py_fwd_adj_2nd(m);
    py_fwd_fwd_2nd(m);
    py_math(m);
    py_cmath(m);
    py_exceptions(m);
    py_jacobian(m);
    py_hessian(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

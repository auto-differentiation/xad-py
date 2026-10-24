/*******************************************************************************

   Caster for double arguments that rejects XAD active types.

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
#include <pybind11/pybind11.h>

// pybind11's double caster, but rejecting XAD Reals: their __float__ must not convert them
// implicitly (drops derivatives, e.g. mixing adj/fwd). Explicit float(x) still works.
namespace pybind11
{
namespace detail
{
template <>
class type_caster<double>
{
  public:
    bool load(handle src, bool convert)
    {
        PyObject* o = src.ptr();
        if (!o)
            return false;
        if (!PyFloat_Check(o) && !PyLong_Check(o) &&
            (!convert || isinstance<xad::AReal<double>>(src) ||
             isinstance<xad::FReal<double>>(src)))
            return false;
        double d = PyFloat_AsDouble(o);
        if (d == -1.0 && PyErr_Occurred())
        {
            PyErr_Clear();
            return false;
        }
        value = d;
        return true;
    }

    static handle cast(double src, return_value_policy, handle) { return PyFloat_FromDouble(src); }

    PYBIND11_TYPE_CASTER(double, io_name("typing.SupportsFloat | typing.SupportsIndex", "float"));
};
}  // namespace detail
}  // namespace pybind11

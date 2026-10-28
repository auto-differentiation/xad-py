/*******************************************************************************

   Caster for double arguments that rejects XAD active types.

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

#include <pybind11/pybind11.h>

// pybind11's double caster, but rejecting the XAD active types (the types bound in xad._xad:
// Reals of any mode and order, Complex): their __float__ must not convert them implicitly
// (drops derivatives, e.g. mixing adj/fwd or orders). Explicit float(x) still works.
namespace pybind11
{
namespace detail
{
// whether src's (base) bound type is defined in xad._xad or its submodules, not in another
// extension sharing pybind11's type registry
inline bool is_xad_type(handle src)
{
    const type_info* tinfo = get_type_info(Py_TYPE(src.ptr()));
    if (!tinfo)
        return false;
    const std::string mod = str(getattr(handle((PyObject*)tinfo->type), "__module__", str("")));
    return mod == "xad._xad" || mod.rfind("xad._xad.", 0) == 0;
}

template <>
class type_caster<double>
{
  public:
    bool load(handle src, bool convert)
    {
        PyObject* o = src.ptr();
        if (!o)
            return false;
        if (!PyFloat_Check(o) && !PyLong_Check(o) && (!convert || is_xad_type(src)))
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

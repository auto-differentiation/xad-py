/*******************************************************************************

   Defines the bindings for the XAD active arrays.

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
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include "double_caster.hpp"

namespace py = pybind11;

// n-d array of active values, C-contiguous
template <class T>
class Array
{
  public:
    using float_array = py::array_t<double, py::array::c_style | py::array::forcecast>;

    explicit Array(const float_array& a)
        : shape_(a.shape(), a.shape() + a.ndim()), data_(a.data(), a.data() + a.size())
    {
    }

    py::tuple shape() const
    {
        py::tuple t(shape_.size());
        for (std::size_t i = 0; i < shape_.size(); ++i)
            t[i] = shape_[i];
        return t;
    }

    float_array values() const
    {
        float_array out(shape_);
        double* p = out.mutable_data();
        for (const T& x : data_)
            *p++ = xad::value(x);
        return out;
    }

    float_array derivatives() const
    {
        float_array out(shape_);
        double* p = out.mutable_data();
        for (const T& x : data_)
            *p++ = x.getDerivative();
        return out;
    }

    T& item(py::ssize_t i)
    {
        if (i < 0 || i >= static_cast<py::ssize_t>(data_.size()))
            throw py::index_error("array index out of range");
        return data_[static_cast<std::size_t>(i)];
    }

  private:
    std::vector<py::ssize_t> shape_;
    std::vector<T> data_;
};

template <class T>
void py_array(py::module_& m)
{
    py::class_<Array<T>>(m, "Array", "n-dimensional array of active values")
        .def(py::init<const typename Array<T>::float_array&>(), py::arg("values"),
             "an array from a float array or (nested) sequence of floats")
        .def_property_readonly("shape", &Array<T>::shape, "the shape of the array")
        .def_property_readonly("values", &Array<T>::values,
                               "the values of the elements as a float array")
        .def_property_readonly("derivatives", &Array<T>::derivatives,
                               "the derivatives of the elements as a float array")
        .def("__getitem__", &Array<T>::item, py::arg("index"),
             py::return_value_policy::reference_internal, "the element itself, not a copy");
}

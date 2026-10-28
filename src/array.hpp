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
#include "double_caster.hpp"
#include <cstddef>
#include <utility>
#include <vector>

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
        for (std::size_t i = 0; i < shape_.size(); ++i) t[i] = shape_[i];
        return t;
    }

    float_array values() const
    {
        float_array out(shape_);
        double* p = out.mutable_data();
        for (const T& x : data_) *p++ = xad::value(x);
        return out;
    }

    float_array derivatives() const
    {
        float_array out(shape_);
        double* p = out.mutable_data();
        for (const T& x : data_) *p++ = x.getDerivative();
        return out;
    }

    T& item(py::ssize_t i)
    {
        const auto n = static_cast<py::ssize_t>(data_.size());
        if (i < 0)
            i += n;
        if (i < 0 || i >= n)
            throw py::index_error("array index out of range");
        return data_[static_cast<std::size_t>(i)];
    }

    // elementwise op(a_i, b_i), same shapes
    template <class Op>
    Array map(const Array& b, Op op) const
    {
        if (shape_ != b.shape_)
            throw py::value_error("operands could not be broadcast together");
        std::vector<T> r(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) r[i] = op(data_[i], b.data_[i]);
        return Array(shape_, std::move(r));
    }

    // elementwise op(a_i)
    template <class Op>
    Array map(Op op) const
    {
        std::vector<T> r(data_.size());
        for (std::size_t i = 0; i < data_.size(); ++i) r[i] = op(data_[i]);
        return Array(shape_, std::move(r));
    }

    T sum() const
    {
        T s = 0.0;
        for (const T& x : data_) s += x;
        return s;
    }

  private:
    Array(std::vector<py::ssize_t> shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data))
    {
    }

    std::vector<py::ssize_t> shape_;
    std::vector<T> data_;
};

// the operator and its reflected form with an Array, Real or float/int operand
template <class T, class Op>
void py_array_binary_op(py::class_<Array<T>>& c, const char* name, const char* rname, Op op)
{
    using A = Array<T>;
    c.def(
         name, [op](const A& a, const A& b) { return a.map(b, op); }, py::is_operator())
        .def(
            name, [op](const A& a, const T& y)
            { return a.map([&](const T& x) -> T { return op(x, y); }); }, py::is_operator())
        .def(
            name, [op](const A& a, double y)
            { return a.map([&](const T& x) -> T { return op(x, y); }); }, py::is_operator())
        .def(
            rname, [op](const A& a, const T& y)
            { return a.map([&](const T& x) -> T { return op(y, x); }); }, py::is_operator())
        .def(
            rname, [op](const A& a, double y)
            { return a.map([&](const T& x) -> T { return op(y, x); }); }, py::is_operator());
}

template <class T>
void py_array(py::module_& m)
{
    auto c = py::class_<Array<T>>(m, "Array", "n-dimensional array of active values")
                 .def(py::init<const typename Array<T>::float_array&>(), py::arg("values"),
                      "an array from a float array or (nested) sequence of floats")
                 .def_property_readonly("shape", &Array<T>::shape, "the shape of the array")
                 .def_property_readonly("values", &Array<T>::values,
                                        "the values of the elements as a float array")
                 .def_property_readonly("derivatives", &Array<T>::derivatives,
                                        "the derivatives of the elements as a float array")
                 .def("__getitem__", &Array<T>::item, py::arg("index"),
                      py::return_value_policy::reference_internal, "the element itself, not a copy")
                 .def("sum", &Array<T>::sum, "the sum of all elements");
    py_array_binary_op(c, "__add__", "__radd__",
                       [](const auto& x, const auto& y) { return T(x + y); });
    py_array_binary_op(c, "__sub__", "__rsub__",
                       [](const auto& x, const auto& y) { return T(x - y); });
    py_array_binary_op(c, "__mul__", "__rmul__",
                       [](const auto& x, const auto& y) { return T(x * y); });
    py_array_binary_op(c, "__truediv__", "__rtruediv__",
                       [](const auto& x, const auto& y) { return T(x / y); });
}

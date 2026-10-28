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
#include <algorithm>
#include <cstddef>
#include <map>
#include <string>
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

    // a 0-d array holding x (an active scalar operand, broadcast like a NumPy scalar)
    explicit Array(const T& x) : data_{x} {}

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

    // the element at a full index (one int per dimension, negative counts from the end)
    T& item(const py::tuple& index)
    {
        if (index.size() != shape_.size())
            throw py::index_error("array index must have one int per dimension");
        std::size_t flat = 0;
        for (std::size_t d = 0; d < shape_.size(); ++d)
        {
            auto i = index[d].cast<py::ssize_t>();
            if (i < 0)
                i += shape_[d];
            if (i < 0 || i >= shape_[d])
                throw py::index_error("array index out of range");
            flat = flat * static_cast<std::size_t>(shape_[d]) + static_cast<std::size_t>(i);
        }
        return data_[flat];
    }

    // elementwise op(a_i, b_i), broadcasting the shapes as NumPy does
    template <class Op>
    Array map(const Array& b, Op op) const
    {
        if (shape_ == b.shape_)
        {
            std::vector<T> r(data_.size());
            for (std::size_t i = 0; i < data_.size(); ++i) r[i] = op(data_[i], b.data_[i]);
            return Array(shape_, std::move(r));
        }
        const std::size_t nd = std::max(shape_.size(), b.shape_.size());
        std::vector<py::ssize_t> shape(nd), sa = broadcast_strides(nd),
                                            sb = b.broadcast_strides(nd);
        for (std::size_t d = 0; d < nd; ++d)
        {
            const py::ssize_t na = extent(nd, d), nb = b.extent(nd, d);
            if (na != nb && na != 1 && nb != 1)
                throw py::value_error("operands could not be broadcast together");
            shape[d] = na == 1 ? nb : na;
        }
        std::size_t n = 1;
        for (py::ssize_t e : shape) n *= static_cast<std::size_t>(e);
        std::vector<T> r(n);
        std::vector<py::ssize_t> idx(nd, 0);
        py::ssize_t ia = 0, ib = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            r[i] = op(data_[static_cast<std::size_t>(ia)], b.data_[static_cast<std::size_t>(ib)]);
            // next index, last dimension fastest
            for (std::size_t d = nd; d-- > 0;)
            {
                ia += sa[d];
                ib += sb[d];
                if (++idx[d] < shape[d])
                    break;
                ia -= sa[d] * shape[d];
                ib -= sb[d] * shape[d];
                idx[d] = 0;
            }
        }
        return Array(std::move(shape), std::move(r));
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

    std::size_t ndim() const { return shape_.size(); }

    py::ssize_t dim(std::size_t d) const { return shape_[d]; }

    // fold of op, starting from init, over the dimensions flagged in reduced; keepdims keeps them
    // with extent 1
    template <class Op>
    Array reduce(const std::vector<bool>& reduced, bool keepdims, double init, Op op) const
    {
        const std::size_t nd = shape_.size();
        std::vector<py::ssize_t> shape;
        // strides of the result in the input's index space, 0 along reduced dimensions
        std::vector<std::size_t> strides(nd, 0);
        std::size_t n = 1;
        for (std::size_t d = nd; d-- > 0;)
        {
            if (!reduced[d])
            {
                strides[d] = n;
                n *= static_cast<std::size_t>(shape_[d]);
            }
        }
        for (std::size_t d = 0; d < nd; ++d)
        {
            if (!reduced[d])
                shape.push_back(shape_[d]);
            else if (keepdims)
                shape.push_back(1);
        }
        std::vector<T> r(n, T(init));
        std::vector<py::ssize_t> idx(nd, 0);
        std::size_t io = 0;
        for (const T& x : data_)
        {
            r[io] = op(r[io], x);
            // next index, last dimension fastest
            for (std::size_t d = nd; d-- > 0;)
            {
                io += strides[d];
                if (++idx[d] < shape_[d])
                    break;
                io -= strides[d] * static_cast<std::size_t>(shape_[d]);
                idx[d] = 0;
            }
        }
        return Array(std::move(shape), std::move(r));
    }

  private:
    Array(std::vector<py::ssize_t> shape, std::vector<T> data)
        : shape_(std::move(shape)), data_(std::move(data))
    {
    }

    // the extent of dimension d with the shape right-aligned to nd dimensions
    py::ssize_t extent(std::size_t nd, std::size_t d) const
    {
        const std::size_t offset = nd - shape_.size();
        return d < offset ? 1 : shape_[d - offset];
    }

    // C-order element strides right-aligned to nd dimensions, 0 where the extent is 1
    std::vector<py::ssize_t> broadcast_strides(std::size_t nd) const
    {
        std::vector<py::ssize_t> strides(nd, 0);
        py::ssize_t stride = 1;
        for (std::size_t d = nd; d-- > 0;)
        {
            const py::ssize_t e = extent(nd, d);
            if (e != 1)
                strides[d] = stride;
            stride *= e;
        }
        return strides;
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

// the elementwise XAD math function for a unary NumPy ufunc name, nullptr if there is none
template <class T>
T (*array_unary_function(const std::string& name))(const T&)
{
    using F = T (*)(const T&);
    static const std::map<std::string, F> functions = {
        {"exp", [](const T& x) { return T(xad::exp(x)); }},
        {"log", [](const T& x) { return T(xad::log(x)); }},
        {"log10", [](const T& x) { return T(xad::log10(x)); }},
        {"log2", [](const T& x) { return T(xad::log2(x)); }},
        {"expm1", [](const T& x) { return T(xad::expm1(x)); }},
        {"log1p", [](const T& x) { return T(xad::log1p(x)); }},
        {"sqrt", [](const T& x) { return T(xad::sqrt(x)); }},
        {"cbrt", [](const T& x) { return T(xad::cbrt(x)); }},
        {"sin", [](const T& x) { return T(xad::sin(x)); }},
        {"cos", [](const T& x) { return T(xad::cos(x)); }},
        {"tan", [](const T& x) { return T(xad::tan(x)); }},
        {"arcsin", [](const T& x) { return T(xad::asin(x)); }},
        {"arccos", [](const T& x) { return T(xad::acos(x)); }},
        {"arctan", [](const T& x) { return T(xad::atan(x)); }},
        {"sinh", [](const T& x) { return T(xad::sinh(x)); }},
        {"cosh", [](const T& x) { return T(xad::cosh(x)); }},
        {"tanh", [](const T& x) { return T(xad::tanh(x)); }},
        {"arcsinh", [](const T& x) { return T(xad::asinh(x)); }},
        {"arctanh", [](const T& x) { return T(xad::atanh(x)); }},
    };
    auto it = functions.find(name);
    return it == functions.end() ? nullptr : it->second;
}

// the elementwise XAD operation for a binary NumPy ufunc name, nullptr if there is none
template <class T>
T (*array_binary_function(const std::string& name))(const T&, const T&)
{
    using F = T (*)(const T&, const T&);
    static const std::map<std::string, F> functions = {
        {"add", [](const T& x, const T& y) { return T(x + y); }},
        {"subtract", [](const T& x, const T& y) { return T(x - y); }},
        {"multiply", [](const T& x, const T& y) { return T(x * y); }},
        {"divide", [](const T& x, const T& y) { return T(x / y); }},
        {"power", [](const T& x, const T& y) { return T(xad::pow(x, y)); }},
    };
    auto it = functions.find(name);
    return it == functions.end() ? nullptr : it->second;
}

// a ufunc operand as an Array: itself (no copy), a 0-d one holding a Real (active, never through
// __float__), or one from a float array or float
template <class T>
py::object array_operand(const py::handle& o)
{
    if (py::isinstance<Array<T>>(o))
        return py::reinterpret_borrow<py::object>(o);
    if (py::isinstance<T>(o))
        return py::cast(Array<T>(o.cast<const T&>()));
    return py::cast(Array<T>(py::cast<typename Array<T>::float_array>(o)));
}

// NumPy's __array_ufunc__: plain calls of the unary and binary ufuncs with an XAD counterpart,
// NotImplemented otherwise (NumPy then raises TypeError)
template <class T>
py::object array_ufunc(const py::object& /*self*/, const py::object& ufunc,
                       const std::string& method, const py::args& inputs, const py::kwargs& kwargs)
{
    // plain calls only: no reduce/accumulate/at/outer, no out=, where=, dtype=, ...
    if (method != "__call__" || kwargs.size() != 0)
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    const std::string name = py::str(ufunc.attr("__name__"));
    if (inputs.size() == 1)
    {
        if (auto f = array_unary_function<T>(name))
        {
            const auto a = array_operand<T>(inputs[0]);
            return py::cast(a.template cast<const Array<T>&>().map(f));
        }
    }
    else if (inputs.size() == 2)
    {
        if (auto f = array_binary_function<T>(name))
        {
            const auto a = array_operand<T>(inputs[0]), b = array_operand<T>(inputs[1]);
            return py::cast(
                a.template cast<const Array<T>&>().map(b.template cast<const Array<T>&>(), f));
        }
    }
    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
}

// raises numpy.exceptions.AxisError for axis out of bounds in ndim dimensions
[[noreturn]] inline void throw_axis_error(py::ssize_t axis, std::size_t ndim)
{
    py::object e = py::module_::import("numpy.exceptions").attr("AxisError")(axis, ndim);
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(e.ptr())), e.ptr());
    throw py::error_already_set();
}

// the dimensions of ndim reduced by a NumPy axis argument: None (all), an int or a sequence
inline std::vector<bool> reduced_dimensions(const py::object& axis, std::size_t ndim)
{
    if (axis.is_none())
        return std::vector<bool>(ndim, true);
    std::vector<bool> reduced(ndim, false);
    const auto nd = static_cast<py::ssize_t>(ndim);
    const py::list axes =
        py::isinstance<py::int_>(axis) ? py::list(py::make_tuple(axis)) : py::list(axis);
    for (const auto& ax : axes)
    {
        auto d = ax.cast<py::ssize_t>();
        if (d < -nd || d >= nd)
            throw_axis_error(d, ndim);
        reduced[static_cast<std::size_t>(d < 0 ? d + nd : d)] = true;
    }
    return reduced;
}

// NumPy's __array_function__: sum, prod and mean (with axis, keepdims) on an Array,
// NotImplemented otherwise (NumPy then raises TypeError)
template <class T>
py::object array_function(const py::object& /*self*/, const py::object& func,
                          const py::object& /*types*/, const py::tuple& args,
                          const py::dict& kwargs)
{
    const auto unsupported = py::reinterpret_borrow<py::object>(Py_NotImplemented);
    const std::string name = py::str(func.attr("__name__"));
    if ((name != "sum" && name != "prod" && name != "mean") || args.size() < 1 || args.size() > 2 ||
        !py::isinstance<Array<T>>(args[0]))
        return unsupported;
    for (const auto& kv : kwargs)
    {
        const std::string key = py::str(kv.first);
        if (key != "axis" && key != "keepdims")
            return unsupported;
    }
    const auto& a = args[0].cast<const Array<T>&>();
    const auto reduced = reduced_dimensions(
        args.size() > 1 ? py::object(args[1]) : kwargs.attr("get")("axis"), a.ndim());
    const bool keepdims = py::bool_(kwargs.attr("get")("keepdims", false));
    py::ssize_t count = 1;
    for (std::size_t d = 0; d < reduced.size(); ++d)
        if (reduced[d])
            count *= a.dim(d);

    Array<T> r =
        name == "prod"
            ? a.reduce(reduced, keepdims, 1.0, [](const T& p, const T& x) { return T(p * x); })
            : a.reduce(reduced, keepdims, 0.0, [](const T& s, const T& x) { return T(s + x); });
    if (name == "mean")
    {
        if (count == 0 && PyErr_WarnEx(PyExc_RuntimeWarning, "Mean of empty slice.", 1) < 0)
            throw py::error_already_set();
        const auto n = static_cast<double>(count);
        r = r.map([n](const T& x) { return T(x / n); });
    }
    if (r.ndim() == 0)
        return py::cast(T(r.item(0)));
    return py::cast(std::move(r));
}

template <class T>
void py_array(py::module_& m)
{
    auto c =
        py::class_<Array<T>>(m, "Array", "n-dimensional array of active values")
            .def(py::init<const typename Array<T>::float_array&>(), py::arg("values"),
                 "an array from a float array or (nested) sequence of floats")
            .def_property_readonly("shape", &Array<T>::shape, "the shape of the array")
            .def_property_readonly("values", &Array<T>::values,
                                   "the values of the elements as a float array")
            .def_property_readonly("derivatives", &Array<T>::derivatives,
                                   "the derivatives of the elements as a float array")
            .def("__getitem__", py::overload_cast<py::ssize_t>(&Array<T>::item), py::arg("index"),
                 py::return_value_policy::reference_internal, "the element itself, not a copy")
            .def("__getitem__", py::overload_cast<const py::tuple&>(&Array<T>::item),
                 py::arg("index"), py::return_value_policy::reference_internal,
                 "the element at a full index itself, not a copy")
            .def("sum", &Array<T>::sum, "the sum of all elements")
            .def("__array_function__", &array_function<T>, py::arg("func"), py::arg("types"),
                 py::arg("args"), py::arg("kwargs"), "NumPy function dispatch to XAD loops")
            .def("__array_ufunc__", &array_ufunc<T>, py::arg("ufunc"), py::arg("method"),
                 "NumPy ufunc dispatch to XAD's elementwise math");
    py_array_binary_op(c, "__add__", "__radd__",
                       [](const auto& x, const auto& y) { return T(x + y); });
    py_array_binary_op(c, "__sub__", "__rsub__",
                       [](const auto& x, const auto& y) { return T(x - y); });
    py_array_binary_op(c, "__mul__", "__rmul__",
                       [](const auto& x, const auto& y) { return T(x * y); });
    py_array_binary_op(c, "__truediv__", "__rtruediv__",
                       [](const auto& x, const auto& y) { return T(x / y); });
}

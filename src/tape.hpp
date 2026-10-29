/*******************************************************************************

   Defines the bindings for the XAD tape.

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
#include <pybind11/stl.h>
#include "double_caster.hpp"
#include "register_input.hpp"
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

// lets Python subclasses override computeAdjoint
template <class Tape>
class PyCheckpointCallback : public xad::CheckpointCallback<Tape>
{
  public:
    void computeAdjoint(Tape *tape) override
    {
        PYBIND11_OVERRIDE_PURE(void, xad::CheckpointCallback<Tape>, computeAdjoint, tape);
    }
};

// Python references to the callbacks handed to a tape with pushCallback or insertCallback,
// keyed by tape. XAD deletes the pointers it owns, but Python callbacks are owned by their
// Python objects, so they are kept here instead of in XAD's list. Inserted callbacks are
// released once XAD drops them from its checkpoints (newRecording, clearAll, resetTo, end of a
// nested recording). Heap-allocated and never freed so that no Python object is destroyed
// after interpreter shutdown.
template <class Tape>
class PyOwnedCallbacks
{
  public:
    using Callback = xad::CheckpointCallback<Tape>;

    static void push(py::handle tape, py::object cb) { entry(tape).pushed.append(cb); }

    static void insert(py::handle tape, py::object cb)
    {
        entry(tape).inserted.emplace_back(cb.cast<Callback *>(), cb);
    }

    static void clear(Tape *key)
    {
        auto it = store().find(key);
        if (it != store().end())
            it->second.pushed = py::list();
    }

    // drop the inserted callbacks that are no longer in the tape's checkpoints
    static void release(Tape &tape)
    {
        auto it = store().find(&tape);
        if (it == store().end() || it->second.inserted.empty())
            return;
        std::unordered_set<const Callback *> live;
        for (const auto &c : tape.getCheckpoints())
            live.insert(c.second);
        std::vector<std::pair<Callback *, py::object>> kept;
        for (auto &e : it->second.inserted)
            if (live.count(e.first))
                kept.push_back(std::move(e));
        // the dropped references go away with `kept`'s old contents, after the swap
        kept.swap(it->second.inserted);
    }

  private:
    struct Owned
    {
        py::list pushed;
        std::vector<std::pair<Callback *, py::object>> inserted;
    };

    static Owned &entry(py::handle tape)
    {
        auto *key = tape.cast<Tape *>();
        auto &entries = store();
        auto it = entries.find(key);
        if (it == entries.end())
        {
            // drop the references when the Python tape object goes away (as pybind11's keep_alive)
            py::cpp_function cleanup(
                [key](py::handle weakref)
                {
                    store().erase(key);
                    weakref.dec_ref();
                });
            py::weakref(tape, cleanup).release();
            it = entries.emplace(key, Owned()).first;
        }
        return it->second;
    }

    static std::unordered_map<Tape *, Owned> &store()
    {
        static auto *entries = new std::unordered_map<Tape *, Owned>();
        return *entries;
    }
};

template <class Tape>
void py_checkpoint_callback(py::module_ &m)
{
    using Callback = xad::CheckpointCallback<Tape>;
    py::class_<Callback, PyCheckpointCallback<Tape>>(m, "CheckpointCallback")
        .def(py::init<>())
        .def("computeAdjoint", &Callback::computeAdjoint,
             "called during adjoint computation when the tape reaches this callback");
}

// owns the nested recording so that it ends at __exit__, not at garbage collection
template <class Tape>
class PyScopedNestedRecording
{
  public:
    explicit PyScopedNestedRecording(Tape &tape)
        : nested_(std::make_unique<xad::ScopedNestedRecording<Tape>>(&tape))
    {
    }
    void end()
    {
        if (!nested_)
            return;
        Tape *tape = nested_->getTape();
        nested_.reset();
        PyOwnedCallbacks<Tape>::release(*tape);
    }
    xad::ScopedNestedRecording<Tape> &get()
    {
        if (!nested_)
            throw std::runtime_error("nested recording has already ended");
        return *nested_;
    }

  private:
    std::unique_ptr<xad::ScopedNestedRecording<Tape>> nested_;
};

template <class Tape>
void py_scoped_nested_recording(py::module_ &m)
{
    using Nested = PyScopedNestedRecording<Tape>;
    py::class_<Nested>(m, "ScopedNestedRecording")
        .def(py::init<Tape &>(), py::keep_alive<1, 2>(),
             "starts a nested recording on the given tape (for use in checkpoint callbacks)")
        .def("__enter__", [](Nested &self) -> Nested & { return self; },
             py::return_value_policy::reference_internal)
        .def(
            "__exit__",
            [](Nested &self, const std::optional<pybind11::type> &,
               const std::optional<pybind11::object> &, const std::optional<pybind11::object> &)
            { self.end(); },
            "ends the nested recording when exiting the context")
        .def(
            "computeAdjoints", [](Nested &self) { self.get().computeAdjoints(); },
            "propagate adjoints through the nested recording")
        .def(
            "incrementAdjoint",
            [](Nested &self, typename Tape::slot_type slot, const typename Tape::value_type &value)
            { self.get().incrementAdjoint(slot, value); },
            "add the given value to the adjoint at the given slot of the outer recording")
        .def(
            "getTape", [](Nested &self) -> Tape & { return *self.get().getTape(); },
            py::return_value_policy::reference, "get the tape of the nested recording");
}

template <class Tape>
void py_tape(py::module_ &m)
{
    using AReal = typename Tape::active_type;
    using slot_type = typename Tape::slot_type;

    py::class_<Tape>(m, "Tape")
        .def(py::init([] { return std::make_unique<Tape>(false); }),
             "constructs a tape without activating it")
        .def(
            "__enter__",
            [](Tape &self) -> Tape &
            {
                self.activate();
                return self;
            },
            "enters a context `with tape`, activating the tape")
        .def(
            "__exit__",
            [](Tape &self, const std::optional<pybind11::type> &,
               const std::optional<pybind11::object> &, const std::optional<pybind11::object> &)
            { self.deactivate(); },
            "deactivates the tape when exiting the context")
        .def("activate", &Tape::activate, "activate the tape")
        .def("deactivate", &Tape::deactivate, "deactivate the tape")
        .def("isActive", &Tape::isActive, "check if the tape is active")
        .def("getActive", &Tape::getActive,
             "class method to get a reference to the currently active tape")
        .def("getPosition", &Tape::getPosition,
             "get the current position on the tape. Used in conjunction with `computeAdjointsTo`.")
        .def(
            "registerInput",
            [](Tape &self, AReal &inp) { register_fresh_input(self, inp); },
            "registers an input variable with tape, for recording")
        .def("registerOutput", py::overload_cast<AReal &>(&Tape::registerOutput),
             "registers an output with the tape (to be called before setting output adjoints)")
        .def("computeAdjoints", &Tape::computeAdjoints,
             "Roll back the tape until the point of calling `newRecording`, propagating adjoints "
             "from outputs to inputs")
        .def("computeAdjointsTo", &Tape::computeAdjointsTo,
             "Roll back the tape until the given position (see `getPosition`), propagating "
             "adjoints from outputs backwards.")
        .def(
            "newRecording",
            [](Tape &self)
            {
                self.newRecording();
                PyOwnedCallbacks<Tape>::release(self);
            },
            "Start a new recording on tape, marking the start of a function to be derived")
        .def(
            "clearAll",
            [](Tape &self)
            {
                self.clearAll();
                PyOwnedCallbacks<Tape>::release(self);
            },
            "clear/reset the tape completely, without de-allocating memory. Should be used for "
             "re-using the tape, rather than creating a new one")
        .def("getMemory", &Tape::getMemory, "Get the total memory consumed by the tape in bytes")
        .def("clearDerivatives", &Tape::clearDerivatives,
             "clear all derivatives stored on the tape")
        .def("clearDerivativesAfter", &Tape::clearDerivativesAfter,
             "clear all derivatives after the given position")
        .def(
            "resetTo",
            [](Tape &self, typename Tape::position_type pos)
            {
                self.resetTo(pos);
                PyOwnedCallbacks<Tape>::release(self);
            },
            "reset the tape back to the given position")
        .def(
            "insertCallback",
            [](py::handle self, py::object cb)
            {
                if (!py::isinstance<xad::CheckpointCallback<Tape>>(cb))
                    throw py::type_error("insertCallback expects a CheckpointCallback");
                self.cast<Tape &>().insertCallback(cb.cast<xad::CheckpointCallback<Tape> *>());
                PyOwnedCallbacks<Tape>::insert(self, cb);
            },
            "insert a checkpoint callback at the current tape position (kept alive until the "
            "tape drops it)")
        .def(
            "pushCallback",
            [](py::handle self, py::object cb)
            {
                if (!py::isinstance<xad::CheckpointCallback<Tape>>(cb))
                    throw py::type_error("pushCallback expects a CheckpointCallback");
                PyOwnedCallbacks<Tape>::push(self, cb);
            },
            py::arg("callback"),
            "hand a checkpoint callback to the tape to keep alive until `clearCallbacks` (it is "
            "not inserted or run)")
        .def(
            "clearCallbacks",
            [](Tape &self)
            {
                PyOwnedCallbacks<Tape>::clear(&self);
                self.clearCallbacks();
            },
            "release the callbacks handed to the tape with `pushCallback`")
        .def("getAndResetOutputAdjoint", &Tape::getAndResetOutputAdjoint,
             "get the adjoint at the given slot and reset it to zero (for use in callbacks)")
        .def("incrementAdjoint",
             static_cast<void (Tape::*)(slot_type, const typename Tape::value_type &)>(
                 &Tape::incrementAdjoint),
             "add the given value to the adjoint at the given slot (for use in callbacks)")
        .def("printStatus", &Tape::printStatus,
             "output the status of the tape (for debugging/information)")
        .def(
            "derivative", [](Tape &self, AReal &d) { return self.derivative(d.getSlot()); },
            "get the slot of the given variable")
        .def(
            "derivative", [](Tape &self, slot_type slot) { return self.derivative(slot); },
            "get the derivative stored at the given slot position")
        .def(
            "getDerivative", [](Tape &self, AReal &d) { return self.derivative(d.getSlot()); },
            "alias for `derivative`")
        .def(
            "getDerivative", [](Tape &self, slot_type slot) { return self.derivative(slot); },
            "alias for `derivative`")
        .def(
            "setDerivative",
            [](Tape &self, AReal &d, double &b) { return self.setDerivative(d.getSlot(), b); },
            "sets the derivative of the given active variable to the value given")
        .def(
            "setDerivative",
            [](Tape &self, slot_type slot, double &b) { return self.setDerivative(slot, b); },
            "sets the derivative at the given slot to the given value");
}

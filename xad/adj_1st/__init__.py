##############################################################################
#
#  First order adjoint mode module for the XAD Python bindings
#
#  This file is part of XAD's Python bindings, a comprehensive library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2026 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##############################################################################

from xad._real import add_float_methods
from xad._xad.adj_1st import (
    Array,
    CheckpointCallback,
    Complex,
    Real,
    ScopedNestedRecording,
    Tape,
)

__all__ = ["Array", "CheckpointCallback", "Complex", "Real", "ScopedNestedRecording", "Tape"]


def _register_inputs(self, inputs):
    if isinstance(inputs, Array):
        inputs._register_inputs(self)
        return
    for i in inputs:
        self.registerInput(i)


setattr(Tape, "registerInputs", _register_inputs)


def _register_outputs(self, outputs):
    if isinstance(outputs, Array):
        outputs._register_outputs(self)
        return
    for o in outputs:
        self.registerOutput(o)


setattr(Tape, "registerOutputs", _register_outputs)

setattr(
    Real,
    "value",
    property(Real.getValue, Real.setValue, doc="get/set the underlying float value of the object"),
)
setattr(
    Real,
    "derivative",
    property(
        Real.getDerivative, Real.setDerivative, doc="get/set the derivative (adjoint) of the object"
    ),
)

add_float_methods(Real)

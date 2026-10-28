##############################################################################
#
#  Sample for external functions with Python
#
#    Computes the length of a vector as
#      y = sqrt(sum(x^2))
#    where the square operation is element-wise.
#
#    The derivatives w.r.t. all inputs are computed using adjoint mode AD,
#    where the summing of the elements is implemented as an external function
#    and a manual adjoint is implemented for the reverse path.
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

import math

import xad.adj_1st as xadj
from xad.math import sqrt


def sum_elements_passive(x):
    """The external function, assumed to be defined in an optimized
    external library, working on floats only."""
    return sum(x)


class ExternalSumElementsCallback(xadj.CheckpointCallback):
    """A callback object that is inserted into the tape and called by
    computeAdjoints to update the adjoints of the external function.

    Its only callback method is computeAdjoint - compute_external is added
    for convenience of implementation.
    """

    def compute_external(self, x):
        """Computes the external function passively during the forward run
        and stores the result in an active variable."""
        tape = xadj.Tape.getActive()

        # store the slots of the input variables
        self.input_slots = [xi.getSlot() for xi in x]

        # run the external function on the passive values
        y = sum_elements_passive([xi.value for xi in x])

        # set the active output and store its slot
        ret = xadj.Real(y)
        tape.registerOutput(ret)
        self.output_slot = ret.getSlot()

        # register callback for the reverse path (the tape keeps it alive)
        tape.insertCallback(self)
        return ret

    def computeAdjoint(self, tape):
        """Updates the adjoints for the external function, using manual AD.
        The derivative of a sum w.r.t. each input is 1, so the input adjoints
        are incremented by the output adjoint itself."""
        output_adj = tape.getAndResetOutputAdjoint(self.output_slot)
        for slot in self.input_slots:
            tape.incrementAdjoint(slot, output_adj)


def sum_elements(x):
    """Wraps the external function for active types."""
    return ExternalSumElementsCallback().compute_external(x)


# setup input variables
n = 5
x = [i + 1 + math.sin(i) for i in range(n)]

with xadj.Tape() as tape:
    # setup independent variables
    x_ad = [xadj.Real(xi) for xi in x]
    for xi in x_ad:
        tape.registerInput(xi)

    # start recording derivatives
    tape.newRecording()

    # compute sqrt(sum(x^2)) with sum_elements as an external function
    xsqr = [xi * xi for xi in x_ad]
    y = sqrt(sum_elements(xsqr))

    # set output adjoint and compute the other adjoints
    tape.registerOutput(y)
    y.derivative = 1.0
    tape.computeAdjoints()

    # output results
    print(f"y = {y.value!r}")
    print("\nfirst order derivatives:")
    for i, xi in enumerate(x_ad):
        print(f"dy/dx{i} = {xi.derivative!r}")

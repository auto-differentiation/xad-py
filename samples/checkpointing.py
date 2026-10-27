##############################################################################
#
#  Sample for checkpointing with Python
#
#    Computes
#      xout = sin(sin(sin(sin(... sin(xin)))))
#      (sin applied n times)
#    and its first order derivative
#      dxout/dxin
#    using checkpointing.
#
#    A checkpoint is created after equi-distant applications of the sin
#    function to save memory.
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

import xad.adj_1st as xadj
from xad.math import sin


def repeated_sin(n, x):
    """Applies the sin function n times, for floats and active types."""
    for _ in range(n):
        x = sin(x)
    return x


class SinCheckpointCallback(xadj.CheckpointCallback):
    """A callback object that is inserted into the tape and called by
    computeAdjoints at the points where it was inserted during the forward run.

    Its only callback method is computeAdjoint - compute_stage is added
    for convenience of implementation.
    """

    def __init__(self):
        super().__init__()
        # storage for checkpoints
        # (all checkpoints in the same object, hence we use a stack)
        self.n = []  # number of iterations for each stage
        self.x = []  # the input value for each stage
        self.slots = []  # slots for the input, then output

    def compute_stage(self, n, x):
        """Computes one stage of the algorithm passively, stores a checkpoint,
        and puts the result back into the active variable x.
        """
        # store data needed for reverse run
        self.n.append(n)  # store how many iterations
        self.slots.append(x.getSlot())  # store the slot of the input variable
        x_p = x.value
        self.x.append(x_p)  # store passive value of the input

        # run the algorithm passively
        x_p = repeated_sin(n, x_p)

        # update the active output value (same variable, same slot)
        x.value = x_p
        self.slots.append(x.getSlot())  # store the slot of the output variable

        # register callback for the reverse path (the tape keeps it alive)
        xadj.Tape.getActive().insertCallback(self)

    def computeAdjoint(self, tape):
        """Callback for the reverse path, called by computeAdjoints to update
        the adjoints with the checkpointed function, computed with XAD in a
        nested recording in this case.
        """
        # restore checkpoint variables
        output_slot = self.slots.pop()
        input_slot = self.slots.pop()
        n = self.n.pop()

        # retrieve adjoint of output
        # Important: retrieve this before any active operations, as that may
        #            overwrite the value
        output_adj = tape.getAndResetOutputAdjoint(output_slot)

        # create new active variable and start nested recording from after that
        x_in = xadj.Real(self.x.pop())
        tape.registerInput(x_in)

        # start a nested recording to compute adjoints for just this part
        with xadj.ScopedNestedRecording(tape) as nested:
            # run actively
            x = repeated_sin(n, x_in)

            tape.registerOutput(x)
            x.derivative = output_adj  # set output adjoint
            nested.computeAdjoints()  # calculate adjoints in nested recording

            # increment input derivative
            nested.incrementAdjoint(input_slot, x_in.derivative)


def repeated_sin_checkpointed(n, x, x_adj):
    """Driver function that returns the value of repeated_sin(n, x) and its
    adjoint, with x_adj the initial adjoint seed of the output."""
    with xadj.Tape() as tape:
        x_ad = xadj.Real(x)  # initialize independent variable
        tape.registerInput(x_ad)
        tape.newRecording()  # start recording derivatives

        # setup checkpointing object
        chkpt = SinCheckpointCallback()

        checkpt_distance = 4  # distance between checkpoints
        for i in range(0, n, checkpt_distance):
            m = min(checkpt_distance, n - i)
            chkpt.compute_stage(m, x_ad)  # run one stage forward + checkpoint

        tape.registerOutput(x_ad)
        x_ad.derivative = x_adj  # set final output derivative
        tape.computeAdjoints()  # compute adjoints of independents

        return x_ad.value, x_ad.derivative


n = 20  # number of iterations
x = 2.1  # input value
xa = 1.0  # initial adjoint of output

# run checkpointed version
x, xa = repeated_sin_checkpointed(n, x, xa)

# output results
print(f"xout       = {x!r}")
print(f"dxout/dxin = {xa!r}")

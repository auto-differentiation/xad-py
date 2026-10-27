##############################################################################
#
#  Sample for second-order forward-over-adjoint calculation with Python
#
#    Computes
#      y = f(x0, x1, x2, x3)
#    its first order derivatives using adjoints
#      dy/dx0, dy/dx1, dy/dx2, dy/dx3
#    and the second order derivatives w.r.t. x0 using forward mode
#      d2y/dx0dx0, d2y/dx0dx1, d2y/dx0dx2, d2y/dx0dx3
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

import xad
import xad.fwd_adj_2nd as xfa
from xad.math import sin, cos, tan


def f(x0, x1, x2, x3):
    a = sin(x0) * cos(x1)
    b = x2 * x3 - tan(x1 - x2)
    c = a + 2 * b
    return c * c


# input values
x0 = 1.0
x1 = 1.5
x2 = 1.3
x3 = 1.2

# set independent variables
x0_ad = xfa.Real(x0)
x1_ad = xfa.Real(x1)
x2_ad = xfa.Real(x2)
x3_ad = xfa.Real(x3)

with xfa.Tape() as tape:
    # and register them
    tape.registerInput(x0_ad)
    tape.registerInput(x1_ad)
    tape.registerInput(x2_ad)
    tape.registerInput(x3_ad)

    # set the variable of which we want the second order derivatives
    xad.value(x0_ad).derivative = 1.0

    # start recording derivatives
    tape.newRecording()

    y = f(x0_ad, x1_ad, x2_ad, x3_ad)

    # seed 1st order adjoint of output
    tape.registerOutput(y)
    xad.derivative(y).value = 1.0

    # compute all other adjoints and the second order tangent linear
    tape.computeAdjoints()

    # output results
    print(f"y      = {xad.value(xad.value(y))}")
    print("\nfirst order derivatives:")
    print(f"dy/dx0 = {xad.value(xad.derivative(x0_ad))}")
    print(f"dy/dx1 = {xad.value(xad.derivative(x1_ad))}")
    print(f"dy/dx2 = {xad.value(xad.derivative(x2_ad))}")
    print(f"dy/dx3 = {xad.value(xad.derivative(x3_ad))}")
    print("\nsecond order derivatives w.r.t. x0:")
    print(f"d2y/dx0dx0 = {xad.derivative(xad.derivative(x0_ad))}")
    print(f"d2y/dx0dx1 = {xad.derivative(xad.derivative(x1_ad))}")
    print(f"d2y/dx0dx2 = {xad.derivative(xad.derivative(x2_ad))}")
    print(f"d2y/dx0dx3 = {xad.derivative(xad.derivative(x3_ad))}")

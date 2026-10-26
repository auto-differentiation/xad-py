##############################################################################
#
#  Sample for complex numbers in adjoint mode
#
#    Computes
#      w = z * exp(z),  z = x + i y
#    and the first order derivatives of Re(w) and |w|
#      d/dx, d/dy
#    using adjoint mode.
#
#  This file is part of XAD's Python bindings, a comprehensive library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
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

import xad.adj_1st as xadj
import xad.cmath


# set independent variables: real and imaginary part
x = xadj.Real(0.5)
y = xadj.Real(1.2)

with xadj.Tape() as tape:
    tape.registerInput(x)
    tape.registerInput(y)

    # start recording derivatives
    tape.newRecording()

    # calculate the output
    z = xadj.Complex(x, y)
    w = z * xad.cmath.exp(z)
    print(f"w = {w}")

    # derivatives of the real part
    tape.registerOutput(w.real)
    w.real.derivative = 1.0
    tape.computeAdjoints()
    print(f"dRe(w)/dx = {x.derivative}")
    print(f"dRe(w)/dy = {y.derivative}")

    # derivatives of the modulus
    r = abs(w)
    tape.registerOutput(r)
    tape.clearDerivatives()
    r.derivative = 1.0
    tape.computeAdjoints()
    print(f"d|w|/dx = {x.derivative}")
    print(f"d|w|/dy = {y.derivative}")

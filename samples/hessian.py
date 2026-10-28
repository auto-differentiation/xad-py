##############################################################################
#
#  Sample for computing a Hessian with Python
#
#    Computes the Hessian of
#      y = sin(x0 x1) - cos(x1 x2) - sin(x2 x3) - cos(x3 x0)
#    using forward over adjoint mode.
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
from xad.math import sin, cos


def f(x):
    return sin(x[0] * x[1]) - cos(x[1] * x[2]) - sin(x[2] * x[3]) - cos(x[3] * x[0])


hessian = xad.hessian(f, [1.0, 1.5, 1.3, 1.2])

# output results
for row in hessian:
    print(" ".join(f"{elem:.6f}" for elem in row))

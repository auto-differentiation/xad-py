##############################################################################
#
#  Sample for NumPy arrays with Python
#
#    Computes
#      y = sum(w * exp(x))
#    for a vector x with whole-array operations on an xad Array, where w are
#    constant weights.
#
#    The derivatives w.r.t. all inputs are computed using adjoint mode AD.
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

import numpy as np

import xad.adj_1st as xadj

# setup input values and constant weights
n = 4
x_values = 0.1 * np.arange(1, n + 1)
w = np.arange(1.0, n + 1)

with xadj.Tape() as tape:
    # setup independent variables: an array of active values
    x = xadj.Array(x_values)
    tape.registerInputs(x)

    # start recording derivatives
    tape.newRecording()

    # compute sum(w * exp(x)) with whole-array operations (one C++ loop each)
    y = np.sum(w * np.exp(x))

    # set output adjoint and compute the other adjoints
    tape.registerOutput(y)
    y.derivative = 1.0
    tape.computeAdjoints()

    # output results
    print(f"y = {y.value!r}")
    print("\nfirst order derivatives:")
    for i, d in enumerate(x.derivatives):
        print(f"dy/dx{i} = {float(d)!r}")

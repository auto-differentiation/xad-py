##############################################################################
#
#  Complex math module for the XAD Python bindings
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

"""XAD cmath module - mimics the standard cmath module, but allows using XAD active
complex types (the Complex of every mode, e.g. xad.adj_1st.Complex) as arguments. Called with
passive arguments only (complex, float, int), the functions return exactly what the
standard cmath module returns.
"""

import cmath as _cmath
from xad._xad import (
    adj_1st as _adj,
    fwd_1st as _fwd,
    fwd_adj_2nd as _fwd_adj,
    fwd_fwd_2nd as _fwd_fwd,
    cmath as _active,
)

__all__ = [
    "exp",
    "log",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "phase",
    "polar",
    "rect",
]

# the Complex type of each active type's mode
_MODE_COMPLEX = {
    _adj.Complex: _adj.Complex,
    _adj.Real: _adj.Complex,
    _fwd.Complex: _fwd.Complex,
    _fwd.Real: _fwd.Complex,
    _fwd_adj.Complex: _fwd_adj.Complex,
    _fwd_adj.Real: _fwd_adj.Complex,
    _fwd_fwd.Complex: _fwd_fwd.Complex,
    _fwd_fwd.Real: _fwd_fwd.Complex,
}


def _promote(arg, mode_complex):
    """Real or Python complex to the mode's Complex; Complex and float stay as they are"""
    if isinstance(arg, (complex, _adj.Real, _fwd.Real, _fwd_adj.Real, _fwd_fwd.Real)):
        return mode_complex(arg)
    return arg


def _dispatch(name: str):
    """XAD's function if any argument is active, otherwise the standard cmath one"""
    active, passive = getattr(_active, name), getattr(_cmath, name)

    def f(*args):
        modes = (_MODE_COMPLEX[type(a)] for a in args if type(a) in _MODE_COMPLEX)
        mode_complex = next(modes, None)
        if mode_complex is None:
            return passive(*args)
        # rect takes real polar coordinates, all others complex arguments
        if name != "rect":
            args = tuple(_promote(a, mode_complex) for a in args)
        return active(*args)

    f.__name__ = f.__qualname__ = name
    f.__doc__ = active.__doc__
    return f


exp = _dispatch("exp")
log = _dispatch("log")
log10 = _dispatch("log10")
sqrt = _dispatch("sqrt")
sin = _dispatch("sin")
cos = _dispatch("cos")
tan = _dispatch("tan")
asin = _dispatch("asin")
acos = _dispatch("acos")
atan = _dispatch("atan")
sinh = _dispatch("sinh")
cosh = _dispatch("cosh")
tanh = _dispatch("tanh")
asinh = _dispatch("asinh")
acosh = _dispatch("acosh")
atanh = _dispatch("atanh")
phase = _dispatch("phase")
polar = _dispatch("polar")
rect = _dispatch("rect")

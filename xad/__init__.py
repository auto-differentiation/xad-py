##############################################################################
#
#  XAD Python bindings
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

"""Python bindings for the XAD comprehensive library for automatic differentiation"""

import os
import sys
from typing import Any, Union

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    # xad.dll, for extensions linking it from elsewhere
    _dll_dir = os.add_dll_directory(_PKG_DIR)

from ._xad import __version__, hessian, jacobian  # noqa: E402

# the mode packages, which add Python-side methods and properties to the bound types
from . import adj_1st, fwd_1st, fwd_adj_2nd, fwd_fwd_2nd  # noqa: E402

__all__ = [
    "value",
    "derivative",
    "jacobian",
    "hessian",
    "get_include",
    "get_cmake_dir",
    "get_library_dir",
    "__version__",
]


def get_include() -> str:
    """Directory with the XAD C++ headers (``XAD/XAD.hpp``)"""
    return os.path.join(_PKG_DIR, "include")


def get_cmake_dir() -> str:
    """Directory with ``XADConfig.cmake``, for ``find_package(XAD CONFIG)`` via ``XAD_DIR``"""
    return os.path.join(_PKG_DIR, "share", "cmake", "XAD")


def get_library_dir() -> str:
    """Directory with the XAD library (import library on Windows, ``xad.dll`` is in the package)"""
    return os.path.join(_PKG_DIR, "lib")


# the types xad.value / xad.derivative accept
_Active = Union[
    adj_1st.Real,
    fwd_1st.Real,
    fwd_adj_2nd.Real,
    fwd_adj_2nd.DerivativeRef,
    fwd_fwd_2nd.Real,
    fwd_fwd_2nd.DerivativeRef,
]
_ACTIVE = (
    adj_1st.Real,
    fwd_1st.Real,
    fwd_adj_2nd.Real,
    fwd_adj_2nd.DerivativeRef,
    fwd_fwd_2nd.Real,
    fwd_fwd_2nd.DerivativeRef,
)


def value(x: Union[_Active, Any]) -> Union[float, fwd_1st.Real]:
    """Get the value of an XAD active type - or return the value itself otherwise

    Args:
        x (Real | any): Argument to get the value of

    Returns:
        float | Real: The value stored in the variable; for second order types the
        inner first order Real, by reference (as in C++)
    """
    if isinstance(x, _ACTIVE):
        return x.getValue()
    else:
        return x


def derivative(
    x: _Active,
) -> Union[float, fwd_adj_2nd.DerivativeRef, fwd_fwd_2nd.DerivativeRef]:
    """Get the derivative of an XAD active type - forward or adjoint mode

    Args:
        x (Real): Argument to extract the derivative information from

    Returns:
        float | DerivativeRef: The derivative; for second order types the inner first order
        Real, by reference (as in C++), resolved on every access (in adjoint mode it is on tape)
    """
    if isinstance(x, _ACTIVE):
        return x.getDerivative()
    else:
        raise TypeError("type " + type(x).__name__ + " is not an XAD active type")

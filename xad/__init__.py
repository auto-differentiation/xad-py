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

from ._xad import adj_1st, fwd_1st, __version__  # noqa: E402

__all__ = [
    "value",
    "derivative",
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


def value(x: Union[adj_1st.Real, fwd_1st.Real, Any]) -> float:
    """Get the value of an XAD active type - or return the value itself otherwise

    Args:
        x (Real | any): Argument to get the value of

    Returns:
        float: The value stored in the variable
    """
    if isinstance(x, adj_1st.Real) or isinstance(x, fwd_1st.Real):
        return x.getValue()
    else:
        return x


def derivative(x: Union[adj_1st.Real, fwd_1st.Real]) -> float:
    """Get the derivative of an XAD active type - forward or adjoint mode

    Args:
        x (Real): Argument to extract the derivative information from

    Returns:
        float: The derivative
    """
    if isinstance(x, adj_1st.Real) or isinstance(x, fwd_1st.Real):
        return x.getDerivative()
    else:
        raise TypeError("type " + type(x).__name__ + " is not an XAD active type")

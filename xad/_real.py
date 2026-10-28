##############################################################################
#
#  Python-side methods of the Real types of every mode
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

from typing import Any, Tuple, Type


def add_float_methods(real: Type[Any]) -> None:
    """Adds the float methods to the Real type of a mode, based on float(x)"""

    def _as_integer_ratio(x: Any) -> Tuple[int, int]:
        """Returns a rational representation of the float with numerator and denominator in a
        tuple"""
        return float(x).as_integer_ratio()

    def _fromhex(cls: Type[Any], hexstr: str) -> Any:
        """Initialize from a hex expression"""
        return cls(float.fromhex(hexstr))

    def _getnewargs(x: Any) -> Tuple[float]:
        return (float(x),)

    def _hash(x: Any) -> int:
        return hash(float(x))

    def _hex(x: Any) -> str:
        return float(x).hex()

    def _is_integer(x: Any) -> bool:
        return float(x).is_integer()

    def _format(x: Any, spec: str) -> str:
        return format(float(x), spec)

    setattr(real, "as_integer_ratio", _as_integer_ratio)
    setattr(real, "fromhex", classmethod(_fromhex))
    setattr(real, "__getnewargs__", _getnewargs)
    setattr(real, "__hash__", _hash)
    setattr(real, "hex", _hex)
    setattr(real, "is_integer", _is_integer)
    setattr(real, "__format__", _format)

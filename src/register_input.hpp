/*******************************************************************************

   Registration of input variables on a tape.

   This file is part of XAD's Python bindings, a comprehensive library for
   automatic differentiation.

   Copyright (C) 2010-2026 Xcelerit Computing Ltd.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU Affero General Public License as published
   by the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Affero General Public License for more details.

   You should have received a copy of the GNU Affero General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

******************************************************************************/

#pragma once

#include <XAD/XAD.hpp>
#include <utility>

// registers inp as a fresh input on tape: drops any slot left from an earlier tape first (XAD
// skips variables with a slot), with no tape active, so the old slot is not unregistered on a
// tape it is not on
template <class Tape>
void register_fresh_input(Tape &tape, typename Tape::active_type &inp)
{
    Tape *active = Tape::getActive();
    Tape::deactivateAll();
    {
        typename Tape::active_type fresh;
        fresh.value() = inp.value();
        inp = std::move(fresh);  // fresh takes the old slot and drops it here
    }
    if (active)
        Tape::setActive(active);
    tape.registerInput(inp);
}

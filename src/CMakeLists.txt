# ##############################################################################
#
# Python bindings using Pybind11 - exension module.
#
# This file is part of XAD's Python bindings, a comprehensive library for
# automatic differentiation.
#
# Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# ##############################################################################

if(MSVC)
  option(XAD_STATIC_MSVC_RUNTIME "Use static runtime" ON)
endif()

add_subdirectory(pybind11)
add_subdirectory(xad)

pybind11_add_module(_xad module.cpp math.hpp tape.hpp real.hpp exceptions.hpp)
target_link_libraries(_xad PRIVATE XAD::xad)
target_compile_definitions(_xad PRIVATE VERSION_INFO=${PROJECT_VERSION})

if(MSVC)
  # to allow using M_PI and the like
  target_compile_definitions(_xad PRIVATE _USE_MATH_DEFINES)
  set_target_properties(_xad PROPERTIES MSVC_RUNTIME_LIBRARY
                                        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

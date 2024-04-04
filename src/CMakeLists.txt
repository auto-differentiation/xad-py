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
option(XAD_NO_THREADLOCAL "Use thread-local tape" ON)

find_package(Threads REQUIRED)

add_subdirectory(pybind11)
add_subdirectory(xad)

# Note: We bake XAD directly into this module, rather than linking, since
# otherwise we get conflicting definitions of the types when used in
# QuantLib-Risks-Py which causes a segfault We assume a platform that at least
# supports the AVX instructions set

pybind11_add_module(
  _xad
  module.cpp
  math.hpp
  tape.hpp
  real.hpp
  exceptions.hpp
  xad/src/Tape.cpp)

target_include_directories(_xad PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/xad/src
                                        xad/src)
set_target_properties(_xad PROPERTIES CXX_STANDARD 17)
target_link_libraries(_xad PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(_xad PRIVATE -nologo -utf-8 "/arch:AVX")
  target_compile_definitions(
    _xad
    PRIVATE VERSION_INFO=${PROJECT_VERSION}
            _UNICODE
            UNICODE
            WIN32_LEAN_AND_MEAN
            WIN32
            _WIN32
            _USE_MATH_DEFINES)
  # static MSVC runtime to avoid linking issues in Python and QuantLib-Risks
  set_target_properties(_xad PROPERTIES MSVC_RUNTIME_LIBRARY
                                        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(_xad PRIVATE -mavx -stdlib=libc++
                                      -mmacosx-version-min=10.9)
  target_compile_definitions(_xad PRIVATE VERSION_INFO=${PROJECT_VERSION})
else()
  target_compile_options(_xad PRIVATE -mavx)
  target_compile_definitions(_xad PRIVATE VERSION_INFO=${PROJECT_VERSION})
endif()

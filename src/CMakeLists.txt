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
option(XAD_PY_STUBS "Generate type stubs with pybind11-stubgen" ON)

find_package(Threads REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)

add_subdirectory(pybind11)
# Only used for XAD's generated headers, its own targets are not built
add_subdirectory(xad EXCLUDE_FROM_ALL)

# Note: We bake XAD directly into this module, rather than linking, since
# otherwise we get conflicting definitions of the types when used in
# QuantLib-Risks-Py which causes a segfault

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

# Package version from pyproject.toml ("dev" otherwise)
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  target_compile_definitions(
    _xad PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION_FULL})
endif()

# AVX only when targeting x86_64
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "#if !defined(__x86_64__) && !defined(_M_X64)
#error not x86_64
#endif
int main() { return 0; }"
  XAD_PY_TARGET_X86_64)

if(MSVC)
  target_compile_options(_xad PRIVATE -nologo -utf-8)
  if(XAD_PY_TARGET_X86_64)
    target_compile_options(_xad PRIVATE /arch:AVX)
  endif()
  target_compile_definitions(_xad PRIVATE _UNICODE UNICODE WIN32_LEAN_AND_MEAN
                                          WIN32 _WIN32 _USE_MATH_DEFINES)
  # static MSVC runtime to avoid linking issues in Python and QuantLib-Risks
  set_target_properties(_xad PROPERTIES MSVC_RUNTIME_LIBRARY
                                        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
elseif(XAD_PY_TARGET_X86_64)
  target_compile_options(_xad PRIVATE -mavx)
endif()

install(TARGETS _xad LIBRARY DESTINATION xad)

# Stubs for the extension, installed as xad/_xad/*.pyi
if(XAD_PY_STUBS)
  set(stubs_dir ${CMAKE_CURRENT_BINARY_DIR}/stubs)
  add_custom_command(
    TARGET _xad
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${stubs_dir}
    COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen -o ${stubs_dir} _xad
    WORKING_DIRECTORY $<TARGET_FILE_DIR:_xad>
    VERBATIM)
  install(DIRECTORY ${stubs_dir}/_xad/ DESTINATION xad/_xad)
endif()

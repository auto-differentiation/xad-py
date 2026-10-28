# ##############################################################################
#
# Python bindings using Pybind11 - exension module.
#
# This file is part of XAD's Python bindings, a comprehensive library for
# automatic differentiation.
#
# Copyright (C) 2010-2026 Xcelerit Computing Ltd.
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

option(XAD_PY_STUBS "Generate type stubs with pybind11-stubgen" ON)

find_package(Threads REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)

add_subdirectory(pybind11)

# shared XAD, so extensions built against xad (get_cmake_dir()) use the same tape
set(XAD_SHARED_LIBS ON)
set(XAD_NO_THREADLOCAL OFF)
set(XAD_STATIC_MSVC_RUNTIME OFF)
set(XAD_WARNINGS_PARANOID OFF)
# AVX only when targeting x86_64
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "#if !defined(__x86_64__) && !defined(_M_X64)
#error not x86_64
#endif
int main() { return 0; }"
  XAD_PY_TARGET_X86_64)
if(XAD_PY_TARGET_X86_64)
  set(XAD_SIMD_OPTION AVX CACHE STRING "")
else()
  set(XAD_SIMD_OPTION NONE CACHE STRING "")
endif()
set(CMAKE_INSTALL_INCLUDEDIR xad/include)
add_subdirectory(xad EXCLUDE_FROM_ALL)
if(WIN32)
  # xad.dll next to _xad, also for importing it at build time (stubs)
  set_target_properties(xad PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                       ${CMAKE_CURRENT_BINARY_DIR})
endif()

pybind11_add_module(
  _xad
  module.cpp
  array.cpp
  array.hpp
  math.hpp
  cmath.hpp
  tape.hpp
  real.hpp
  complex.hpp
  exceptions.hpp
  jacobian_hessian.hpp
  double_caster.hpp)

set_target_properties(_xad PROPERTIES CXX_STANDARD 17)
target_link_libraries(_xad PRIVATE XAD::xad Threads::Threads)
if(APPLE)
  set_target_properties(_xad PROPERTIES INSTALL_RPATH "@loader_path/lib")
elseif(UNIX)
  set_target_properties(_xad PROPERTIES INSTALL_RPATH "$ORIGIN/lib")
endif()

# Package version from pyproject.toml ("dev" otherwise)
if(DEFINED SKBUILD_PROJECT_VERSION_FULL)
  target_compile_definitions(
    _xad PRIVATE VERSION_INFO=${SKBUILD_PROJECT_VERSION_FULL})
endif()


if(MSVC)
  target_compile_options(_xad PRIVATE -nologo -utf-8)
  if(XAD_PY_TARGET_X86_64)
    target_compile_options(_xad PRIVATE /arch:AVX)
  endif()
  target_compile_definitions(_xad PRIVATE _UNICODE UNICODE WIN32_LEAN_AND_MEAN
                                          WIN32 _WIN32 _USE_MATH_DEFINES)
  set_target_properties(
    _xad PROPERTIES MSVC_RUNTIME_LIBRARY
                    "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
elseif(XAD_PY_TARGET_X86_64)
  target_compile_options(_xad PRIVATE -mavx)
endif()

install(
  TARGETS _xad
  LIBRARY DESTINATION xad
  COMPONENT python)

# libxad, headers and a relocatable CMake config (XAD's install layout doesn't fit the wheel)
set(xad_cmake_dir xad/share/cmake/XAD)
install(
  TARGETS xad
  EXPORT XADPyTargets
  RUNTIME DESTINATION xad COMPONENT python
  LIBRARY DESTINATION xad/lib COMPONENT python
  ARCHIVE DESTINATION xad/lib COMPONENT python)
install(
  EXPORT XADPyTargets
  FILE XADTargets.cmake
  NAMESPACE XAD::
  DESTINATION ${xad_cmake_dir}
  COMPONENT python)

get_target_property(xad_src_dir xad SOURCE_DIR)
get_target_property(xad_sources xad SOURCES)
foreach(f IN LISTS xad_sources)
  if(f MATCHES "\\.hpp$")
    if(NOT IS_ABSOLUTE "${f}")
      set(f "${xad_src_dir}/${f}")
    endif()
    install(
      FILES ${f}
      DESTINATION xad/include/XAD
      COMPONENT python)
  endif()
endforeach()

include(CMakePackageConfigHelpers)
set(xad_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/xad-config)
configure_package_config_file(
  xad/cmake/XADConfig.cmake.in ${xad_gen_dir}/XADConfig.cmake
  INSTALL_DESTINATION ${xad_cmake_dir})
install(
  FILES ${xad_gen_dir}/XADConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/xad/src/generated/XADConfigVersion.cmake
  DESTINATION ${xad_cmake_dir}
  COMPONENT python)

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
  install(
    DIRECTORY ${stubs_dir}/_xad/
    DESTINATION xad/_xad
    COMPONENT python)
endif()

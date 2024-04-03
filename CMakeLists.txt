# ##############################################################################
#
# Python bindings for XAD - CMake file called during poetry build.
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

cmake_minimum_required(VERSION 3.15.2)
cmake_policy(SET CMP0092 NEW) # don't set warning flags by default in Windows

project(
  xad-py
  LANGUAGES CXX
  DESCRIPTION "Python bindings for the XAD automatic differentiation library"
  HOMEPAGE_URL "https://auto-differentiation.github.io")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/xad/cmake)
include(SetupVersion)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src)

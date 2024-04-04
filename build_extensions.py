##############################################################################
#
#  Build file for extension module - using pre-built binary with pybind.
#
#  This was inspired by:
#  https://github.com/pybind/cmake_example/blob/master/setup.py
#
#  This file is part of XAD's Python bindings, a comprehensive library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
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

import re
import subprocess
import sys
import os
from pathlib import Path

try:
    from setuptools import Extension as _Extension
    from setuptools.command.build_ext import build_ext as _build_ext
except ImportError:
    from distutils.command.build_ext import (  # type: ignore[assignment]
        build_ext as _build_ext,
    )
    from distutils.extension import Extension as _Extension  # type: ignore[assignment]


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


def get_vsvars_environment(architecture="amd64", toolset="14.3"):
    """Returns a dictionary containing the environment variables set up by vsvarsall.bat
    architecture - Architecture to pass to vcvarsall.bat. Normally "x86" or "amd64"
    win32-specific
    """
    result = None
    python = sys.executable

    for vcvarsall in [
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2022 Enterprise
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2022 Pro
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2022 Community edition
        "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2022 Build Tools
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2019 Enterprise
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2019 Pro
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2019 Community edition
        "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\VC\\Auxiliary\\Build\\vcvarsall.bat",  # VS2019 Build tools
    ]:
        if os.path.isfile(vcvarsall):
            command = f'("{vcvarsall}" {architecture} -vcvars_ver={toolset}>nul)&&"{python}" -c "import os; print(repr(os.environ))"'
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                shell=True,
            )
            stdout, _ = process.communicate()
            exitcode = process.wait()
            if exitcode == 0:
                result = eval(stdout.decode("ascii").strip("environ"))
                break
    if not result:
        raise Exception("Couldn't find/process vcvarsall batch file")
    return result


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(_Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(_build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        env = get_vsvars_environment() if self.compiler.compiler_type == "msvc" else None

        # Using Ninja-build
        if not cmake_generator or cmake_generator == "Ninja":
            import ninja

            ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
            cmake_args += [
                "-GNinja",
                f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
            ]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True, env=env)
        subprocess.run(["cmake", "--build", ".", *build_args], cwd=build_temp, check=True, env=env)

        # generate stubs
        import pybind11_stubgen

        save_args = sys.argv
        save_dir = os.getcwd()
        save_path = sys.path
        sys.argv = ["<dummy>", "-o", ".", "_xad"]
        os.chdir(str(extdir))
        sys.path = [str(extdir), *save_path]
        pybind11_stubgen.main()
        os.chdir(save_dir)
        sys.argv = save_args
        sys.path = save_path


def build(setup_kwargs: dict):
    """Main extension build command."""

    ext_modules = [CMakeExtension("xad._xad")]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {"build_ext": CMakeBuild},
            "zip_safe": False,
        }
    )

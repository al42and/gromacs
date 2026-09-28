#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2022- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

if ("$ENV{RUNNER_OS}" STREQUAL "Windows" AND NOT "x$ENV{ENVIRONMENT_SCRIPT}" STREQUAL "x")
  set(environment_script "$ENV{ENVIRONMENT_SCRIPT}")
  if (NOT EXISTS "${environment_script}" AND "$ENV{RUNNER_ARCH}" STREQUAL "ARM64")
    file(GLOB vcvars_candidates
      "C:/Program Files/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvarsarm64.bat"
      "C:/Program Files/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvarsamd64_arm64.bat"
      "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvarsarm64.bat"
      "C:/Program Files (x86)/Microsoft Visual Studio/*/*/VC/Auxiliary/Build/vcvarsamd64_arm64.bat"
    )
    if (vcvars_candidates)
      list(GET vcvars_candidates 0 environment_script)
    endif()
  endif()
  if (NOT EXISTS "${environment_script}")
    message(FATAL_ERROR "Visual Studio environment script not found: ${environment_script}")
  endif()
  set(environment_dump_script "${CMAKE_CURRENT_BINARY_DIR}/environment_dump.bat")
  file(WRITE "${environment_dump_script}" "@echo off\r\ncall \"${environment_script}\" >nul\r\nset\r\n")
  execute_process(
    COMMAND cmd /C "${environment_dump_script}"
    OUTPUT_FILE environment_script_output.txt
  )
  file(REMOVE "${environment_dump_script}")
  file(STRINGS environment_script_output.txt output_lines)
  foreach(line IN LISTS output_lines)
    if (line MATCHES "^([a-zA-Z0-9_-]+)=(.*)$")
      set(ENV{${CMAKE_MATCH_1}} "${CMAKE_MATCH_2}")
    endif()
  endforeach()
endif()

set(path_separator ":")
if ("$ENV{RUNNER_OS}" STREQUAL "Windows")
  set(path_separator ";")
endif()
set(ENV{PATH} "$ENV{GITHUB_WORKSPACE}${path_separator}$ENV{PATH}")

message(STATUS "Using GPU_VAR: $ENV{GPU_VAR}")

execute_process(
  COMMAND cmake
    -S .
    -B build
    -D CMAKE_BUILD_TYPE=$ENV{BUILD_TYPE}
    -G Ninja
    -D CMAKE_MAKE_PROGRAM=ninja
    -D CMAKE_C_COMPILER_LAUNCHER=ccache
    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache
    -D GMX_COMPILER_WARNINGS=ON
    -D GMX_DEFAULT_SUFFIX=OFF
    -D GMX_GPU=$ENV{GPU_VAR}
    -D GMX_SIMD=None
    -D GMX_FFT_LIBRARY=FFTPACK
    -D GMX_OPENMP=$ENV{OPENMP_VAR}
    -D REGRESSIONTEST_DOWNLOAD=ON
  RESULT_VARIABLE result
)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Bad exit status")
endif()

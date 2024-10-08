/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2010- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * main() for unit tests that use \ref module_testutils.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_testutils
 */
#include "gmxpre.h"

#include <filesystem>

#include <gtest/gtest.h>

#include "testutils/testinit.h"

#ifndef TEST_DATA_PATH
//! Path to test input data directory (needs to be set by the build system).
#    define TEST_DATA_PATH ""
#endif

#ifndef TEST_TEMP_PATH
//! Path to test output temporary directory (needs to be set by the build system).
#    define TEST_TEMP_PATH ""
#endif

#ifndef TEST_USES_MPI
//! Whether the test expects/supports running with multiple MPI ranks.
#    define TEST_USES_MPI false
#endif

#ifndef TEST_USES_HARDWARE_DETECTION
//! Whether the test expects/supports running with knowledge of the hardware.
#    define TEST_USES_HARDWARE_DETECTION false
#endif

#ifndef TEST_USES_DYNAMIC_REGISTRATION
//! Whether tests will be dynamically registered
#    define TEST_USES_DYNAMIC_REGISTRATION false
namespace gmx
{
namespace test
{
// Stub implementation for test suites that do not use dynamic
// registration.
void registerTestsDynamically() {}
} // namespace test
} // namespace gmx
#endif

/*! \brief
 * Initializes unit testing for \ref module_testutils.
 */
int main(int argc, char* argv[])
{
    // Calls ::testing::InitGoogleMock()
    ::gmx::test::initTestUtils(TEST_DATA_PATH,
                               TEST_TEMP_PATH,
                               TEST_USES_MPI,
                               TEST_USES_HARDWARE_DETECTION,
                               TEST_USES_DYNAMIC_REGISTRATION,
                               &argc,
                               &argv);
    int errcode = RUN_ALL_TESTS();
    ::gmx::test::finalizeTestUtils(TEST_USES_HARDWARE_DETECTION, TEST_USES_DYNAMIC_REGISTRATION);
    return errcode;
}

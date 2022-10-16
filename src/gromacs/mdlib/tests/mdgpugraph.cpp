/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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
 * \brief Tests for MD GPU graph
 *
 * \author Alan Gray <alang@nvidia.com>
 */
#include "gmxpre.h"

#include "config.h"

#include <gtest/gtest.h>

#if GMX_HAVE_CUDA_GRAPH_SUPPORT

#    include "gromacs/gpu_utils/device_stream.h"
#    include "gromacs/gpu_utils/device_stream_manager.h"
#    include "gromacs/gpu_utils/devicebuffer.h"
#    include "gromacs/gpu_utils/gpueventsynchronizer.h"
#    include "gromacs/gpu_utils/hostallocator.h"
#    include "gromacs/mdlib/mdgraph_gpu.h"

#    include "testutils/refdata.h"
#    include "testutils/test_hardware_environment.h"
#    include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{


TEST(MdGraphTest, MdGpuGraphExecutesActivities)
{

    const auto& testDevice    = getTestHardwareEnvironment()->getTestDeviceList()[0];
    const auto& deviceContext = testDevice->deviceContext();

    // Initialize required structures
    SimulationWorkload simulationWork;
    simulationWork.useGpuPme                 = true;
    simulationWork.useGpuUpdate              = true;
    simulationWork.useMdGpuGraph             = true;
    simulationWork.havePpDomainDecomposition = false;
    simulationWork.haveSeparatePmeRank       = true;
    DeviceStreamManager  deviceStreamManager(testDevice->deviceInfo(), simulationWork, false);
    GpuEventSynchronizer xReadyOnDeviceEvent;
    GpuEventSynchronizer xUpdatedOnDeviceEvent;
    gmx::MdGpuGraph      mdGpuGraph(
            deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
    gmx::MdGpuGraph mdGpuGraphAlternate(
            deviceStreamManager, simulationWork, MPI_COMM_WORLD, MdGraphEvenOrOddStep::EvenStep, nullptr);
    mdGpuGraph.setAlternateStepPpTaskCompletionEvent(mdGpuGraphAlternate.getPpTaskCompletionEvent());

    // Allocate 2 device buffers
    DeviceBuffer<int> d_output;
    int               d_output_size       = -1;
    int               d_output_size_alloc = -1;
    reallocateDeviceBuffer(&d_output, 1, &d_output_size, &d_output_size_alloc, deviceContext);
    DeviceBuffer<int> d_staging;
    int               d_staging_size       = -1;
    int               d_staging_size_alloc = -1;
    reallocateDeviceBuffer(&d_staging, 1, &d_staging_size, &d_staging_size_alloc, deviceContext);

    // Perform below steps with and without graph
    for (bool useGraph : { false, true })
    {

        HostVector<int> h_one;
        changePinningPolicy(&h_one, PinningPolicy::PinnedIfSupported);
        h_one.resize(1);
        h_one[0] = 1;

        HostVector<int> h_output;
        changePinningPolicy(&h_output, PinningPolicy::PinnedIfSupported);
        h_output.resize(1);

        // Set output to 1 on GPU
        copyToDeviceBuffer(&d_output,
                           h_one.data(),
                           0,
                           1,
                           deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                           GpuApiCallBehavior::Sync,
                           nullptr);

        if (useGraph && mdGpuGraph.captureThisStep(true)) // denote start of graph region
        {
            // Start graph capture (automatically on local stream)
            bool usedGraphLastStep = true;
            mdGpuGraph.setUsedGraphLastStep(usedGraphLastStep);
            mdGpuGraph.startRecord(&xReadyOnDeviceEvent);
        }

        // Clear output on GPU in update stream, which will be automatically forked from
        // local stream in the graph. Can be done in single call, but instead perform in
        // 2 stages to create a 2-node graph.
        const DeviceStream& stream =
                deviceStreamManager.stream(gmx::DeviceStreamType::UpdateAndConstraints);
        clearDeviceBufferAsync(&d_staging, 0, 1, stream);
        copyBetweenDeviceBuffers(&d_output, &d_staging, 1, stream, GpuApiCallBehavior::Async, nullptr);

        if (mdGpuGraph.graphIsCapturingThisStep()) // denote end of graph region
        {
            // End graph capture and instantiate
            mdGpuGraph.endRecord();
            mdGpuGraph.createExecutableGraph();
        }

        // Wait for update stream
        deviceStreamManager.stream(gmx::DeviceStreamType::UpdateAndConstraints).synchronize();
        // Synchronously copy output to host buffer
        copyFromDeviceBuffer(h_output.data(),
                             &d_output,
                             0,
                             1,
                             deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                             GpuApiCallBehavior::Sync,
                             nullptr);

        if (mdGpuGraph.useGraphThisStep())
        {
            // Graph has not yet been launched so output has not yet been cleared
            EXPECT_EQ(h_output[0], 1);
        }
        else
        {
            // Without graph capture active, the above memory operations will have been performed
            // directly and output will now be cleared
            EXPECT_EQ(h_output[0], 0);
        }

        if (mdGpuGraph.useGraphThisStep())
        {
            // Now launch graph and check output is cleared
            mdGpuGraph.launchGraphMdStep(&xUpdatedOnDeviceEvent);
            xUpdatedOnDeviceEvent.waitForEvent();

            // Synchronously copy output to host buffer in local stream
            copyFromDeviceBuffer(h_output.data(),
                                 &d_output,
                                 0,
                                 1,
                                 deviceStreamManager.stream(gmx::DeviceStreamType::NonBondedLocal),
                                 GpuApiCallBehavior::Sync,
                                 nullptr);

            // Graph has now been executed, so output is cleared
            EXPECT_EQ(h_output[0], 0);
        }
    }
}

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_GPU_CUDA

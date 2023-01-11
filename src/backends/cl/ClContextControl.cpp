//
// Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControl.hpp"

#include <armnn/Exceptions.hpp>

#include <LeakChecking.hpp>

#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <fmt/format.h>

namespace cl
{
class Context;
class CommandQueue;
class Device;
}

namespace armnn
{

ClContextControl::ClContextControl(arm_compute::CLTuner *tuner,
                                   arm_compute::CLGEMMHeuristicsHandle* heuristicsHandle,
                                   bool profilingEnabled)
    : m_Tuner(tuner)
    , m_HeuristicsHandle(heuristicsHandle)
    , m_ProfilingEnabled(profilingEnabled)
{
    // Ignore m_ProfilingEnabled if unused to avoid compiling problems when ArmCompute is disabled.
    IgnoreUnused(m_ProfilingEnabled);

    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Selects default platform for the first element.
        cl::Platform::setDefault(platforms[0]);

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Selects default device for the first element.
        cl::Device::setDefault(devices[0]);
    }
    catch (const cl::Error& clError)
    {
        throw ClRuntimeUnavailableException(fmt::format(
            "Could not initialize the CL runtime. Error description: {0}. CL error code: {1}",
            clError.what(), clError.err()));
    }

    // Removes the use of global CL context.
    cl::Context::setDefault(cl::Context{});
    ARMNN_ASSERT(cl::Context::getDefault()() == NULL);

    // Removes the use of global CL command queue.
    cl::CommandQueue::setDefault(cl::CommandQueue{});
    ARMNN_ASSERT(cl::CommandQueue::getDefault()() == NULL);

    // Always load the OpenCL runtime.
    LoadOpenClRuntime();
}

ClContextControl::~ClContextControl()
{
    // Load the OpencCL runtime without the tuned parameters to free the memory for them.
    try
    {
        UnloadOpenClRuntime();
    }
    catch (const cl::Error& clError)
    {
        // This should not happen, it is ignored if it does.

        // Coverity fix: BOOST_LOG_TRIVIAL (previously used here to report the error) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "A CL error occurred unloading the runtime tuner parameters: "
                  << clError.what() << ". CL error code is: " << clError.err() << std::endl;
    }
}

void ClContextControl::LoadOpenClRuntime()
{
    DoLoadOpenClRuntime(true);
}

void ClContextControl::UnloadOpenClRuntime()
{
    DoLoadOpenClRuntime(false);
}

void ClContextControl::DoLoadOpenClRuntime(bool updateTunedParameters)
{
    cl::Device device = cl::Device::getDefault();
    cl::Context context;
    cl::CommandQueue commandQueue;

    if (arm_compute::CLScheduler::get().is_initialised() && arm_compute::CLScheduler::get().context()() != NULL)
    {
        // Wait for all queued CL requests to finish before reinitialising it.
        arm_compute::CLScheduler::get().sync();
    }

    try
    {
        arm_compute::CLKernelLibrary::get().clear_programs_cache();
        // Initialise the scheduler with a dummy context to release the LLVM data (which only happens when there are no
        // context references); it is initialised again, with a proper context, later.
        arm_compute::CLScheduler::get().init(context, commandQueue, device);
        arm_compute::CLKernelLibrary::get().init(".", context, device);

        {
            //
            // Here we replace the context with a new one in which
            // the memory leak checks show it as an extra allocation but
            // because of the scope of the leak checks, it doesn't count
            // the disposal of the original object. On the other hand it
            // does count the creation of this context which it flags
            // as a memory leak. By adding the following line we prevent
            // this to happen.
            //
            ARMNN_DISABLE_LEAK_CHECKING_IN_SCOPE();
            context = cl::Context(device);
        }

        // NOTE: In this specific case profiling has to be enabled on the command queue
        // in order for the CLTuner to work.
        bool profilingNeededForClTuner = updateTunedParameters && m_Tuner &&
            m_Tuner->tune_new_kernels();

        if (m_ProfilingEnabled || profilingNeededForClTuner)
        {
            // Create a new queue with profiling enabled.
            commandQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }
        else
        {
            // Use default queue.
            commandQueue = cl::CommandQueue(context, device);
        }
    }
    catch (const cl::Error& clError)
    {
        throw ClRuntimeUnavailableException(fmt::format(
            "Could not initialize the CL runtime. Error description: {0}. CL error code: {1}",
            clError.what(), clError.err()));
    }

    // Note the first argument (path to cl source code) will be ignored as they should be embedded in the armcompute.
    arm_compute::CLKernelLibrary::get().init(".", context, device);
    arm_compute::CLScheduler::get().init(context, commandQueue, device, m_Tuner, m_HeuristicsHandle);
}

void ClContextControl::ClearClCache()
{
    DoLoadOpenClRuntime(true);
}

} // namespace armnn

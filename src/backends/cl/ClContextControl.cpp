//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControl.hpp"

#include <armnn/Exceptions.hpp>

#include <LeakChecking.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/polymorphic_cast.hpp>
#include <boost/core/ignore_unused.hpp>

namespace cl
{
class Context;
class CommandQueue;
class Device;
}

namespace armnn
{

ClContextControl::ClContextControl(IGpuAccTunedParameters* clTunedParameters,
                                   bool profilingEnabled)
    : m_clTunedParameters(boost::polymorphic_downcast<ClTunedParameters*>(clTunedParameters))
    , m_ProfilingEnabled(profilingEnabled)
{
    // Ignore m_ProfilingEnabled if unused to avoid compiling problems when ArmCompute is disabled.
    boost::ignore_unused(m_ProfilingEnabled);

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
        throw ClRuntimeUnavailableException(boost::str(boost::format(
            "Could not initialize the CL runtime. Error description: %1%. CL error code: %2%"
        ) % clError.what() % clError.err()));
    }

    // Removes the use of global CL context.
    cl::Context::setDefault(cl::Context{});
    BOOST_ASSERT(cl::Context::getDefault()() == NULL);

    // Removes the use of global CL command queue.
    cl::CommandQueue::setDefault(cl::CommandQueue{});
    BOOST_ASSERT(cl::CommandQueue::getDefault()() == NULL);

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

void ClContextControl::DoLoadOpenClRuntime(bool useTunedParameters)
{
    cl::Device device = cl::Device::getDefault();
    cl::Context context;
    cl::CommandQueue commandQueue;

    if (arm_compute::CLScheduler::get().context()() != NULL)
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
        bool profilingNeededForClTuner = useTunedParameters && m_clTunedParameters &&
            m_clTunedParameters->m_Mode == IGpuAccTunedParameters::Mode::UpdateTunedParameters;

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
        throw ClRuntimeUnavailableException(boost::str(boost::format(
            "Could not initialize the CL runtime. Error description: %1%. CL error code: %2%"
        ) % clError.what() % clError.err()));
    }

    // Note the first argument (path to cl source code) will be ignored as they should be embedded in the armcompute.
    arm_compute::CLKernelLibrary::get().init(".", context, device);

    arm_compute::ICLTuner* tuner = nullptr;
    if (useTunedParameters && m_clTunedParameters)
    {
        tuner = &m_clTunedParameters->m_Tuner;
        auto clTuner = boost::polymorphic_downcast<arm_compute::CLTuner*>(tuner);

        auto ConvertTuningLevel = [](IGpuAccTunedParameters::TuningLevel level)
        {
            switch(level)
            {
                case IGpuAccTunedParameters::TuningLevel::Rapid:
                    return arm_compute::CLTunerMode::RAPID;
                case IGpuAccTunedParameters::TuningLevel::Normal:
                    return arm_compute::CLTunerMode::NORMAL;
                case IGpuAccTunedParameters::TuningLevel::Exhaustive:
                    return arm_compute::CLTunerMode::EXHAUSTIVE;
                default:
                {
                    BOOST_ASSERT_MSG(false, "Tuning level not recognised.");
                    return arm_compute::CLTunerMode::NORMAL;
                }
            }
        };

        clTuner->set_tuner_mode(ConvertTuningLevel(m_clTunedParameters->m_TuningLevel));
    }
    arm_compute::CLScheduler::get().init(context, commandQueue, device, tuner);
}

void ClContextControl::ClearClCache()
{
    DoLoadOpenClRuntime(true);
}

armnn::IGpuAccTunedParameters* IGpuAccTunedParameters::CreateRaw(armnn::IGpuAccTunedParameters::Mode mode,
                                                                 armnn::IGpuAccTunedParameters::TuningLevel tuningLevel)
{
    return new ClTunedParameters(mode, tuningLevel);
}

armnn::IGpuAccTunedParametersPtr IGpuAccTunedParameters::Create(armnn::IGpuAccTunedParameters::Mode mode,
                                                                armnn::IGpuAccTunedParameters::TuningLevel tuningLevel)
{
    return IGpuAccTunedParametersPtr(CreateRaw(mode, tuningLevel), &IGpuAccTunedParameters::Destroy);
}

void IGpuAccTunedParameters::Destroy(IGpuAccTunedParameters* params)
{
    delete params;
}

ClTunedParameters::ClTunedParameters(armnn::IGpuAccTunedParameters::Mode mode,
                                     armnn::IGpuAccTunedParameters::TuningLevel tuningLevel)
    : m_Mode(mode)
    , m_TuningLevel(tuningLevel)
    , m_Tuner(mode == ClTunedParameters::Mode::UpdateTunedParameters)
{
}

void ClTunedParameters::Load(const char* filename)
{
    try
    {
        m_Tuner.load_from_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to load tuned parameters file '") + filename + "': " +
            e.what());
    }
}

void ClTunedParameters::Save(const char* filename) const
{
    try
    {
        m_Tuner.save_to_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to save tuned parameters file to '") + filename + "': " +
            e.what());
    }
}

} // namespace armnn

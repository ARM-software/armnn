//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClContextControl.hpp"

#include "armnn/Exceptions.hpp"

#ifdef ARMCOMPUTECL_ENABLED
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#endif

#include <boost/assert.hpp>
#include <boost/format.hpp>
#include <boost/log/trivial.hpp>
#include <boost/polymorphic_cast.hpp>

#include "LeakChecking.hpp"

namespace cl
{
class Context;
class CommandQueue;
class Device;
}

namespace armnn
{

ClContextControl::ClContextControl(IClTunedParameters* clTunedParameters)
    : m_clTunedParameters(boost::polymorphic_downcast<ClTunedParameters*>(clTunedParameters))
{
#ifdef ARMCOMPUTECL_ENABLED
    try
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        // Select default platform as the first element
        cl::Platform::setDefault(platforms[0]);

        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // Select default device as the first element
        cl::Device::setDefault(devices[0]);
    }
    catch (const cl::Error& clError)
    {
        throw ClRuntimeUnavailableException(boost::str(boost::format(
            "Could not initialize the CL runtime. Error description: %1%. CL error code: %2%"
        ) % clError.what() % clError.err()));
    }

    // Remove the use of global CL context
    cl::Context::setDefault(cl::Context{});
    BOOST_ASSERT(cl::Context::getDefault()() == NULL);

    // Remove the use of global CL command queue
    cl::CommandQueue::setDefault(cl::CommandQueue{});
    BOOST_ASSERT(cl::CommandQueue::getDefault()() == NULL);

    // always load the OpenCL runtime
    LoadOpenClRuntime();
#endif
}

ClContextControl::~ClContextControl()
{
#ifdef ARMCOMPUTECL_ENABLED
    // load the OpencCL runtime without the tuned parameters to free the memory for them
    try
    {
        UnloadOpenClRuntime();
    }
    catch (const cl::Error& clError)
    {
        // this should not happen, it is ignored if it does

        // Coverity fix: BOOST_LOG_TRIVIAL (previously used here to report the error) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "A CL error occurred unloading the runtime tuner parameters: "
                  << clError.what() << ". CL error code is: " << clError.err() << std::endl;
    }
#endif
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
#ifdef ARMCOMPUTECL_ENABLED
    cl::Device device = cl::Device::getDefault();
    cl::Context context;
    cl::CommandQueue commandQueue;

    if (arm_compute::CLScheduler::get().context()() != NULL)
    {
        // wait for all queued CL requests to finish before reinitialising it
        arm_compute::CLScheduler::get().sync();
    }

    try
    {
        arm_compute::CLKernelLibrary::get().clear_programs_cache();
        // initialise the scheduler with a dummy context to release the LLVM data (which only happens when there are no
        // context references); it is initialised again, with a proper context, later.
        arm_compute::CLScheduler::get().init(context, commandQueue, device);
        arm_compute::CLKernelLibrary::get().init(".", context, device);

        {
            //
            // Here we replace the context with a new one which in
            // the memory leak checks shows as an extra allocation but
            // because of the scope of the leak check it doesn't count
            // the disposal of the original object. On the other hand it
            // does count the creation of this context which it flags
            // as a memory leak. By adding the following line we prevent
            // this to happen.
            //
            ARMNN_DISABLE_LEAK_CHECKING_IN_SCOPE();
            context = cl::Context(device);
        }

        bool enableProfiling = false;
#if ARMNN_PROFILING_ENABLED
        enableProfiling = true;
#endif
        if (useTunedParameters &&
            m_clTunedParameters && m_clTunedParameters->m_Mode == IClTunedParameters::Mode::UpdateTunedParameters)
        {
            enableProfiling = true; // Needed for the CLTuner to work.
        }

        if (enableProfiling)
        {
            // Create a new queue with profiling enabled
            commandQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }
        else
        {
            // Use default queue
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
    }
    arm_compute::CLScheduler::get().init(context, commandQueue, device, tuner);
#endif
}

void ClContextControl::ClearClCache()
{
    DoLoadOpenClRuntime(true);
}

armnn::IClTunedParameters* IClTunedParameters::CreateRaw(armnn::IClTunedParameters::Mode mode)
{
    return new ClTunedParameters(mode);
}

armnn::IClTunedParametersPtr IClTunedParameters::Create(armnn::IClTunedParameters::Mode mode)
{
    return IClTunedParametersPtr(CreateRaw(mode), &IClTunedParameters::Destroy);
}

void IClTunedParameters::Destroy(IClTunedParameters* params)
{
    delete params;
}

ClTunedParameters::ClTunedParameters(armnn::IClTunedParameters::Mode mode)
    : m_Mode(mode)
#ifdef ARMCOMPUTECL_ENABLED
    , m_Tuner(mode == ClTunedParameters::Mode::UpdateTunedParameters)
#endif
{
}

void ClTunedParameters::Load(const char* filename)
{
#ifdef ARMCOMPUTECL_ENABLED
    try
    {
        m_Tuner.load_from_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to load tuned parameters file '") + filename + "': " +
            e.what());
    }
#endif
}

void ClTunedParameters::Save(const char* filename) const
{
#ifdef ARMCOMPUTECL_ENABLED
    try
    {
        m_Tuner.save_to_file(filename);
    }
    catch (const std::exception& e)
    {
        throw armnn::Exception(std::string("Failed to save tuned parameters file to '") + filename + "': " +
            e.what());
    }
#endif
}

} // namespace armnn

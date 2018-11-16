//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackendContext.hpp"
#include "ClContextControl.hpp"

#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <boost/log/trivial.hpp>

namespace armnn
{

struct ClBackendContext::ClContextControlWrapper
{
    ClContextControlWrapper(IGpuAccTunedParameters* clTunedParameters,
                            bool profilingEnabled)
        : m_ClContextControl(clTunedParameters, profilingEnabled)
    {}

    bool Sync()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // Waits for all queued CL requests to finish before unloading the network they may be using.
            try
            {
                // Coverity fix: arm_compute::CLScheduler::sync() may throw an exception of type cl::Error.
                arm_compute::CLScheduler::get().sync();
            }
            catch (const cl::Error&)
            {
                BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): an error occurred while waiting for "
                                              "the queued CL requests to finish";
                return false;
            }
        }

        return true;
    }

    void ClearClCache()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // There are no loaded networks left, so clear the CL cache to free up memory
            m_ClContextControl.ClearClCache();
        }
    }

    ClContextControl m_ClContextControl;
};


ClBackendContext::ClBackendContext(const IRuntime::CreationOptions& options)
    : IBackendContext(options)
    , m_ClContextControlWrapper(
        std::make_unique<ClContextControlWrapper>(options.m_GpuAccTunedParameters.get(),
                                                  options.m_EnableGpuProfiling))
{
}

bool ClBackendContext::BeforeLoadNetwork(NetworkId)
{
    return true;
}

bool ClBackendContext::AfterLoadNetwork(NetworkId networkId)
{
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.insert(networkId);
    }
    return true;
}

bool ClBackendContext::BeforeUnloadNetwork(NetworkId)
{
    return m_ClContextControlWrapper->Sync();
}

bool ClBackendContext::AfterUnloadNetwork(NetworkId networkId)
{
    bool clearCache = false;
    {
        std::lock_guard<std::mutex> lockGuard(m_Mutex);
        m_NetworkIds.erase(networkId);
        clearCache = m_NetworkIds.empty();
    }

    if (clearCache)
    {
        m_ClContextControlWrapper->ClearClCache();
    }

    return true;
}

ClBackendContext::~ClBackendContext()
{
}

} // namespace armnn
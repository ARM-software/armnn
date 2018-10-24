//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackendContext.hpp"
#include "ClBackendId.hpp"
#include "ClContextControl.hpp"

#include <backends/BackendContextRegistry.hpp>
#include <boost/log/trivial.hpp>

#include <mutex>

#ifdef ARMCOMPUTECL_ENABLED
// Needed for the CL scheduler calls
#include <arm_compute/core/CL/OpenCL.h>
#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>
#endif

namespace armnn
{

namespace
{

static StaticRegistryInitializer<BackendContextRegistry> g_RegisterHelper
{
    BackendContextRegistryInstance(),
    ClBackendId(),
    [](const IRuntime::CreationOptions& options)
    {
        return IBackendContextUniquePtr(new ClBackendContext{options});
    }
};

static std::mutex g_ContextControlMutex;

std::shared_ptr<ClBackendContext::ContextControlWrapper>
GetContextControlWrapper(const IRuntime::CreationOptions& options)
{
    static std::weak_ptr<ClBackendContext::ContextControlWrapper> contextControlWrapper;

    std::lock_guard<std::mutex> lockGuard(g_ContextControlMutex);
    std::shared_ptr<ClBackendContext::ContextControlWrapper> result;

    if (contextControlWrapper.expired())
    {
        result = std::make_shared<ClBackendContext::ContextControlWrapper>(options);
        contextControlWrapper = result;
    }
    else
    {
        result = contextControlWrapper.lock();
    }

    return result;
}

} // anonymous namespace


#ifdef ARMCOMPUTECL_ENABLED
struct ClBackendContext::ContextControlWrapper
{
    ContextControlWrapper(const IRuntime::CreationOptions& options)
    : m_ClContextControl{options.m_GpuAccTunedParameters.get(),
                         options.m_EnableGpuProfiling}
    {
    }

    ~ContextControlWrapper()
    {
        if (arm_compute::CLScheduler::get().context()() != NULL)
        {
            // Waits for all queued CL requests to finish before unloading the network they may be using.
            try
            {
                // Coverity fix: arm_compute::CLScheduler::sync() may throw an exception of type cl::Error.
                arm_compute::CLScheduler::get().sync();
                m_ClContextControl.ClearClCache();
            }
            catch (const cl::Error&)
            {
                BOOST_LOG_TRIVIAL(warning) << "WARNING: Runtime::UnloadNetwork(): an error occurred while waiting for "
                                            "the queued CL requests to finish";
            }
        }
    }

    ClContextControl m_ClContextControl;
};
#else //ARMCOMPUTECL_ENABLED
struct ClBackendContext::ContextControlWrapper
{
    ContextControlWrapper(const IRuntime::CreationOptions&){}
};
#endif //ARMCOMPUTECL_ENABLED

ClBackendContext::ClBackendContext(const IRuntime::CreationOptions& options)
: IBackendContext{options}
, m_ContextControl{GetContextControlWrapper(options)}
{
}

ClBackendContext::~ClBackendContext()
{
}

} // namespace armnn
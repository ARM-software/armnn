//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"
#include "NeonBackendId.hpp"
#include "NeonWorkloadFactory.hpp"

#include <backendsCommon/IBackendContext.hpp>
#include <backendsCommon/BackendRegistry.hpp>
#include <Optimizer.hpp>

#include <boost/cast.hpp>

namespace armnn
{

namespace
{

static StaticRegistryInitializer<BackendRegistry> g_RegisterHelper
{
    BackendRegistryInstance(),
    NeonBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new NeonBackend);
    }
};

}

const BackendId& NeonBackend::GetIdStatic()
{
    static const BackendId s_Id{NeonBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr NeonBackend::CreateWorkloadFactory() const
{
    return std::make_unique<NeonWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr NeonBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::Optimizations NeonBackend::GetOptimizations() const
{
    return Optimizations{};
}

} // namespace armnn
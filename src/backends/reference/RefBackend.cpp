//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"
#include "RefBackendId.hpp"
#include "RefWorkloadFactory.hpp"

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
    RefBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new RefBackend);
    }
};

}

const BackendId& RefBackend::GetIdStatic()
{
    static const BackendId s_Id{RefBackendId()};
    return s_Id;
}

IBackendInternal::IWorkloadFactoryPtr RefBackend::CreateWorkloadFactory() const
{
    return std::make_unique<RefWorkloadFactory>();
}

IBackendInternal::IBackendContextPtr RefBackend::CreateBackendContext(const IRuntime::CreationOptions&) const
{
    return IBackendContextPtr{};
}

IBackendInternal::Optimizations RefBackend::GetOptimizations() const
{
    return Optimizations{};
}

} // namespace armnn
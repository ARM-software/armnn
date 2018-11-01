//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"
#include "RefBackendId.hpp"
#include "RefWorkloadFactory.hpp"

#include <backendsCommon/BackendRegistry.hpp>

namespace armnn
{

namespace
{

static StaticRegistryInitializer<BackendRegistry> g_RegisterHelper
{
    BackendRegistryInstance(),
    RefBackend::GetIdStatic(),
    [](const EmptyInitializer&)
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

} // namespace armnn
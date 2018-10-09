//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"
#include <backends/BackendRegistry.hpp>
#include <boost/cast.hpp>

namespace armnn
{

namespace
{
const std::string s_Id = "CpuRef";

static BackendRegistry::Helper s_RegisterHelper{
    s_Id,
    []()
    {
        return IBackendUniquePtr(new RefBackend, &RefBackend::Destroy);
    }
};

}

const std::string& RefBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& RefBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> RefBackend::CreateWorkloadFactory() const
{
    return nullptr;
}

void RefBackend::Destroy(IBackend* backend)
{
    delete boost::polymorphic_downcast<RefBackend*>(backend);
}

} // namespace armnn
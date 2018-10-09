//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"
#include <backends/BackendRegistry.hpp>
#include <boost/cast.hpp>

namespace armnn
{

namespace
{

static const std::string s_Id = "CpuAcc";

static BackendRegistry::Helper g_RegisterHelper{
    s_Id,
    []()
    {
        return IBackendUniquePtr(new NeonBackend, &NeonBackend::Destroy);
    }
};

}

const std::string& NeonBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& NeonBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> NeonBackend::CreateWorkloadFactory() const
{
    return nullptr;
}

void NeonBackend::Destroy(IBackend* backend)
{
    delete boost::polymorphic_downcast<NeonBackend*>(backend);
}

} // namespace armnn
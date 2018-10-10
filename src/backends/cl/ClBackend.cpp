//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"
#include <backends/BackendRegistry.hpp>
#include <boost/cast.hpp>

namespace armnn
{

namespace
{
static const BackendId s_Id{"GpuAcc"};

static BackendRegistry::Helper g_RegisterHelper{
    s_Id,
    []()
    {
        return IBackendUniquePtr(new ClBackend, &ClBackend::Destroy);
    }
};

}

const BackendId& ClBackend::GetId() const
{
    return s_Id;
}

const ILayerSupport& ClBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> ClBackend::CreateWorkloadFactory() const
{
    return nullptr;
}

void ClBackend::Destroy(IBackend* backend)
{
    delete boost::polymorphic_downcast<ClBackend*>(backend);
}

} // namespace armnn
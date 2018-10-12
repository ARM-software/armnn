//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"
#include "ClWorkloadFactory.hpp"

#include <backends/BackendRegistry.hpp>

#include <boost/cast.hpp>

namespace armnn
{

namespace
{

static BackendRegistry::Helper g_RegisterHelper{
    ClBackend::GetIdStatic(),
    []()
    {
        return IBackendUniquePtr(new ClBackend, &ClBackend::Destroy);
    }
};

}

const BackendId& ClBackend::GetIdStatic()
{
    static const BackendId s_Id{"GpuAcc"};
    return s_Id;
}

const ILayerSupport& ClBackend::GetLayerSupport() const
{
    return m_LayerSupport;
}

std::unique_ptr<IWorkloadFactory> ClBackend::CreateWorkloadFactory() const
{
    return std::make_unique<ClWorkloadFactory>();
}

void ClBackend::Destroy(IBackend* backend)
{
    delete boost::polymorphic_downcast<ClBackend*>(backend);
}

} // namespace armnn
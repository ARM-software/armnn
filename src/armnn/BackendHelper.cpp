//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendHelper.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendInternal.hpp>

namespace armnn
{

std::shared_ptr<ILayerSupport> GetILayerSupportByBackendId(const armnn::BackendId& backend)
{
    BackendRegistry& backendRegistry = armnn::BackendRegistryInstance();

    if (!backendRegistry.IsBackendRegistered(backend))
    {
        return nullptr;
    }

    auto factoryFunc = backendRegistry.GetFactory(backend);
    auto backendObject = factoryFunc();
    return backendObject->GetLayerSupport();
}

}

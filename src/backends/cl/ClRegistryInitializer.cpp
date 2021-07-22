//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    ClBackend::GetIdStatic(),
    []()
    {
        // Check if we have a CustomMemoryAllocator associated with the backend
        // and if so register it with the backend.
        auto customAllocators = BackendRegistryInstance().GetAllocators();
        auto allocatorIterator = customAllocators.find(ClBackend::GetIdStatic());
        if (allocatorIterator != customAllocators.end())
        {
            return IBackendInternalUniquePtr(new ClBackend(allocatorIterator->second));
        }
        return IBackendInternalUniquePtr(new ClBackend);
    }
};

} // Anonymous namespace

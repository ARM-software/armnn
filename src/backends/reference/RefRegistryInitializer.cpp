//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    RefBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new RefBackend);
    }
};

} // Anonymous namespace

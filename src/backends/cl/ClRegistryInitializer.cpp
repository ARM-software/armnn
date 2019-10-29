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
        return IBackendInternalUniquePtr(new ClBackend);
    }
};

} // Anonymous namespace

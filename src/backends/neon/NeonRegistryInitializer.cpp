//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    NeonBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new NeonBackend);
    }
};

} // Anonymous namespace

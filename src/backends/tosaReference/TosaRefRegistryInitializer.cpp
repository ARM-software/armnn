//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefBackend.hpp"

#include <armnn/BackendRegistry.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    TosaRefBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new TosaRefBackend);
    }
};

} // Anonymous namespace

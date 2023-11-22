//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GpuFsaBackend.hpp"
#include <armnn/BackendRegistry.hpp>

namespace
{
using namespace armnn;
static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    GpuFsaBackend::GetIdStatic(),
    []()
    {
        return IBackendInternalUniquePtr(new GpuFsaBackend);
    }
};
} // Anonymous namespace
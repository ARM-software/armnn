//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
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
ARMNN_NO_DEPRECATE_WARN_BEGIN
            return IBackendInternalUniquePtr(new GpuFsaBackend);
ARMNN_NO_DEPRECATE_WARN_END
    }
};
} // Anonymous namespace
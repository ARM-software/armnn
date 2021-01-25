//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonBackend.hpp"

#include <armnn/BackendRegistry.hpp>
#include <armnn/Utils.hpp>

namespace
{

using namespace armnn;

static BackendRegistry::StaticRegistryInitializer g_RegisterHelper
{
    BackendRegistryInstance(),
    NeonBackend::GetIdStatic(),
    []()
    {
        // Check if device supports Neon.
        if (NeonDetected())
        {
            return IBackendInternalUniquePtr(new NeonBackend);
        }

        // If device does not support Neon throw exception so the Backend is not added to supportedBackends
        ARMNN_LOG(info) << "Neon support not found on device, could not register CpuAcc Backend.";
        throw armnn::BackendUnavailableException(
                "Neon support not found on device, could not register CpuAcc Backend.\n");
    }
};

} // Anonymous namespace

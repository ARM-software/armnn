//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DriverOptions.hpp"

#include <armnn/ArmNN.hpp>

#include <nnapi/IPreparedModel.h>
#include <nnapi/Result.h>
#include <nnapi/TypeUtils.h>
#include <nnapi/Types.h>
#include <nnapi/Validation.h>

using namespace android::nn;

namespace armnn_driver
{

class ArmnnDriverImpl
{
public:
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("The Shim and support library will be removed from Arm NN in 24.08", "24.08")
    static GeneralResult<SharedPreparedModel> PrepareArmnnModel(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const Model& model,
        const std::vector<SharedHandle>& modelCacheHandle,
        const std::vector<SharedHandle>& dataCacheHandle,
        const CacheToken& token,
        bool float32ToFloat16 = false,
        Priority priority = Priority::MEDIUM);

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("The Shim and support library will be removed from Arm NN in 24.08", "24.08")
    static GeneralResult<SharedPreparedModel> PrepareArmnnModelFromCache(
        const armnn::IRuntimePtr& runtime,
        const armnn::IGpuAccTunedParametersPtr& clTunedParameters,
        const DriverOptions& options,
        const std::vector<SharedHandle>& modelCacheHandle,
        const std::vector<SharedHandle>& dataCacheHandle,
        const CacheToken& token,
        bool float32ToFloat16 = false);

    static const Capabilities& GetCapabilities(const armnn::IRuntimePtr& runtime);

private:
    static bool ValidateSharedHandle(const SharedHandle& sharedHandle);
    static bool ValidateDataCacheHandle(const std::vector<SharedHandle>& dataCacheHandle, const size_t dataSize);
};

} // namespace armnn_driver
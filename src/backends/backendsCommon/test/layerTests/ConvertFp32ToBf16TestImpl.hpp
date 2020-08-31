//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <BFloat16.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

LayerTestResult<armnn::BFloat16, 4> ConvertFp32ToBf16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <Half.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

LayerTestResult<armnn::Half, 4> SimpleConvertFp32ToFp16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

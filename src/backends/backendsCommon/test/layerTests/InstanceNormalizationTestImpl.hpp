//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <Half.hpp>

#include <armnn/Types.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

LayerTestResult<float, 4> InstanceNormFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout);

LayerTestResult<armnn::Half, 4> InstanceNormFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout);

LayerTestResult<float, 4> InstanceNormFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout);

LayerTestResult<armnn::Half, 4> InstanceNormFloat16Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout);


//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <BFloat16.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

LayerTestResult<float, 4> Debug4dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 3> Debug3dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 2> Debug2dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<float, 1> Debug1dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::BFloat16, 4> Debug4dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::BFloat16, 3> Debug3dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::BFloat16, 2> Debug2dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<armnn::BFloat16, 1> Debug1dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 4> Debug4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 3> Debug3dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 2> Debug2dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<uint8_t, 1> Debug1dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 4> Debug4dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 3> Debug3dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 2> Debug2dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

LayerTestResult<int16_t, 1> Debug1dInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

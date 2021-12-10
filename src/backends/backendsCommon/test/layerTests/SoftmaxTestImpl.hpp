//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <Half.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

LayerTestResult<float, 2> SimpleSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta);

LayerTestResult<float, 2> SimpleAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis);

LayerTestResult<float, 3> Simple3dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<float, 3> Simple3dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis);

LayerTestResult<float, 4> Simple4dSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<float, 4> Simple4dAxisSoftmaxTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta,
        int axis);

LayerTestResult<uint8_t, 2> SimpleSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float beta);

LayerTestResult<uint8_t,3> Simple3dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<uint8_t,4> Simple4dSoftmaxUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<armnn::Half,2> SimpleSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<armnn::Half,3> Simple3dSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<armnn::Half,4> Simple4dSoftmaxFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<int16_t,2> SimpleSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<int16_t,3> Simple3dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<int16_t,4> Simple4dSoftmaxUint16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float beta);

LayerTestResult<float, 2> CompareSoftmaxTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    float beta);

LayerTestResult<uint8_t, 2> CompareSoftmaxUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    float beta);

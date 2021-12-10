//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <Half.hpp>

template<armnn::DataType inputDataType, armnn::DataType outputDataType,
        typename TInput=armnn::ResolveType<inputDataType>,
        typename TOutput=armnn::ResolveType<outputDataType>>
LayerTestResult<TOutput, 4> CastTest(armnn::IWorkloadFactory& workloadFactory,
                                     const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                     const armnn::ITensorHandleFactory& tensorHandleFactory,
                                     const std::vector<TInput>& inputTensor,
                                     const std::vector<TOutput>& outputTensor);


LayerTestResult<float, 4> CastInt32ToFloat2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastInt16ToFloat2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastInt8ToFloat2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastInt8AsymmToFloat2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastUInt8ToFloat2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<uint8_t, 4> CastInt8ToUInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<uint8_t, 4> CastInt8AsymmToUInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastFloat16ToFloat322dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<float, 4> CastBFloat16ToFloat322dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<armnn::Half, 4> CastFloat32ToFloat162dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<int8_t , 4> CastFloat32ToInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

LayerTestResult<uint8_t , 4> CastFloat32ToUInt82dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

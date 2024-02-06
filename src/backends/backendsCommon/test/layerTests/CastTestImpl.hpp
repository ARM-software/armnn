//
// Copyright © 2021, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/test/EndToEndTestImpl.hpp>

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

template<armnn::DataType inputDataType, armnn::DataType outputDataType, typename TInput, typename TOutput>
void CastSimpleTest(const std::vector<armnn::BackendId>& backends,
                    const std::vector<unsigned int>& shape,
                    const std::vector<TInput>& inputValues,
                    const std::vector<TOutput>& outputValues,
                    float qScale = 1.0f,
                    int32_t qOffset = 0)
{
    using namespace armnn;

    const TensorShape inputShape(static_cast<unsigned int>(shape.size()), shape.data());
    const TensorShape outputShape(static_cast<unsigned int>(shape.size()), shape.data());

    TensorInfo inputTensorInfo(inputShape, inputDataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, outputDataType, qScale, qOffset);

    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    INetworkPtr network(INetwork::Create());

    IConnectableLayer* input     = network->AddInputLayer(0, "input");
    IConnectableLayer* castLayer = network->AddCastLayer("cast");
    IConnectableLayer* output    = network->AddOutputLayer(0, "output");

    Connect(input, castLayer, inputTensorInfo, 0, 0);
    Connect(castLayer, output, outputTensorInfo, 0, 0);

    std::map<int, std::vector<TInput>> inputTensorData     = {{ 0, inputValues }};
    std::map<int, std::vector<TOutput>> expectedOutputData = {{ 0, outputValues }};

    EndToEndLayerTestImpl<inputDataType, outputDataType>(std::move(network),
                                                         inputTensorData,
                                                         expectedOutputData,
                                                         backends);
}
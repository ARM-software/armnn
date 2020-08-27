//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ResolveType.hpp>


#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

template<typename T>
LayerTestResult<T, 4> SimplePermuteTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        armnn::PermuteDescriptor descriptor,
        armnn::TensorInfo inputTensorInfo,
        armnn::TensorInfo outputTensorInfo,
        const std::vector<T>& inputData,
        const std::vector<T>& outputExpectedData)
{
    IgnoreUnused(memoryManager);
    auto input = MakeTensor<T, 4>(inputTensorInfo, inputData);

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputExpectedData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PermuteQueueDescriptor data;
    data.m_Parameters = descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePermute(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimplePermuteTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1, 2, 2, 2 };
    unsigned int outputShape[] = { 1, 2, 2, 2 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    float qScale = 0.5f;
    int32_t qOffset = 5;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    },
    qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(
    {
        1, 5, 2, 6,
        3, 7, 4, 8
    },
    qScale, qOffset);

    return SimplePermuteTestImpl<T>(workloadFactory, memoryManager, tensorHandleFactory,
                                    descriptor, inputTensorInfo,
                                    outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PermuteValueSet1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 2, 2, 3 };
    unsigned int outputShape[] = { 1, 3, 2, 2 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 2U, 3U, 1U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    float qScale = 0.5f;
    int32_t qOffset = 5;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
         1,  2,  3,
        11, 12, 13,
        21, 22, 23,
        31, 32, 33
    },
    qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(
    {
        1, 11, 21, 31,
        2, 12, 22, 32,
        3, 13, 23, 33
    },
    qScale, qOffset);

    return SimplePermuteTestImpl<T>(workloadFactory, memoryManager, tensorHandleFactory,
                                    descriptor, inputTensorInfo,
                                    outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PermuteValueSet2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 3, 2, 2 };
    unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    float qScale = 0.5f;
    int32_t qOffset = 5;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
        1, 11, 21, 31,
        2, 12, 22, 32,
        3, 13, 23, 33
    },
    qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(
    {
         1,  2,  3,
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
    },
    qScale, qOffset);

    return SimplePermuteTestImpl<T>(workloadFactory, memoryManager, tensorHandleFactory,
                                    descriptor, inputTensorInfo,
                                    outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PermuteValueSet3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 2, 3, 3 };
    unsigned int outputShape[] = { 1, 3, 2, 3 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 2U, 3U, 1U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    float qScale = 0.5f;
    int32_t qOffset = 5;
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
         1,  2,  3,
        11, 12, 13,
        21, 22, 23,
        31, 32, 33,
        41, 42, 43,
        51, 52, 53
    },
    qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(
    {
        1, 11, 21, 31, 41, 51,
        2, 12, 22, 32, 42, 52,
        3, 13, 23, 33, 43, 53
    },
    qScale, qOffset);

    return SimplePermuteTestImpl<T>(workloadFactory, memoryManager, tensorHandleFactory,
                                    descriptor, inputTensorInfo,
                                    outputTensorInfo, input, outputExpected);
}

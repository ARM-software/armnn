//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ChannelShuffleTestImpl.hpp"

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

namespace
{

template<typename T, size_t NumDims>
LayerTestResult<T, NumDims> ChannelShuffleTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        armnn::ChannelShuffleDescriptor descriptor,
        armnn::TensorInfo inputTensorInfo,
        armnn::TensorInfo outputTensorInfo,
        const std::vector<T>& inputData,
        const std::vector<T>& outputExpectedData)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ChannelShuffleQueueDescriptor data;
    data.m_Parameters = descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::ChannelShuffle, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(actualOutput,
                                       outputExpectedData,
                                       outputHandle->GetShape(),
                                       outputTensorInfo.GetShape());
}
} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleChannelShuffleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1,9,1,1 };
    unsigned int outputShape[] = { 1,9,1,1 };

    armnn::ChannelShuffleDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumGroups = 3;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
            {
                0.0f, 1.0f, 2.0f,   3.0f, 4.0f, 5.0f,   6.0f, 7.0f, 8.0f
            },
            inputTensorInfo);
    auto outputExpected = ConvertToDataType<ArmnnType>(
            {
                0.0f, 3.0f, 6.0f, 1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f
            },
            outputTensorInfo);

    return ChannelShuffleTestImpl<T, 4>(
                workloadFactory,
                memoryManager,
                tensorHandleFactory,
                descriptor,
                inputTensorInfo,
                outputTensorInfo,
                input,
                outputExpected);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ChannelShuffle2DTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 3, 12 };
    unsigned int outputShape[] = { 3, 12 };

    armnn::ChannelShuffleDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumGroups = 3;

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
            {
                0, 1, 2, 3,       4, 5, 6, 7,       8, 9, 10, 11,
               12, 13, 14, 15,   16, 17, 18, 19,   20, 21, 22, 23,
               24, 25, 26, 27,   28, 29, 30, 31,   32, 33, 34, 35
            },
            inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
            {
                0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
                12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23,
                24, 28, 32, 25, 29, 33, 26, 30, 34, 27, 31, 35
            },
            outputTensorInfo);

    return ChannelShuffleTestImpl<T, 2>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            descriptor,
            inputTensorInfo,
            outputTensorInfo,
            input,
            outputExpected);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ChannelShuffle4DTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 9, 1, 2 };
    unsigned int outputShape[] = { 2, 9, 1, 2 };

    armnn::ChannelShuffleDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumGroups = 3;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
            {
                    0.0f, 1.0f,
                    2.0f, 3.0f,
                    4.0f, 5.0f,

                    6.0f, 7.0f,
                    8.0f, 9.0f,
                    10.0f, 11.0f,

                    12.0f, 13.0f,
                    14.0f, 15.0f,
                    16.0f, 17.0f,

                    18.0f, 19.0f,
                    20.0f, 21.0f,
                    22.0f, 23.0f,

                    24.0f, 25.0f,
                    26.0f, 27.0f,
                    28.0f, 29.0f,

                    30.0f, 31.0f,
                    32.0f, 33.0f,
                    34.0f, 35.0f
            },
            inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
            {
                    0.0f, 1.0f,
                    6.0f, 7.0f,
                    12.0f, 13.0f,
                    2.0f, 3.0f,
                    8.0f, 9.0f,
                    14.0f, 15.0f,
                    4.0f, 5.0f,
                    10.0f, 11.0f,
                    16.0f, 17.0f,

                    18.0f, 19.0f,
                    24.0f, 25.0f,
                    30.0f, 31.0f,
                    20.0f, 21.0f,
                    26.0f, 27.0f,
                    32.0f, 33.0f,
                    22.0f, 23.0f,
                    28.0f, 29.0f,
                    34.0f, 35.0f
            },
            outputTensorInfo);

    return ChannelShuffleTestImpl<T, 4>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            descriptor,
            inputTensorInfo,
            outputTensorInfo,
            input,
            outputExpected);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleChannelShuffleTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
SimpleChannelShuffleTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleChannelShuffleTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ChannelShuffle2DTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
ChannelShuffle2DTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
ChannelShuffle2DTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ChannelShuffle4DTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
ChannelShuffle4DTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
ChannelShuffle4DTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);
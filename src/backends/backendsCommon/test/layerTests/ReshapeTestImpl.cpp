//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReshapeTestImpl.hpp"

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T, size_t NumDims>
LayerTestResult<T, NumDims> SimpleReshapeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::TensorInfo inputTensorInfo,
    armnn::TensorInfo outputTensorInfo,
    const std::vector<T>& inputData,
    const std::vector<T>& outputExpectedData)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ReshapeQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Reshape, data, info);

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
LayerTestResult<T, 4> SimpleReshapeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 3, 3 };
    unsigned int outputShape[] = { 2, 2, 9, 1 };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f,

            9.0f, 10.0f, 11.0f,
            12.0f, 13.0f, 14.0f,
            15.0f, 16.0f, 17.0f,

            18.0f, 19.0f, 20.0f,
            21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f,

            27.0f, 28.0f, 29.0f,
            30.0f, 31.0f, 32.0f,
            33.0f, 34.0f, 35.0f,
        },
        inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,

            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,

            18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,

            27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
        },
        outputTensorInfo);

    return SimpleReshapeTestImpl<T, 4>(
        workloadFactory, memoryManager, tensorHandleFactory, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 5> Reshape5dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 8, 1, 1 };
    unsigned int outputShape[] = { 2, 2, 2, 2, 2 };

    inputTensorInfo = armnn::TensorInfo(5, inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(5, outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(1.0f);

    auto input = ConvertToDataType<ArmnnType>(
        {
            0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,

            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
            24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f,
        },
        inputTensorInfo);

    auto outputExpected = ConvertToDataType<ArmnnType>(
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
        },
        outputTensorInfo);

    return SimpleReshapeTestImpl<T, 5>(
        workloadFactory, memoryManager, tensorHandleFactory, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

LayerTestResult<uint8_t, 2> ReshapeBooleanTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1, 4 };
    unsigned int outputShape[] = { 2, 2 };

    inputTensorInfo = armnn::TensorInfo(2, inputShape, armnn::DataType::Boolean);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Boolean);
    outputTensorInfo.SetQuantizationScale(1.0f);

    const std::vector<uint8_t> input = { true, false, false, true };

    const std::vector<uint8_t> outputExpected = { true, false, false, true };

    return SimpleReshapeTestImpl<uint8_t, 2>(
        workloadFactory, memoryManager, tensorHandleFactory, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleReshapeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleReshapeTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
SimpleReshapeTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
SimpleReshapeTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 5>
Reshape5dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 5>
Reshape5dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 5>
Reshape5dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 5>
Reshape5dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

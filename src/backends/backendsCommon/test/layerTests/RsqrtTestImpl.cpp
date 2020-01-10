//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReshapeTestImpl.hpp"

#include <armnn/ArmNN.hpp>

#include <backendsCommon/test/DataTypeUtils.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Rsqrt2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo inputTensorInfo,
    const armnn::TensorInfo outputTensorInfo,
    const std::vector<float>& inputValues,
    const std::vector<float>& expectedOutputValues)
{
    boost::ignore_unused(memoryManager);
    auto inputTensor = MakeTensor<T, 2>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputValues,inputTensorInfo));

    LayerTestResult<T, 2> result(outputTensorInfo);

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                             ConvertToDataType<ArmnnType>(expectedOutputValues,outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::RsqrtQueueDescriptor descriptor;

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateRsqrt(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Rsqrt2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 2, 2 };
    const armnn::TensorShape outputShape{ 2, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        1.f, 4.f,
        16.f, 25.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f,
        0.25f, 0.2f
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Rsqrt3dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    boost::ignore_unused(memoryManager);
    const armnn::TensorShape inputShape{ 3, 1, 2 };
    const armnn::TensorShape outputShape{ 3, 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        1.f, 4.f, 16.f,
        25.f, 64.f, 100.f
    };

    std::vector<float> expectedOutputValues
    {
        1.f, 0.5f, 0.25f,
        0.2f, 0.125f, 0.1f
    };

    auto inputTensor = MakeTensor<T, 3>(inputTensorInfo, ConvertToDataType<ArmnnType>(inputValues,inputTensorInfo));

    LayerTestResult<T, 3> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo,
                                             ConvertToDataType<ArmnnType>(expectedOutputValues,outputTensorInfo));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::RsqrtQueueDescriptor descriptor;

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateRsqrt(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtZeroTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 1, 2 };
    const armnn::TensorShape outputShape{ 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);

    std::vector<float> inputValues
    {
        0.f, -0.f
    };

    std::vector<float> expectedOutputValues
    {
        INFINITY, -INFINITY
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> RsqrtNegativeTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    const armnn::TensorShape inputShape{ 1, 2 };
    const armnn::TensorShape outputShape{ 1, 2 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(0);

    armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);
    outputTensorInfo.SetQuantizationOffset(0);

    std::vector<float> inputValues
    {
        -25.f, -16.f
    };

    std::vector<float> expectedOutputValues
    {
        -NAN, -NAN
    };

    return Rsqrt2dTestCommon<ArmnnType>(workloadFactory, memoryManager,
                                inputTensorInfo, outputTensorInfo,
                                inputValues, expectedOutputValues);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Rsqrt2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Rsqrt2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Rsqrt2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Rsqrt2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Rsqrt3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Rsqrt3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Rsqrt3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Rsqrt3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
RsqrtZeroTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
RsqrtNegativeTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager);

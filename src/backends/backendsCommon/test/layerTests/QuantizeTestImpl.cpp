//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizeTestImpl.hpp"

#include <ResolveType.hpp>


#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t Dim>
LayerTestResult<T, Dim> QuantizeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorInfo& inputTensorInfo,
    const armnn::TensorInfo& outputTensorInfo,
    const std::vector<float>& inputData,
    const std::vector<T>& expectedOutputData,
    armnn::QuantizeQueueDescriptor descriptor)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Quantize,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, Dim>(actualOutput,
                                   expectedOutputData,
                                   outputHandle->GetShape(),
                                   outputTensorInfo.GetShape());
}

template <armnn::DataType ArmnnOutputType, typename T = armnn::ResolveType<ArmnnOutputType>>
LayerTestResult<T, 4> QuantizeSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::QuantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 2, 2, 3}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo({1, 2, 2, 3}, ArmnnOutputType, 0.5f, 1);

    std::vector<float> inputData = std::vector<float>(
    {
        1.0f,   2.0f,  3.0f,
        4.0f,   5.0f,  6.0f,
        7.0f,   8.0f,  9.0f,
        10.0f, 11.0f, 12.0f,
    });

    std::vector<T> expectedOutputData = std::vector<T>(
    {
         3,  5,  7,
         9, 11, 13,
        15, 17, 19,
        21, 23, 25,
    });

    return QuantizeTestImpl<T, 4>(workloadFactory,
                                  memoryManager,
                                  tensorHandleFactory,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  desc);
}

template <armnn::DataType ArmnnOutputType, typename T = armnn::ResolveType<ArmnnOutputType>>
LayerTestResult<T, 4> QuantizeClampTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::QuantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 1, 2, 1}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo({1, 1, 2, 1}, ArmnnOutputType, 0.0001f, 0);

    const T max = std::numeric_limits<T>::max();
    const T min = std::numeric_limits<T>::lowest();

    std::vector<float> inputData = std::vector<float>(
    {
        -100.0f, 100.0f
    });

    std::vector<T> expectedOutputData = std::vector<T>(
    {
        min, max
    });

    return QuantizeTestImpl<T, 4>(workloadFactory,
                                  memoryManager,
                                  tensorHandleFactory,
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  desc);
}

} // anonymous namespace

LayerTestResult<uint8_t, 4> QuantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return QuantizeSimpleTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> QuantizeClampUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return QuantizeClampTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int8_t, 4> QuantizeClampAsymmInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return QuantizeClampTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int8_t, 4> QuantizeClampInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return QuantizeClampTest<armnn::DataType::QSymmS8>(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> QuantizeClampInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return QuantizeClampTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory);
}

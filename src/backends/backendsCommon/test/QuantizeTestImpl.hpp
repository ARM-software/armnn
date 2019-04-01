//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadTestUtils.hpp"

#include <test/TensorHelpers.hpp>

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>


namespace
{

template<typename T, std::size_t Dim>
LayerTestResult<T, Dim> QuantizeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& inputTensorInfo,
    const armnn::TensorInfo& outputTensorInfo,
    const std::vector<float>& inputData,
    const std::vector<T>& expectedOutputData,
    armnn::QuantizeQueueDescriptor descriptor)
{
    boost::multi_array<float, Dim> input = MakeTensor<float, Dim>(inputTensorInfo, inputData);

    LayerTestResult<T, Dim> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, Dim>(outputTensorInfo, expectedOutputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateQuantize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.data(), outputHandle.get());

    return ret;
}

template <armnn::DataType ArmnnOutputType, typename T = armnn::ResolveType<ArmnnOutputType>>
LayerTestResult<T, 4> QuantizeSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  desc);
}

template <armnn::DataType ArmnnOutputType, typename T = armnn::ResolveType<ArmnnOutputType>>
LayerTestResult<T, 4> QuantizeClampTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
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
                                  inputTensorInfo,
                                  outputTensorInfo,
                                  inputData,
                                  expectedOutputData,
                                  desc);
}

} // anonymous namespace

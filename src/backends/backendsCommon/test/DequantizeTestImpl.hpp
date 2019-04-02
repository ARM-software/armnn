//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "WorkloadTestUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t Dim>
LayerTestResult<float, Dim> DequantizeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& inputTensorInfo,
    const armnn::TensorInfo& outputTensorInfo,
    const std::vector<T>& inputData,
    const std::vector<float>& expectedOutputData,
    armnn::DequantizeQueueDescriptor descriptor)
{
    boost::multi_array<T, Dim> input = MakeTensor<T, Dim>(inputTensorInfo, inputData);

    LayerTestResult<float, Dim> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<float, Dim>(outputTensorInfo, expectedOutputData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateDequantize(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.data(), outputHandle.get());

    return ret;
}

template <armnn::DataType ArmnnInputType>
LayerTestResult<float, 4> DequantizeSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using T = armnn::ResolveType<ArmnnInputType>;

    armnn::DequantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 2, 2, 3}, ArmnnInputType, 0.5f, 0);
    const armnn::TensorInfo outputTensorInfo({1, 2, 2, 3}, armnn::DataType::Float32);

    std::vector<T> inputData = std::vector<T>(
    {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24,
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f,   2.0f,  3.0f,
        4.0f,   5.0f,  6.0f,
        7.0f,   8.0f,  9.0f,
        10.0f, 11.0f, 12.0f,
    });

    return DequantizeTestImpl<T, 4>(workloadFactory,
                                    memoryManager,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputData,
                                    expectedOutputData,
                                    desc);
}

template <armnn::DataType ArmnnInputType>
LayerTestResult<float, 4> DequantizeOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using T = armnn::ResolveType<ArmnnInputType>;

    armnn::DequantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 2, 2, 3}, ArmnnInputType, 0.5f, 1);
    const armnn::TensorInfo outputTensorInfo({1, 2, 2, 3}, armnn::DataType::Float32);

    std::vector<T> inputData = std::vector<T>(
    {
         3,  5,  7,
         9, 11, 13,
        15, 17, 19,
        21, 23, 25,
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f,   2.0f,  3.0f,
        4.0f,   5.0f,  6.0f,
        7.0f,   8.0f,  9.0f,
        10.0f, 11.0f, 12.0f,
    });

    return DequantizeTestImpl<T, 4>(workloadFactory,
                                    memoryManager,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputData,
                                    expectedOutputData,
                                    desc);
}

} // anonymous namespace

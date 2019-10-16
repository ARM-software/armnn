//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <armnn/ArmNN.hpp>

#include <ResolveType.hpp>

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

template <std::size_t NumDims,
          armnn::DataType ArmnnInType,
          typename InType = armnn::ResolveType<ArmnnInType>>
LayerTestResult<uint8_t, NumDims> ComparisonTestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    const armnn::ComparisonDescriptor& descriptor,
    const unsigned int shape0[NumDims],
    std::vector<InType> values0,
    float quantScale0,
    int quantOffset0,
    const unsigned int shape1[NumDims],
    std::vector<InType> values1,
    float quantScale1,
    int quantOffset1,
    const unsigned int outShape[NumDims],
    std::vector<uint8_t> outValues,
    float outQuantScale,
    int outQuantOffset)
{
    armnn::TensorInfo inputTensorInfo0{NumDims, shape0, ArmnnInType};
    armnn::TensorInfo inputTensorInfo1{NumDims, shape1, ArmnnInType};
    armnn::TensorInfo outputTensorInfo{NumDims, outShape, armnn::DataType::Boolean};

    auto input0 = MakeTensor<InType, NumDims>(inputTensorInfo0, values0);
    auto input1 = MakeTensor<InType, NumDims>(inputTensorInfo1, values1);

    inputTensorInfo0.SetQuantizationScale(quantScale0);
    inputTensorInfo0.SetQuantizationOffset(quantOffset0);

    inputTensorInfo1.SetQuantizationScale(quantScale1);
    inputTensorInfo1.SetQuantizationOffset(quantOffset1);

    outputTensorInfo.SetQuantizationScale(outQuantScale);
    outputTensorInfo.SetQuantizationOffset(outQuantOffset);

    LayerTestResult<uint8_t, NumDims> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle0 = workloadFactory.CreateTensorHandle(inputTensorInfo0);
    std::unique_ptr<armnn::ITensorHandle> inputHandle1 = workloadFactory.CreateTensorHandle(inputTensorInfo1);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ComparisonQueueDescriptor qDescriptor;
    qDescriptor.m_Parameters = descriptor;

    armnn::WorkloadInfo info;
    AddInputToWorkload(qDescriptor, info, inputTensorInfo0, inputHandle0.get());
    AddInputToWorkload(qDescriptor, info, inputTensorInfo1, inputHandle1.get());
    AddOutputToWorkload(qDescriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateComparison(qDescriptor, info);

    inputHandle0->Allocate();
    inputHandle1->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle0.get(), input0.origin());
    CopyDataToITensorHandle(inputHandle1.get(), input1.origin());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.origin(), outputHandle.get());

    ret.outputExpected = MakeTensor<uint8_t, NumDims>(outputTensorInfo, outValues);
    ret.compareBoolean = true;

    return ret;
}

template <std::size_t NumDims,
          armnn::DataType ArmnnInType,
          typename InType = armnn::ResolveType<ArmnnInType>>
LayerTestResult<uint8_t, NumDims> ComparisonTestImpl(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    const armnn::ComparisonDescriptor& descriptor,
    const unsigned int shape0[NumDims],
    std::vector<InType> values0,
    const unsigned int shape1[NumDims],
    std::vector<InType> values1,
    const unsigned int outShape[NumDims],
    std::vector<uint8_t> outValues,
    float qScale = 10.f,
    int qOffset = 0)
{
    return ComparisonTestImpl<NumDims, ArmnnInType>(
        workloadFactory,
        memoryManager,
        descriptor,
        shape0,
        values0,
        qScale,
        qOffset,
        shape1,
        values1,
        qScale,
        qOffset,
        outShape,
        outValues,
        qScale,
        qOffset);
}

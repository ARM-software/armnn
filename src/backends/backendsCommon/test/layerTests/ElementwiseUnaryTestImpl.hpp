//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <armnn/ArmNN.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/DataTypeUtils.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

#include <memory>

std::unique_ptr<armnn::IWorkload> CreateWorkload(
    const armnn::IWorkloadFactory& workloadFactory,
    const armnn::WorkloadInfo& info,
    const armnn::ElementwiseUnaryQueueDescriptor& descriptor);

template <std::size_t NumDims,
          armnn::DataType ArmnnType,
          typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, NumDims> ElementwiseUnaryTestHelper(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    armnn::UnaryOperation op,
    const unsigned int shape[NumDims],
    std::vector<float> values,
    float quantScale,
    int quantOffset,
    const unsigned int outShape[NumDims],
    std::vector<float> outValues,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float outQuantScale,
    int outQuantOffset)
{
    armnn::TensorInfo inputTensorInfo{NumDims, shape, ArmnnType};
    armnn::TensorInfo outputTensorInfo{NumDims, outShape, ArmnnType};

    inputTensorInfo.SetQuantizationScale(quantScale);
    inputTensorInfo.SetQuantizationOffset(quantOffset);

    outputTensorInfo.SetQuantizationScale(outQuantScale);
    outputTensorInfo.SetQuantizationOffset(outQuantOffset);

    auto input = MakeTensor<T, NumDims>(inputTensorInfo, ConvertToDataType<ArmnnType>(values, inputTensorInfo));

    LayerTestResult<T, NumDims> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ElementwiseUnaryDescriptor desc(op);
    armnn::ElementwiseUnaryQueueDescriptor qDesc;
    qDesc.m_Parameters = desc;
    armnn::WorkloadInfo info;
    AddInputToWorkload(qDesc, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(qDesc, info, outputTensorInfo, outputHandle.get());
    auto workload = CreateWorkload(workloadFactory, info, qDesc);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.origin(), outputHandle.get());

    ret.outputExpected = MakeTensor<T, NumDims>(outputTensorInfo, ConvertToDataType<ArmnnType>(outValues,
        inputTensorInfo));
    return ret;
}

template <std::size_t NumDims,
          armnn::DataType ArmnnType,
          typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, NumDims> ElementwiseUnaryTestHelper(
    armnn::IWorkloadFactory & workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr & memoryManager,
    armnn::UnaryOperation op,
    const unsigned int shape[NumDims],
    std::vector<float> values,
    const unsigned int outShape[NumDims],
    std::vector<float> outValues,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float quantScale = 1.0f,
    int quantOffset = 0)
{
    return ElementwiseUnaryTestHelper<NumDims, ArmnnType>(
        workloadFactory,
        memoryManager,
        op,
        shape,
        values,
        quantScale,
        quantOffset,
        outShape,
        outValues,
        tensorHandleFactory,
        quantScale,
        quantOffset);
}
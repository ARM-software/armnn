//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <armnn/ArmNN.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

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

    std::vector<T> input = ConvertToDataType<ArmnnType>(values, inputTensorInfo);
    std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(outValues, inputTensorInfo);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

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

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(actualOutput,
                                       expectedOutput,
                                       outputHandle->GetShape(),
                                       outputTensorInfo.GetShape());

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
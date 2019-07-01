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

template<typename T>
LayerTestResult<T, 4> SpaceToDepthTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::TensorInfo& inputTensorInfo,
    armnn::TensorInfo& outputTensorInfo,
    std::vector<float>& inputData,
    std::vector<float>& outputExpectedData,
    armnn::SpaceToDepthQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int32_t qOffset = 0)
{
    const armnn::PermutationVector NHWCToNCHW = {0, 2, 3, 1};

    if (descriptor.m_Parameters.m_DataLayout == armnn::DataLayout::NCHW)
    {
        inputTensorInfo = armnnUtils::Permuted(inputTensorInfo, NHWCToNCHW);
        outputTensorInfo = armnnUtils::Permuted(outputTensorInfo, NHWCToNCHW);

        std::vector<float> inputTmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NHWCToNCHW,
            inputData.data(), inputTmp.data(), sizeof(float));
        inputData = inputTmp;

        std::vector<float> outputTmp(outputExpectedData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NHWCToNCHW,
            outputExpectedData.data(), outputTmp.data(), sizeof(float));
        outputExpectedData = outputTmp;
    }

    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputTensorInfo, QuantizedVector<T>(qScale, qOffset, inputData));

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, QuantizedVector<T>(qScale, qOffset, outputExpectedData));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSpaceToDepth(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SpaceToDepthSimpleTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout = armnn::DataLayout::NHWC)
{
    unsigned int inputShape[] = {1, 2, 2, 1};
    unsigned int outputShape[] = {1, 1, 1, 4};

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    armnn::SpaceToDepthQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize = 2;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    return SpaceToDepthTestImpl<T>(
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SpaceToDepthSimpleTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout = armnn::DataLayout::NHWC)
{
    unsigned int inputShape[] = {1, 2, 2, 2};
    unsigned int outputShape[] = {1, 1, 1, 8};

    std::vector<float> input = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    armnn::SpaceToDepthQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize = 2;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    return SpaceToDepthTestImpl<T>(
        workloadFactory, memoryManager, inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

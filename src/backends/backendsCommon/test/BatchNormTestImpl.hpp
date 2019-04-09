//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>
#include "WorkloadTestUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/test/QuantizeHelper.hpp>

#include <test/TensorHelpers.hpp>

#include <DataLayoutIndexed.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BatchNormTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorShape& inputOutputTensorShape,
    const std::vector<float>& inputValues,
    const std::vector<float>& expectedOutputValues,
    float qScale,
    int32_t qOffset,
    armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo(inputOutputTensorShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputTensorShape, ArmnnType);

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);

    armnn::TensorInfo tensorInfo({ inputOutputTensorShape[dataLayoutIndexed.GetChannelsIndex()] },
                                 ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        tensorInfo.SetQuantizationScale(qScale);
        tensorInfo.SetQuantizationOffset(qOffset);
    }

    auto inputTensor = MakeTensor<T, 4>(inputTensorInfo,
                                        QuantizedVector<T>(qScale, qOffset, inputValues));

    // These values are per-channel of the input.
    auto mean     = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {3, -2}));
    auto variance = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {4,  9}));
    auto beta     = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {3,  2}));
    auto gamma    = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {2,  1}));

    LayerTestResult<T, 4> result(outputTensorInfo);

    result.outputExpected = MakeTensor<T, 4>(inputTensorInfo,
                                             QuantizedVector<T>(qScale, qOffset, expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ScopedCpuTensorHandle meanTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle varianceTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle betaTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle gammaTensor(tensorInfo);

    armnn::BatchNormalizationQueueDescriptor descriptor;
    descriptor.m_Mean                    = &meanTensor;
    descriptor.m_Variance                = &varianceTensor;
    descriptor.m_Beta                    = &betaTensor;
    descriptor.m_Gamma                   = &gammaTensor;
    descriptor.m_Parameters.m_Eps        = 0.0f;
    descriptor.m_Parameters.m_DataLayout = dataLayout;
    armnn::WorkloadInfo info;

    AllocateAndCopyDataToITensorHandle(&meanTensor, &mean[0]);
    AllocateAndCopyDataToITensorHandle(&varianceTensor, &variance[0]);
    AllocateAndCopyDataToITensorHandle(&betaTensor, &beta[0]);
    AllocateAndCopyDataToITensorHandle(&gammaTensor, &gamma[0]);

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchNormalization(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T,4> BatchNormTestNhwcImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float qScale,
    int32_t qOffset)
{
    const unsigned int width    = 2;
    const unsigned int height   = 3;
    const unsigned int channels = 2;
    const unsigned int num      = 1;

    armnn::TensorInfo inputTensorInfo({num, height, width, channels}, ArmnnType);
    armnn::TensorInfo outputTensorInfo({num, height, width, channels}, ArmnnType);
    armnn::TensorInfo tensorInfo({channels}, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
        tensorInfo.SetQuantizationScale(qScale);
        tensorInfo.SetQuantizationOffset(qOffset);
    }

    auto input = MakeTensor<T, 4>(inputTensorInfo,
        QuantizedVector<T>(qScale, qOffset,
        {
            1.f, 1.f, 4.f, 1.f,
            4.f, 4.f, 2.f, 1.f,
            1.f, -2.f, 6.f, 4.f
        }));
    // These values are per-channel of the input.
    auto mean     = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {3, -2}));
    auto variance = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {4, 9}));
    auto beta     = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {3, 2}));
    auto gamma    = MakeTensor<T, 1>(tensorInfo, QuantizedVector<T>(qScale, qOffset, {2, 1}));
    LayerTestResult<T,4> ret(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::BatchNormalizationQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle meanTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle varianceTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle betaTensor(tensorInfo);
    armnn::ScopedCpuTensorHandle gammaTensor(tensorInfo);

    AllocateAndCopyDataToITensorHandle(&meanTensor, &mean[0]);
    AllocateAndCopyDataToITensorHandle(&varianceTensor, &variance[0]);
    AllocateAndCopyDataToITensorHandle(&betaTensor, &beta[0]);
    AllocateAndCopyDataToITensorHandle(&gammaTensor, &gamma[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Mean             = &meanTensor;
    data.m_Variance         = &varianceTensor;
    data.m_Beta             = &betaTensor;
    data.m_Gamma            = &gammaTensor;
    data.m_Parameters.m_Eps = 0.0f;
    data.m_Parameters.m_DataLayout = armnn::DataLayout::NHWC;

    // For each channel:
    // substract mean, divide by standard deviation (with an epsilon to avoid div by 0),
    // multiply by gamma and add beta
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset,
        {
            1.f, 3.f, 4.f, 3.f,
            4.f, 4.f, 2.f, 3.f,
            1.f, 2.f, 6.f, 4.f
        }));

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchNormalization(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

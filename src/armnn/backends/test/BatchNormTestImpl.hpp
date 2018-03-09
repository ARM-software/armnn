//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <backends/WorkloadInfo.hpp>

#include "test/TensorHelpers.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/WorkloadFactory.hpp"

#include "backends/test/QuantizeHelper.hpp"


template<typename T>
LayerTestResult<T,4> BatchNormTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                   float qScale,
                                   int32_t qOffset)
{
    const unsigned int width    = 2;
    const unsigned int height   = 3;
    const unsigned int channels = 2;
    const unsigned int num      = 1;

    armnn::TensorInfo inputTensorInfo({num, channels, height, width}, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo({num, channels, height, width}, armnn::GetDataType<T>());
    armnn::TensorInfo tensorInfo({channels}, armnn::GetDataType<T>());

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
            1.f, 4.f,
            4.f, 2.f,
            1.f, 6.f,

            1.f, 1.f,
            4.f, 1.f,
            -2.f, 4.f
        }));
    // these values are per-channel of the input
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

    // for each channel:
    // substract mean, divide by standard deviation (with an epsilon to avoid div by 0)
    // multiply by gamma and add beta
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset,
        {
            1.f, 4.f,
            4.f, 2.f,
            1.f, 6.f,

            3.f, 3.f,
            4.f, 3.f,
            2.f, 4.f
        }));

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateBatchNormalization(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}
//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>

#include <test/TensorHelpers.hpp>

#include <backends/CpuTensorHandle.hpp>
#include <backends/WorkloadFactory.hpp>

#include <backends/test/QuantizeHelper.hpp>

template<typename T>
LayerTestResult<T, 4> BatchNormTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                        const armnn::TensorShape& inputOutputTensorShape,
                                        const std::vector<float>& inputValues,
                                        const std::vector<float>& expectedOutputValues,
                                        float qScale,
                                        int32_t qOffset,
                                        armnn::DataLayout dataLayout)
{
    armnn::TensorInfo inputTensorInfo(inputOutputTensorShape, armnn::GetDataType<T>());
    armnn::TensorInfo outputTensorInfo(inputOutputTensorShape, armnn::GetDataType<T>());

    armnn::DataLayoutIndexed dataLayoutIndexed(dataLayout);

    armnn::TensorInfo tensorInfo({ inputOutputTensorShape[dataLayoutIndexed.GetChannelsIndex()] },
                                 armnn::GetDataType<T>());

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

    workloadFactory.Finalize();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "QuantizeHelper.hpp"
#include "WorkloadTestUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <test/TensorHelpers.hpp>

#include <algorithm>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> SimpleSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta)
{
    using std::exp;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 4 };

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    float qScale = 1.f / 256.f;
    int qOffset = 0;
    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);

    outputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    LayerTestResult<T, 2> ret(outputTensorInfo);

    // Each row is independently softmax'd.
    auto input = MakeTensor<T, 2>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(qScale, 0, {
            0.f, 1.f, 0.f, 0.f,
            .5f, 0.f, 0.f, 0.f,
        })));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SoftmaxQueueDescriptor data;
    data.m_Parameters.m_Beta = beta;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSoftmax(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0]);

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&ret.output[0][0], outputHandle.get());

    float x0[4] = { exp((0.f - 1.0f) * beta), exp((1.0f - 1.0f) * beta),
        exp((0.0f - 1.0f) * beta), exp((0.0f - 1.0f) * beta) };
    float sum0 = x0[0] + x0[1] + x0[2] + x0[3];
    float x1[4] = { exp((0.5f - 0.5f) * beta), exp((0.0f - 0.5f) * beta),
        exp((0.0f - 0.5f) * beta), exp((0.0f - 0.5f) * beta) };
    float sum1 = x1[0] + x1[1] + x1[2] + x1[3];

    ret.outputExpected = MakeTensor<T, 2>(outputTensorInfo, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, {
        x0[0] / sum0, x0[1] / sum0, x0[2] / sum0, x0[3] / sum0,
        x1[0] / sum1, x1[1] / sum1, x1[2] / sum1, x1[3] / sum1
        })));

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> CompareSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta)
{

    const int batchSize = 20;
    const int channels = 30;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { batchSize, channels };

    inputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, inputShape, ArmnnType);
    float qScale = 1.f / 256.f;
    int qOffset = 0;
    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);


    LayerTestResult<T, 2> ret(outputTensorInfo);
    auto input = MakeRandomTensor<T, 2>(inputTensorInfo, 0xF00D, 0.0f, 1.0f);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SoftmaxQueueDescriptor data;
    data.m_Parameters.m_Beta = beta;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refWorkloadFactory.CreateTensorHandle(outputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refWorkloadFactory.CreateTensorHandle(inputTensorInfo);


    armnn::SoftmaxQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSoftmax(data, info);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateSoftmax(refData, refInfo);

    outputHandleRef->Allocate();
    inputHandleRef->Allocate();

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0]);
    CopyDataToITensorHandle(inputHandleRef.get(), &input[0][0]);

    ExecuteWorkload(*workload, memoryManager);

    workloadRef->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0], outputHandle.get());
    CopyDataFromITensorHandle(&ret.outputExpected[0][0], outputHandleRef.get());

    return ret;
}

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

template<armnn::DataType ArmnnType, std::size_t n, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, n> SimpleSoftmaxBaseTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = 1)
{
    using std::exp;

    const float qScale = 1.f / 256.f;
    const int qOffset = 0;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    inputTensorInfo = armnn::TensorInfo(inputShape, ArmnnType);
    inputTensorInfo.SetQuantizationScale(qScale);
    inputTensorInfo.SetQuantizationOffset(qOffset);

    outputTensorInfo = armnn::TensorInfo(inputShape, ArmnnType);
    outputTensorInfo.SetQuantizationScale(qScale);
    outputTensorInfo.SetQuantizationOffset(qOffset);

    LayerTestResult<T, n> ret(outputTensorInfo);

    // Each row is independently softmax'd.
    auto input = MakeTensor<T, n>(inputTensorInfo, std::vector<T>(
        QuantizedVector<T>(qScale, qOffset, inputData)));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::SoftmaxQueueDescriptor data;
    data.m_Parameters.m_Beta = beta;
    data.m_Parameters.m_Axis = axis;

    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateSoftmax(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    BOOST_ASSERT(workload);

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(ret.output.origin(), outputHandle.get());

    std::vector<T> expectedOutput = std::vector<T>(
            QuantizedVector<T>(qScale, qOffset, outputData));
    ret.outputExpected = MakeTensor<T, n>(outputTensorInfo, expectedOutput);

    return ret;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> SimpleSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta)
{
    using std::exp;
    const armnn::TensorShape inputShape{ 2, 4 };

    float x0[4] = { exp((0.f - 1.0f) * beta), exp((1.0f - 1.0f) * beta),
                    exp((0.0f - 1.0f) * beta), exp((0.0f - 1.0f) * beta) };
    float sum0 = x0[0] + x0[1] + x0[2] + x0[3];
    float x1[4] = { exp((0.5f - 0.5f) * beta), exp((0.0f - 0.5f) * beta),
                    exp((0.0f - 0.5f) * beta), exp((0.0f - 0.5f) * beta) };
    float sum1 = x1[0] + x1[1] + x1[2] + x1[3];

    const std::vector<float> outputData = { x0[0] / sum0, x0[1] / sum0, x0[2] / sum0, x0[3] / sum0,
                                            x1[0] / sum1, x1[1] / sum1, x1[2] / sum1, x1[3] / sum1 };

    const std::vector<float> inputData =
            {
                0.f, 1.f, 0.f, 0.f,
                .5f, 0.f, 0.f, 0.f,
            };

    return SimpleSoftmaxBaseTestImpl<ArmnnType, 2>(workloadFactory, memoryManager, beta,
                                                   inputShape, outputData, inputData);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> SimpleSoftmaxTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        float beta,
        int axis)
{
    armnn::TensorShape inputShape;
    std::vector<float> inputData;
    std::vector<float> outputData;
    switch (axis)
    {
    case -2:
    case 0:
        {
        inputShape = {5, 2};

        inputData =
                {
                        17.0f, -1.0f, 16.0f, -2.0f, 15.0f, -3.0f, 14.0f, -4.0f, 1.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.643914213228014f, 0.236882800924671f, 0.236882800924671f,
                        0.087144312427294f,
                        0.087144312427294f, 0.032058600957022f, 0.032058600957022f, 7.246299848982885e-08f,
                        7.246299848982885e-08f
                };
        break;
        }
    case -1:
    case 1:
        {
        inputShape = {2, 5};

        inputData =
                {
                        17.0f, 16.0f, 15.0f, 14.0f, 1.0f, -1.0f, -2.0f, -3.0f, -4.0f, -17.0f
                };

        outputData =
                {
                        0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                        7.246299848982885e-08f,
                        0.643914213228014f, 0.236882800924671f, 0.087144312427294f, 0.032058600957022f,
                        7.246299848982885e-08f
                };
        break;
        }
    }
    return SimpleSoftmaxBaseTestImpl<ArmnnType, 2>(workloadFactory, memoryManager, beta,
                                                   inputShape, outputData, inputData, axis);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Simple3dSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = 1)
{
    return SimpleSoftmaxBaseTestImpl<ArmnnType, 3>(workloadFactory, memoryManager, beta,
                                                   inputShape, outputData, inputData, axis);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Simple4dSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    float beta,
    const armnn::TensorShape& inputShape,
    const std::vector<float>& outputData,
    const std::vector<float>& inputData,
    int axis = 1)
{

    return SimpleSoftmaxBaseTestImpl<ArmnnType, 4>(workloadFactory, memoryManager, beta,
                                                   inputShape, outputData, inputData, axis);
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

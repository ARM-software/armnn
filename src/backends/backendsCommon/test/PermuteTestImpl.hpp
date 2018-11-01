//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "QuantizeHelper.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <test/TensorHelpers.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

template<typename T>
LayerTestResult<T, 4> SimplePermuteTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        armnn::PermuteDescriptor descriptor,
        armnn::TensorInfo inputTensorInfo,
        armnn::TensorInfo outputTensorInfo,
        const std::vector<T>& inputData,
        const std::vector<T>& outputExpectedData)
{
    auto input = MakeTensor<T, 4>(inputTensorInfo, inputData);

    LayerTestResult<T, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<T, 4>(outputTensorInfo, outputExpectedData);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PermuteQueueDescriptor data;
    data.m_Parameters = descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePermute(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float, 4> SimplePermuteFloat32TestCommon(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1, 2, 2, 2 };
    unsigned int outputShape[] = { 1, 2, 2, 2 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f, 2.0f,
                    3.0f, 4.0f,

                    5.0f, 6.0f,
                    7.0f, 8.0f
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 5.0f, 2.0f, 6.0f,
                    3.0f, 7.0f, 4.0f, 8.0f
            });

    return SimplePermuteTestImpl<float>(workloadFactory, descriptor, inputTensorInfo,
                                        outputTensorInfo, input, outputExpected);
}

LayerTestResult<uint8_t, 4> SimplePermuteUint8TestCommon(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 1, 2, 2, 2 };
    unsigned int outputShape[] = { 1, 2, 2, 2 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(1.0f);

    std::vector<uint8_t> input = std::vector<uint8_t>(
            {
                    1, 2,
                    3, 4,

                    5, 6,
                    7, 8
            });

    std::vector<uint8_t> outputExpected = std::vector<uint8_t>(
            {
                    1, 5, 2, 6,
                    3, 7, 4, 8
            });

    return SimplePermuteTestImpl<uint8_t>(workloadFactory, descriptor, inputTensorInfo,
                                          outputTensorInfo, input, outputExpected);
}

LayerTestResult<float, 4>
PermuteFloat32ValueSet1TestCommon(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 2, 2, 3 };
    unsigned int outputShape[] = { 1, 3, 2, 2 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 2U, 3U, 1U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input = std::vector<float>(
            {
                    1.0f,   2.0f,  3.0f,
                    11.0f, 12.0f, 13.0f,
                    21.0f, 22.0f, 23.0f,
                    31.0f, 32.0f, 33.0f,
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                    1.0f, 11.0f, 21.0f, 31.0f,
                    2.0f, 12.0f, 22.0f, 32.0f,
                    3.0f, 13.0f, 23.0f, 33.0f,
            });

    return SimplePermuteTestImpl<float>(workloadFactory, descriptor, inputTensorInfo,
                                        outputTensorInfo, input, outputExpected);
}

LayerTestResult<float, 4>
PermuteFloat32ValueSet2TestCommon(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 3, 2, 2 };
    unsigned int outputShape[] = { 1, 2, 2, 3 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 3U, 1U, 2U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input = std::vector<float>(
            {
                1.0f, 11.0f, 21.0f, 31.0f,
                2.0f, 12.0f, 22.0f, 32.0f,
                3.0f, 13.0f, 23.0f, 33.0f,
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                1.0f,   2.0f,  3.0f,
                11.0f, 12.0f, 13.0f,
                21.0f, 22.0f, 23.0f,
                31.0f, 32.0f, 33.0f,
            });

    return SimplePermuteTestImpl<float>(workloadFactory, descriptor, inputTensorInfo,
                                        outputTensorInfo, input, outputExpected);
}

LayerTestResult<float, 4>
PermuteFloat32ValueSet3TestCommon(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[]  = { 1, 2, 3, 3 };
    unsigned int outputShape[] = { 1, 3, 2, 3 };

    armnn::PermuteDescriptor descriptor;
    descriptor.m_DimMappings = {0U, 2U, 3U, 1U};

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input = std::vector<float>(
            {
                1.0f,   2.0f,  3.0f,
                11.0f, 12.0f, 13.0f,
                21.0f, 22.0f, 23.0f,
                31.0f, 32.0f, 33.0f,
                41.0f, 42.0f, 43.0f,
                51.0f, 52.0f, 53.0f,
            });

    std::vector<float> outputExpected = std::vector<float>(
            {
                1.0f, 11.0f, 21.0f, 31.0f, 41.0f, 51.0f,
                2.0f, 12.0f, 22.0f, 32.0f, 42.0f, 52.0f,
                3.0f, 13.0f, 23.0f, 33.0f, 43.0f, 53.0f,
            });

    return SimplePermuteTestImpl<float>(workloadFactory, descriptor, inputTensorInfo,
                                        outputTensorInfo, input, outputExpected);
}

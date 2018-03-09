//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadInfo.hpp>

#include "test/TensorHelpers.hpp"
#include "QuantizeHelper.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/WorkloadFactory.hpp"

template<typename T>
LayerTestResult<T, 4> SimpleReshapeTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
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

    armnn::ReshapeQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateReshape(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<float, 4> SimpleReshapeFloat32Test(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 3, 3 };
    unsigned int outputShape[] = { 2, 2, 9, 1 };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

    std::vector<float> input = std::vector<float>(
    {
        0.0f, 1.0f, 2.0f,
        3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f,

        9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f,
        15.0f, 16.0f, 17.0f,

        18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f,

        27.0f, 28.0f, 29.0f,
        30.0f, 31.0f, 32.0f,
        33.0f, 34.0f, 35.0f,
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,

        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,

        18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f,

        27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f,
    });

    return SimpleReshapeTestImpl<float>(workloadFactory, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

LayerTestResult<float, 4> SimpleFloorTest(armnn::IWorkloadFactory& workloadFactory)
{
    const armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);
    const armnn::TensorInfo outputTensorInfo(inputTensorInfo);

    auto input = MakeTensor<float, 4>(inputTensorInfo,
        { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
          1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f });

    LayerTestResult<float, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<float, 4>(outputTensorInfo,
        { -38.0f, -16.0f, -9.0f, -2.0f, -2.0f, -2.0f, -1.0f, -1.0f, 0.0f,
          1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 8.0f, 15.0f, 37.0f });

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::FloorQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFloor(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}

LayerTestResult<uint8_t, 4> SimpleReshapeUint8Test(armnn::IWorkloadFactory& workloadFactory)
{
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int inputShape[] = { 2, 2, 3, 3 };
    unsigned int outputShape[] = { 2, 2, 9, 1 };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::QuantisedAsymm8);
    inputTensorInfo.SetQuantizationScale(1.0f);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::QuantisedAsymm8);
    outputTensorInfo.SetQuantizationScale(1.0f);

    std::vector<uint8_t> input = std::vector<uint8_t>(
    {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,

        9, 10, 11,
        12, 13, 14,
        15, 16, 17,

        18, 19, 20,
        21, 22, 23,
        24, 25, 26,

        27, 28, 29,
        30, 31, 32,
        33, 34, 35,
    });

    std::vector<uint8_t> outputExpected = std::vector<uint8_t>(
    {
        0, 1, 2, 3, 4, 5, 6, 7, 8,

        9, 10, 11, 12, 13, 14, 15, 16, 17,

        18, 19, 20, 21, 22, 23, 24, 25, 26,

        27, 28, 29, 30, 31, 32, 33, 34, 35,
    });

    return SimpleReshapeTestImpl<uint8_t>(workloadFactory, inputTensorInfo, outputTensorInfo, input, outputExpected);
}

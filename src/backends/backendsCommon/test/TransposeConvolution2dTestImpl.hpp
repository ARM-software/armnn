//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "QuantizeHelper.hpp"

#include <armnn/ArmNN.hpp>

#include <ResolveType.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/test/CommonTestUtils.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <boost/test/unit_test.hpp>

#include <string>
#include <utility>
#include <vector>

namespace
{

template<typename T>
using TensorData = std::pair<armnn::TensorInfo, std::vector<T>>;

template<typename T>
void VerifyInputTensorData(const TensorData<T>& data, const std::string& tensorName)
{
    if (data.first.GetNumElements() > data.second.size())
    {
        throw armnn::InvalidArgumentException("Size of data too small for " + tensorName + ": expected " +
            std::to_string(data.first.GetNumElements()) + "but got " + std::to_string(data.second.size()));
    }
}

template<typename T, typename BT>
void TransposeConvolution2dTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                    const armnn::TransposeConvolution2dDescriptor& descriptor,
                                    const TensorData<T>& input,
                                    TensorData<T>& output,
                                    const TensorData<T>& weights,
                                    const armnn::Optional<TensorData<BT>>& biases)
{
    using namespace armnn;

    VerifyInputTensorData(input, "input");
    VerifyInputTensorData(weights, "biases");

    if (descriptor.m_BiasEnabled)
    {
        if (!biases.has_value())
        {
            throw InvalidArgumentException("Bias enabled but no bias data provided");
        }
        VerifyInputTensorData(biases.value(), "biases");
    }

    // set up weights
    ScopedCpuTensorHandle weightsTensor(weights.first);

    TransposeConvolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    queueDescriptor.m_Weight     = &weightsTensor;

    AllocateAndCopyDataToITensorHandle(&weightsTensor, weights.second.data());

    std::unique_ptr<ScopedCpuTensorHandle> biasesTensor;
    if (descriptor.m_BiasEnabled)
    {
        // set up biases
        biasesTensor = std::make_unique<ScopedCpuTensorHandle>(biases.value().first);
        queueDescriptor.m_Bias = biasesTensor.get();

        AllocateAndCopyDataToITensorHandle(biasesTensor.get(), biases.value().second.data());
    }

    // set up input and output handles
    std::unique_ptr<ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(input.first);
    std::unique_ptr<ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(output.first);

    // set up workload
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, input.first, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, output.first, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload =
            workloadFactory.CreateTransposeConvolution2d(queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.second.data());

    ExecuteWorkload(*workload, memoryManager);

    // copy output
    output.second = std::vector<T>(output.first.GetNumElements(), 0.0f);
    CopyDataFromITensorHandle(output.second.data(), outputHandle.get());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> TransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TransposeConvolution2dDescriptor& descriptor,
    armnn::TensorInfo& inputInfo,
    const std::vector<float>& inputData,
    armnn::TensorInfo& outputInfo,
    const std::vector<float>& expectedOutputData,
    armnn::TensorInfo& weightsInfo,
    const std::vector<float>& weightsData,
    armnn::TensorInfo& biasesInfo,
    const std::vector<float>& biasesData)
{
    using namespace armnn;

    // set up quantization parameters
    if (armnn::IsQuantizedType<T>())
    {
        constexpr float   qScale  = 0.50f;
        constexpr int32_t qOffset = 10;

        inputInfo.SetQuantizationScale(qScale);
        inputInfo.SetQuantizationOffset(qOffset);

        outputInfo.SetQuantizationScale(qScale);
        outputInfo.SetQuantizationOffset(qOffset);

        weightsInfo.SetQuantizationScale(qScale);
        weightsInfo.SetQuantizationOffset(qOffset);

        biasesInfo.SetQuantizationScale(qScale * qScale);
        biasesInfo.SetQuantizationOffset(0);
    }

    // set up input
    TensorData<T> input =
    {
        inputInfo,
        QuantizedVector<T>(inputInfo.GetQuantizationScale(), inputInfo.GetQuantizationOffset(), inputData)
    };

    // set up weights
    TensorData<T> weights =
    {
        weightsInfo,
        QuantizedVector<T>(weightsInfo.GetQuantizationScale(), weightsInfo.GetQuantizationOffset(), weightsData)
    };

    // set up biases
    using BT = armnn::ResolveType<ArmnnBType>;
    Optional<TensorData<BT>> optionalBiases;
    if (descriptor.m_BiasEnabled)
    {
        TensorData<BT> biases =
        {
            biasesInfo,
            QuantizedVector<BT>(biasesInfo.GetQuantizationScale(), biasesInfo.GetQuantizationOffset(), biasesData)
        };

        optionalBiases = Optional<TensorData<BT>>(biases);
    }

    // set up output
    TensorData<T> output = { outputInfo, {} };

    // execute test
    TransposeConvolution2dTestImpl(workloadFactory,
                                   memoryManager,
                                   descriptor,
                                   input,
                                   output,
                                   weights,
                                   optionalBiases);

    // construct result object
    LayerTestResult<T, 4> testResult(outputInfo);
    testResult.output         = MakeTensor<T, 4>(outputInfo, output.second);
    testResult.outputExpected = MakeTensor<T, 4>(outputInfo,
                                                 QuantizedVector<T>(outputInfo.GetQuantizationScale(),
                                                                    outputInfo.GetQuantizationOffset(),
                                                                    expectedOutputData));

    return testResult;
}

template<typename T>
void SwizzleData(const armnn::TensorInfo& inputInfo,
                 std::vector<T>& inputData,
                 const armnn::TensorInfo& outputInfo,
                 std::vector<T>& outputData,
                 const armnn::TensorInfo& weightsInfo,
                 std::vector<T>& weightsData)
{
    constexpr size_t dataTypeSize = sizeof(float);
    const armnn::PermutationVector nchwToNhwc = { 0, 3, 1, 2 };

    std::vector<T> tmp(inputData.size());
    armnnUtils::Permute(inputInfo.GetShape(), nchwToNhwc, inputData.data(), tmp.data(), dataTypeSize);
    inputData = tmp;

    tmp.resize(weightsData.size());
    armnnUtils::Permute(weightsInfo.GetShape(), nchwToNhwc, weightsData.data(), tmp.data(), dataTypeSize);
    weightsData = tmp;

    tmp.resize(outputData.size());
    armnnUtils::Permute(outputInfo.GetShape(), nchwToNhwc, outputData.data(), tmp.data(), dataTypeSize);
    outputData = tmp;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    using namespace armnn;

    constexpr unsigned int batches  = 1u;
    constexpr unsigned int channels = 1u;

    constexpr unsigned int wInput = 3u;
    constexpr unsigned int hInput = wInput;

    constexpr unsigned int wOutput = 5u;
    constexpr unsigned int hOutput = wOutput;

    constexpr unsigned int wWeights = 3u;
    constexpr unsigned int hWeights = wWeights;

    TensorShape inputShape   = MakeTensorShape(batches, channels, hInput, wInput, layout);
    TensorShape outputShape  = MakeTensorShape(batches, channels, hOutput, wOutput, layout);
    TensorShape weightsShape = MakeTensorShape(batches, channels, hWeights, wWeights, layout);

    TensorInfo inputInfo(inputShape, ArmnnType);
    TensorInfo outputInfo(outputShape, ArmnnType);
    TensorInfo weightsInfo(weightsShape, ArmnnType);
    TensorInfo biasesInfo({ channels }, ArmnnBType);

    std::vector<float> inputData =
    {
       1.f, 1.f, 1.f,
       1.f, 1.f, 1.f,
       1.f, 1.f, 1.f
    };

    std::vector<float> weightsData =
    {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f
    };

    std::vector<float> biasesData = { 1.f };

    std::vector<float> expectedOutputData =
    {
         1.f,  3.f,  6.f,  5.f,  3.f,
         5.f, 12.f, 21.f, 16.f,  9.f,
        12.f, 27.f, 45.f, 33.f, 18.f,
        11.f, 24.f, 39.f, 28.f, 15.f,
         7.f, 15.f, 24.f, 17.f,  9.f
    };

    if (biasEnabled)
    {
        // apply bias to expected output data
        std::transform(expectedOutputData.begin(), expectedOutputData.end(), expectedOutputData.begin(),
                       [&](float f) -> float { return f + biasesData[0]; });
    }

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout  = layout;

    // swizzle data if needed
    if (layout == armnn::DataLayout::NHWC)
    {
       SwizzleData(inputInfo, inputData, outputInfo, expectedOutputData, weightsInfo, weightsData);
    }

    return TransposeConvolution2dTest<ArmnnType, ArmnnBType>(workloadFactory,
                                                             memoryManager,
                                                             descriptor,
                                                             inputInfo,
                                                             inputData,
                                                             outputInfo,
                                                             expectedOutputData,
                                                             weightsInfo,
                                                             weightsData,
                                                             biasesInfo,
                                                             biasesData);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PaddedTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    using namespace armnn;

    constexpr unsigned int batches  = 1u;
    constexpr unsigned int channels = 1u;

    constexpr unsigned int wInput = 4u;
    constexpr unsigned int hInput = wInput;

    constexpr unsigned int wOutput = 2u;
    constexpr unsigned int hOutput = wOutput;

    constexpr unsigned int wWeights = 3u;
    constexpr unsigned int hWeights = wWeights;

    TensorShape inputShape   = MakeTensorShape(batches, channels, hInput, wInput, layout);
    TensorShape outputShape  = MakeTensorShape(batches, channels, hOutput, wOutput, layout);
    TensorShape weightsShape = MakeTensorShape(batches, channels, hWeights, wWeights, layout);

    TensorInfo inputInfo(inputShape, ArmnnType);
    TensorInfo outputInfo(outputShape, ArmnnType);
    TensorInfo weightsInfo(weightsShape, ArmnnType);
    TensorInfo biasesInfo({ channels }, ArmnnBType);

    std::vector<float> inputData =
    {
       1.f, 3.f, 2.f, 1.f,
       1.f, 3.f, 3.f, 1.f,
       2.f, 1.f, 1.f, 3.f,
       3.f, 2.f, 3.f, 3.f
    };

    std::vector<float> weightsData =
    {
        1.f, 2.f, 3.f,
        0.f, 1.f, 0.f,
        2.f, 1.f, 2.f
    };

    std::vector<float> biasesData = { 1.f };

    std::vector<float> expectedOutputData =
    {
         21.f, 21.f,
         28.f, 27.f
    };

    if (biasEnabled)
    {
        // apply bias to expected output data
        std::transform(expectedOutputData.begin(), expectedOutputData.end(), expectedOutputData.begin(),
                       [&](float f) -> float { return f + biasesData[0]; });
    }

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 2;
    descriptor.m_PadRight    = 2;
    descriptor.m_PadTop      = 2;
    descriptor.m_PadBottom   = 2;
    descriptor.m_StrideX     = 1;
    descriptor.m_StrideY     = 1;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout  = layout;

    // swizzle data if needed
    if (layout == armnn::DataLayout::NHWC)
    {
        SwizzleData(inputInfo, inputData, outputInfo, expectedOutputData, weightsInfo, weightsData);
    }

    return TransposeConvolution2dTest<ArmnnType, ArmnnBType>(workloadFactory,
                                                             memoryManager,
                                                             descriptor,
                                                             inputInfo,
                                                             inputData,
                                                             outputInfo,
                                                             expectedOutputData,
                                                             weightsInfo,
                                                             weightsData,
                                                             biasesInfo,
                                                             biasesData);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> StridedTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    bool biasEnabled,
    const armnn::DataLayout layout)
{
    using namespace armnn;

    constexpr unsigned int batches  = 1u;
    constexpr unsigned int channels = 1u;

    constexpr unsigned int wInput = 3u;
    constexpr unsigned int hInput = wInput;

    constexpr unsigned int wOutput = 7u;
    constexpr unsigned int hOutput = wOutput;

    constexpr unsigned int wWeights = 3u;
    constexpr unsigned int hWeights = wWeights;

    TensorShape inputShape   = MakeTensorShape(batches, channels, hInput, wInput, layout);
    TensorShape outputShape  = MakeTensorShape(batches, channels, hOutput, wOutput, layout);
    TensorShape weightsShape = MakeTensorShape(batches, channels, hWeights, wWeights, layout);

    TensorInfo inputInfo(inputShape, ArmnnType);
    TensorInfo outputInfo(outputShape, ArmnnType);
    TensorInfo weightsInfo(weightsShape, ArmnnType);
    TensorInfo biasesInfo({ channels }, ArmnnBType);

    std::vector<float> inputData =
    {
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f,
        1.f, 1.f, 1.f
    };

    std::vector<float> weightsData =
    {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f,
        7.f, 8.f, 9.f
    };

    std::vector<float> biasesData = { 1.f };

    std::vector<float> expectedOutputData =
    {
        1.f,  2.f,  4.f,  2.f,  4.f,  2.f,  3.f,
        4.f,  5.f, 10.f,  5.f, 10.f,  5.f,  6.f,
        8.f, 10.f, 20.f, 10.f, 20.f, 10.f, 12.f,
        4.f,  5.f, 10.f,  5.f, 10.f,  5.f,  6.f,
        8.f, 10.f, 20.f, 10.f, 20.f, 10.f, 12.f,
        4.f,  5.f, 10.f,  5.f, 10.f,  5.f,  6.f,
        7.f,  8.f, 16.f,  8.f, 16.f,  8.f,  9.f
    };

    if (biasEnabled)
    {
        // apply bias to expected output data
        std::transform(expectedOutputData.begin(), expectedOutputData.end(), expectedOutputData.begin(),
                    [&](float f) -> float { return f + biasesData[0]; });
    }

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout  = layout;

    // swizzle data if needed
    if (layout == armnn::DataLayout::NHWC)
    {
        SwizzleData(inputInfo, inputData, outputInfo, expectedOutputData, weightsInfo, weightsData);
    }

    return TransposeConvolution2dTest<ArmnnType, ArmnnBType>(workloadFactory,
                                                             memoryManager,
                                                             descriptor,
                                                             inputInfo,
                                                             inputData,
                                                             outputInfo,
                                                             expectedOutputData,
                                                             weightsInfo,
                                                             weightsData,
                                                             biasesInfo,
                                                             biasesData);
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MultiChannelTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::DataLayout layout)
{
    using namespace armnn;

    TensorShape inputShape   = MakeTensorShape(1, 1, 2, 2, layout);
    TensorShape outputShape  = MakeTensorShape(1, 2, 5, 5, layout);

    // OIHW for NCHW; OHWI for NHWC
    TensorShape weightsShape = MakeTensorShape(2, 1, 3, 3, layout);
    TensorShape biasesShape  = { 2 };

    TensorInfo inputInfo(inputShape, ArmnnType);
    TensorInfo outputInfo(outputShape, ArmnnType);
    TensorInfo weightsInfo(weightsShape, ArmnnType);
    TensorInfo biasesInfo(biasesShape, ArmnnBType);

    std::vector<float> inputData =
    {
        1.f, 2.f,
        3.f, 4.f,
    };

    std::vector<float> weightsData =
    {
         1.f,  3.f,  5.f,
         7.f,  9.f, 11.f,
        13.f, 15.f, 17.f,

         2.f,  4.f,  6.f,
         8.f, 10.f, 12.f,
        14.f, 16.f, 18.f
    };

    std::vector<float> biasesData = { -1.5f, -2.0f };

    std::vector<float> expectedOutputData =
    {
        -0.5f,  1.5f,   5.5f,  4.5f,  8.5f,
         5.5f,  7.5f,  23.5f, 16.5f, 20.5f,
        14.5f, 22.5f,  60.5f, 40.5f, 52.5f,
        19.5f, 25.5f,  59.5f, 34.5f, 42.5f,
        37.5f, 43.5f, 101.5f, 58.5f, 66.5f,

         0.0f,  2.0f,   8.0f,  6.0f, 10.0f,
         6.0f,  8.0f,  26.0f, 18.0f, 22.0f,
        18.0f, 26.0f,  70.0f, 46.0f, 58.0f,
        22.0f, 28.0f,  66.0f, 38.0f, 46.0f,
        40.0f, 46.0f, 108.0f, 62.0f, 70.0f,
    };

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = layout;

    // swizzle data if needed
    if (layout == armnn::DataLayout::NHWC)
    {
        SwizzleData(inputInfo, inputData, outputInfo, expectedOutputData, weightsInfo, weightsData);
    }

    return TransposeConvolution2dTest<ArmnnType, ArmnnBType>(workloadFactory,
                                                             memoryManager,
                                                             descriptor,
                                                             inputInfo,
                                                             inputData,
                                                             outputInfo,
                                                             expectedOutputData,
                                                             weightsInfo,
                                                             weightsData,
                                                             biasesInfo,
                                                             biasesData);
}

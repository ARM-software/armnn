//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConvolution2dTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>


#include <armnnUtils/Permute.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <doctest/doctest.h>

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
                                    const armnn::ITensorHandleFactory& tensorHandleFactory,
                                    const armnn::TransposeConvolution2dDescriptor& descriptor,
                                    const TensorData<T>& input,
                                    TensorData<T>& output,
                                    const TensorData<T>& weights,
                                    const armnn::Optional<TensorData<BT>>& biases)
{
    IgnoreUnused(memoryManager);
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
    ScopedTensorHandle weightsTensor(weights.first);

    TransposeConvolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    queueDescriptor.m_Weight     = &weightsTensor;

    AllocateAndCopyDataToITensorHandle(&weightsTensor, weights.second.data());

    std::unique_ptr<ScopedTensorHandle> biasesTensor;
    if (descriptor.m_BiasEnabled)
    {
        // set up biases
        biasesTensor = std::make_unique<ScopedTensorHandle>(biases.value().first);
        queueDescriptor.m_Bias = biasesTensor.get();

        AllocateAndCopyDataToITensorHandle(biasesTensor.get(), biases.value().second.data());
    }

    // set up input and output handles
    std::unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(input.first);
    std::unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(output.first);

    // set up workload
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(queueDescriptor, workloadInfo, input.first, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, output.first, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload =
            workloadFactory.CreateWorkload(armnn::LayerType::TransposeConvolution2d, queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.second.data());

    ExecuteWorkload(*workload, memoryManager);

    // copy output
    output.second = std::vector<T>(output.first.GetNumElements(), T());
    CopyDataFromITensorHandle(output.second.data(), outputHandle.get());
}

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> TransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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
        armnnUtils::QuantizedVector<T>(inputData, inputInfo.GetQuantizationScale(), inputInfo.GetQuantizationOffset())
    };

    // set up weights
    TensorData<T> weights =
    {
        weightsInfo,
        armnnUtils::QuantizedVector<T>(weightsData,
                                       weightsInfo.GetQuantizationScale(),
                                       weightsInfo.GetQuantizationOffset())
    };

    // set up biases
    using BT = armnn::ResolveType<ArmnnBType>;
    Optional<TensorData<BT>> optionalBiases;
    if (descriptor.m_BiasEnabled)
    {
        TensorData<BT> biases =
        {
            biasesInfo,
            armnnUtils::QuantizedVector<BT>(biasesData,
                                            biasesInfo.GetQuantizationScale(),
                                            biasesInfo.GetQuantizationOffset())
        };

        optionalBiases = Optional<TensorData<BT>>(biases);
    }

    // set up output
    TensorData<T> output = { outputInfo, {} };

    // execute test
    TransposeConvolution2dTestImpl(workloadFactory,
                                   memoryManager,
                                   tensorHandleFactory,
                                   descriptor,
                                   input,
                                   output,
                                   weights,
                                   optionalBiases);

    // construct result object
    LayerTestResult<T, 4> testResult(outputInfo);
    testResult.m_ActualData = output.second;
    testResult.m_ExpectedData = armnnUtils::QuantizedVector<T>(expectedOutputData,
                                                               outputInfo.GetQuantizationScale(),
                                                               outputInfo.GetQuantizationOffset());

    return testResult;
}

template<typename T>
void SwizzleData(armnn::TensorInfo& inputInfo,
                 std::vector<T>& inputData,
                 armnn::TensorInfo& outputInfo,
                 std::vector<T>& outputData,
                 armnn::TensorInfo& weightsInfo,
                 std::vector<T>& weightsData)
{
    PermuteTensorNchwToNhwc<T>(inputInfo, inputData);
    PermuteTensorNchwToNhwc<T>(outputInfo, outputData);
    PermuteTensorNchwToNhwc<T>(weightsInfo, weightsData);
}

} // anonymous namespace

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> SimpleTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    TensorShape inputShape   = { batches, channels, hInput,   wInput   };
    TensorShape outputShape  = { batches, channels, hOutput,  wOutput  };
    TensorShape weightsShape = { batches, channels, hWeights, wWeights };

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
                                                             tensorHandleFactory,
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

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> PaddedTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    TensorShape inputShape   = { batches, channels, hInput,   wInput   };
    TensorShape outputShape  = { batches, channels, hOutput,  wOutput  };
    TensorShape weightsShape = { batches, channels, hWeights, wWeights };

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
                                                             tensorHandleFactory,
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

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> StridedTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
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

    TensorShape inputShape   = { batches, channels, hInput,   wInput   };
    TensorShape outputShape  = { batches, channels, hOutput,  wOutput  };
    TensorShape weightsShape = { batches, channels, hWeights, wWeights };

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
                                                             tensorHandleFactory,
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

template<armnn::DataType ArmnnType, armnn::DataType ArmnnBType, typename T>
LayerTestResult<T, 4> MultiChannelTransposeConvolution2dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
    using namespace armnn;

    TensorShape inputShape   = { 1, 1, 2, 2 };
    TensorShape outputShape  = { 1, 2, 5, 5 };

    // OIHW for NCHW; OHWI for NHWC
    TensorShape weightsShape = { 2, 1, 3, 3 };
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
        40.0f, 46.0f, 108.0f, 62.0f, 70.0f
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
                                                             tensorHandleFactory,
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

LayerTestResult<uint8_t, 4> TransposeConvolution2dPerAxisQuantTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout)
{
   using namespace armnn;

    const DataType inputType  = DataType::QAsymmU8;
    const DataType kernelType = DataType::QSymmS8;
    const DataType biasType   = DataType::Signed32;

    TensorInfo inputInfo ({ 1, 1, 2, 2 }, inputType, 0.50f, 10);
    TensorInfo outputInfo({ 1, 2, 5, 5 }, inputType, 0.50f, 10);

    const std::vector<float> quantScales{ 0.25f, 0.5f };
    constexpr unsigned int quantDimension = 0;

    TensorInfo kernelInfo({ 2, 1, 3, 3 }, kernelType, quantScales, quantDimension);

    const std::vector<float> biasQuantScales{ 0.125f, 0.25f };
    TensorInfo biasInfo({ 2 }, biasType, biasQuantScales, quantDimension);

    std::vector<uint8_t> inputData =
    {
        12, 14,
        16, 18
    };

    std::vector<int8_t> kernelData =
    {
         4, 12, 20,
        28, 36, 44,
        52, 60, 68,

         4,  8, 12,
        16, 20, 24,
        28, 32, 36
    };

    std::vector<int32_t> biasData = { -12, -8 };

    std::vector<uint8_t> actualOutput(outputInfo.GetNumElements());

    std::vector<uint8_t> expectedOutputData =
    {
         9,  13,  21,  19,  27,
        21,  25,  57,  43,  51,
        39,  55, 131,  91, 115,
        49,  61, 129,  79,  95,
        85,  97, 213, 127, 143,

        10,  14,  26,  22,  30,
        22,  26,  62,  46,  54,
        46,  62, 150, 102, 126,
        54,  66, 142,  86, 102,
        90, 102, 226, 134, 150
    };

    if (layout == DataLayout::NHWC)
    {
        PermuteTensorNchwToNhwc(inputInfo, inputData);
        PermuteTensorNchwToNhwc(kernelInfo, kernelData);
        PermuteTensorNchwToNhwc(outputInfo, expectedOutputData);
    }

    TransposeConvolution2dDescriptor descriptor;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_BiasEnabled = true;
    descriptor.m_DataLayout  = layout;

    std::unique_ptr<ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    WorkloadInfo workloadInfo;
    ScopedTensorHandle weightTensor(kernelInfo);
    ScopedTensorHandle biasTensor(biasInfo);

    AllocateAndCopyDataToITensorHandle(&weightTensor, kernelData.data());
    AllocateAndCopyDataToITensorHandle(&biasTensor, biasData.data());

    TransposeConvolution2dQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    queueDescriptor.m_Weight     = &weightTensor;
    queueDescriptor.m_Bias       = &biasTensor;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    std::unique_ptr<IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::TransposeConvolution2d,
                                                                         queueDescriptor,
                                                                         workloadInfo);
    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<uint8_t, 4>(actualOutput,
                                       expectedOutputData,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
SimpleTransposeConvolution2dTest<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
SimpleTransposeConvolution2dTest<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
SimpleTransposeConvolution2dTest<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
SimpleTransposeConvolution2dTest<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
PaddedTransposeConvolution2dTest<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
PaddedTransposeConvolution2dTest<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
PaddedTransposeConvolution2dTest<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
PaddedTransposeConvolution2dTest<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
StridedTransposeConvolution2dTest<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
StridedTransposeConvolution2dTest<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
StridedTransposeConvolution2dTest<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
StridedTransposeConvolution2dTest<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
MultiChannelTransposeConvolution2dTest<armnn::DataType::Float32, armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
MultiChannelTransposeConvolution2dTest<armnn::DataType::QAsymmS8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
MultiChannelTransposeConvolution2dTest<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
MultiChannelTransposeConvolution2dTest<armnn::DataType::QSymmS16, armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::DataLayout layout);

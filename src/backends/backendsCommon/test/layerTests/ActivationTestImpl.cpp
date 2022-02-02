//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ActivationTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <backendsCommon/test/ActivationFixture.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <algorithm>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BoundedReLuTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float upperBound,
    float lowerBound,
    float inputScale,
    int32_t inputOffset,
    float outputScale,
    int32_t outputOffset,
    const std::vector<T>& inputData,
    const std::vector<T>& outputExpectedData,
    unsigned int inputWidth,
    unsigned int inputHeight,
    unsigned int inputChannels,
    unsigned int inputBatchSize)
{
    IgnoreUnused(memoryManager);
    unsigned int outputWidth = inputWidth;
    unsigned int outputHeight = inputHeight;
    unsigned int outputChannels = inputChannels;
    unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth }, ArmnnType);

    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth }, ArmnnType);

    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(inputScale);
        inputTensorInfo.SetQuantizationOffset(inputOffset);

        outputTensorInfo.SetQuantizationScale(outputScale);
        outputTensorInfo.SetQuantizationOffset(outputOffset);
    }

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    // Setup bounded ReLu.
    armnn::ActivationQueueDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(descriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    descriptor.m_Parameters.m_Function = armnn::ActivationFunction::BoundedReLu;
    descriptor.m_Parameters.m_A = upperBound;
    descriptor.m_Parameters.m_B = lowerBound;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                descriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputExpectedData,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> BoundedReLuUpperAndLowerBoundTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int inputWidth = 4u;
    unsigned int inputHeight = 5u;
    unsigned int inputChannels = 1u;
    unsigned int inputBatchSize = 1;

    std::vector<float> input = std::vector<float>{
      -2.0f,       0.1f,     0.5f,     1.25f,
     0.786f,    0.9875f,    -1.5f,    0.384f,
    1.0001f,       3.5f,     7.5f,    0.896f,
     2.126f,       2.0f,     0.3f,     0.15f,
     0.999f,       1.2f,    0.89f,      6.1f,
    };

    // Calculated manually.
    std::vector<float> output = std::vector<float>{
      -1.0f,       0.1f,     0.5f,      1.0f,
     0.786f,    0.9875f,    -1.0f,    0.384f,
       1.0f,       1.0f,     1.0f,    0.896f,
       1.0f,       1.0f,     0.3f,     0.15f,
     0.999f,       1.0f,    0.89f,      1.0f,
    };

    return BoundedReLuTestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 1.0f, -1.0f, 1.0f, 0, 1.0f, 0, input, output,
        inputWidth, inputHeight, inputChannels, inputBatchSize);
}

LayerTestResult<float, 4> BoundedReLuUpperBoundOnlyTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int inputWidth = 4u;
    unsigned int inputHeight = 5u;
    unsigned int inputChannels = 1u;
    unsigned int inputBatchSize = 1;

    std::vector<float> input = std::vector<float>{
      -1.0f,       0.1f,     0.5f,      6.25f,
     0.786f,    5.9875f,    -0.5f,     0.384f,
    6.0001f,       3.5f,     7.5f,     0.896f,
     2.126f,      12.0f,     0.3f,      0.15f,
     0.999f,       1.2f,    0.89f,       6.1f,
    };

    // Calculated manually.
    std::vector<float> output = std::vector<float>{
       0.0f,       0.1f,     0.5f,       6.0f,
     0.786f,    5.9875f,     0.0f,     0.384f,
       6.0f,       3.5f,     6.0f,     0.896f,
     2.126f,       6.0f,     0.3f,      0.15f,
     0.999f,       1.2f,    0.89f,       6.0f,
    };

    return BoundedReLuTestCommon<armnn::DataType::Float32>(
        workloadFactory, memoryManager, tensorHandleFactory, 6.0f, 0.0f, 1.0f, 0, 1.0f, 0, input, output,
        inputWidth, inputHeight, inputChannels, inputBatchSize);
}

LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperBoundOnlyTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int inputWidth     = 3u;
    unsigned int inputHeight    = 2u;
    unsigned int inputChannels  = 1u;
    unsigned int inputBatchSize = 1;

    std::vector<uint8_t> input = std::vector<uint8_t>{
         51, 124, 28,
        251,   8, 92
    };

    // Calculated manually.
    std::vector<uint8_t> output = std::vector<uint8_t>{
          0, 122,  0,
        255,   0, 58
    };

    float inputScale     = 12.0f / 255.0f;
    int32_t inputOffset  = 63;
    float outputScale    = 6.0f / 255.0f;
    int32_t outputOffset = 0;

    return BoundedReLuTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, tensorHandleFactory, 6.0f, 0.0f,
        inputScale, inputOffset, outputScale, outputOffset,
        input, output, inputWidth, inputHeight, inputChannels, inputBatchSize);
}

LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperAndLowerBoundTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int inputWidth     = 3u;
    unsigned int inputHeight    = 2u;
    unsigned int inputChannels  = 1u;
    unsigned int inputBatchSize = 1;

    std::vector<uint8_t> input = std::vector<uint8_t>{
         51, 230, 28,
        251,   8, 92
    };

    // Calculated manually.
    std::vector<uint8_t> output = std::vector<uint8_t>{
         51, 192, 32,
        192,  32, 92
    };

    int32_t inputOffset = 112;
    float inputScale    = 0.0125f;

    return BoundedReLuTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, tensorHandleFactory, 1.0f, -1.0f,
        inputScale, inputOffset, inputScale, inputOffset, // Input/output scale & offset same.
        input, output, inputWidth, inputHeight, inputChannels, inputBatchSize);
}

namespace
{

struct BoundedReLuRandomInputTestTraits
{
    constexpr static unsigned int inputHeight = 31u;
    constexpr static unsigned int inputWidth = 19u;
    constexpr static unsigned int inputChannels = 4u;
    constexpr static unsigned int inputBatchSize = 2;

    constexpr static unsigned int outputHeight = inputHeight;
    constexpr static unsigned int outputWidth = inputWidth;
    constexpr static unsigned int outputChannels = inputChannels;
    constexpr static unsigned int outputBatchSize = inputBatchSize;

    static armnn::TensorInfo GetInputTensorInfo()
    {
        return armnn::TensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth },
            armnn::DataType::Float32);
    }

    static armnn::TensorInfo GetOutputTensorInfo()
    {
        return armnn::TensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth },
            armnn::DataType::Float32);
    }
};

std::vector<float> BoundedReLuRandomInputTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float lowerBound,
    float upperBound,
    const armnn::ActivationDescriptor& activationDescriptor)
{
    IgnoreUnused(memoryManager);
    const armnn::TensorInfo inputTensorInfo = BoundedReLuRandomInputTestTraits::GetInputTensorInfo();
    const armnn::TensorInfo outputTensorInfo = BoundedReLuRandomInputTestTraits::GetOutputTensorInfo();

    // Min/max random values passed to MakeRandomTensor are purposely outside of the ReLu
    // range [lowerBound, upperBound].
    std::vector<float> input = MakeRandomTensor<float>(inputTensorInfo, 4605828, lowerBound - 5.0f, upperBound * 2.0f);
    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    // Set up bounded ReLu.
    armnn::ActivationQueueDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(descriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, workloadInfo, outputTensorInfo, outputHandle.get());
    descriptor.m_Parameters = activationDescriptor;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                descriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return actualOutput;
}

} // namespace

LayerTestResult<float, 4> CompareBoundedReLuTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    float upperBound,
    float lowerBound)
{
    LayerTestResult<float, 4> result(BoundedReLuRandomInputTestTraits::GetOutputTensorInfo());

    armnn::ActivationDescriptor activationDescriptor;
    activationDescriptor.m_Function = armnn::ActivationFunction::BoundedReLu;
    activationDescriptor.m_A = upperBound;
    activationDescriptor.m_B = lowerBound;

    result.m_ActualData = BoundedReLuRandomInputTest(
        workloadFactory, memoryManager, tensorHandleFactory, 0.0f, upperBound, activationDescriptor);
    result.m_ExpectedData = BoundedReLuRandomInputTest(
        refWorkloadFactory, nullptr, refTensorHandleFactory, 0.0f, upperBound, activationDescriptor);

    return result;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ConstantLinearActivationTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale = 0.0f,
    int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    unsigned int inputHeight    = 20;
    unsigned int inputWidth     = 17;
    unsigned int inputChannels  = 3;
    unsigned int batchSize      = 5;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[]  = {batchSize, inputChannels, inputHeight, inputWidth};

    inputTensorInfo = armnn::TensorInfo(4, shape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, shape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    // Do linear activation that should leave the tensor unchanged.
    armnn::ActivationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_A = 1.0f;
    data.m_Parameters.m_B = 0.0f;
    data.m_Parameters.m_Function = armnn::ActivationFunction::Linear;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    std::vector<T> input = MakeRandomTensor<T>(inputTensorInfo, 7123561);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    // Use input as ExpectedData as tensor doesn't change.
    return LayerTestResult<T, 4>(actualOutput,
                                 input,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

LayerTestResult<float, 4> ConstantLinearActivationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ConstantLinearActivationTestCommon<armnn::DataType::Float32>(workloadFactory,
                                                                        memoryManager,
                                                                        tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> ConstantLinearActivationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ConstantLinearActivationTestCommon<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, tensorHandleFactory, 4.0f, 3);
}

LayerTestResult<int16_t, 4> ConstantLinearActivationInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ConstantLinearActivationTestCommon<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleActivationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::ActivationFunction activationFunction,
    float activationParameterA,
    float activationParameterB,
    float scale,
    int32_t offset,
    const std::vector<float>& inputData,
    float outScale,
    int32_t outOffset,
    const std::vector<float>& outputExpectedData)
{
    IgnoreUnused(memoryManager);
    constexpr static unsigned int inputWidth = 16u;
    constexpr static unsigned int inputHeight = 1u;
    constexpr static unsigned int inputChannels = 1u;
    constexpr static unsigned int inputBatchSize = 1u;

    constexpr static unsigned int outputWidth = inputWidth;
    constexpr static unsigned int outputHeight = inputHeight;
    constexpr static unsigned int outputChannels = inputChannels;
    constexpr static unsigned int outputBatchSize = inputBatchSize;

    armnn::TensorInfo inputTensorInfo({ inputBatchSize, inputChannels, inputHeight, inputWidth }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ outputBatchSize, outputChannels, outputHeight, outputWidth }, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(scale);
        inputTensorInfo.SetQuantizationOffset(offset);
        outputTensorInfo.SetQuantizationScale(outScale);
        outputTensorInfo.SetQuantizationOffset(outOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, scale, offset);

    // Calculated outputExpected manually.
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>(outputExpectedData, outScale, outOffset);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    // Setup bounded ReLu.
    armnn::ActivationQueueDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(descriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    descriptor.m_Parameters.m_Function = activationFunction;
    descriptor.m_Parameters.m_A = activationParameterA;
    descriptor.m_Parameters.m_B = activationParameterB;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                descriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 outputExpected,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SimpleSigmoidTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset)
{
    std::vector<float> inputData =
    {
        -0.1f, -0.2f, -0.3f, -0.4f,
        0.1f,  0.2f,  0.3f,  0.4f,
        -1.0f, -2.0f, -3.0f, -4.0f,
        1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return 1.0f / (1.0f + std::exp(-value));
    };
    std::vector<float> m_OutputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), m_OutputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::Sigmoid,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           1.f / 256.f,
                                           0,
                                           m_OutputExpected);
}

LayerTestResult<float, 4> SimpleSigmoidTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SimpleSigmoidTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager,
                                                            tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> SimpleSigmoidUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SimpleSigmoidTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                              tensorHandleFactory, 0.1f, 50);
}

LayerTestResult<int16_t, 4> SimpleSigmoidInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SimpleSigmoidTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager,
                                                              tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> ReLuTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return std::fmax(0.0f, value);
    };
    std::vector<float> outputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), outputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::ReLu,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           outputExpected);
}

LayerTestResult<int16_t, 4> ReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ReLuTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}


LayerTestResult<uint8_t, 4> ReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ReLuTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<float, 4> ReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ReLuTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> BoundedReLuTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };
    const float a = 1.0f;
    const float b = -1.0f;
    // Calculate output values for input.
    auto f = [a, b](float value)
    {
        return std::min(a, std::max(b, value));
    };
    std::vector<float> outputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), outputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::BoundedReLu,
                                           a,
                                           b,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           outputExpected);
}

LayerTestResult<int16_t, 4> BoundedReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return ReLuTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}



template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SoftReLuTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return std::log(1.0f + std::exp(value));
    };
    std::vector<float> outputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), outputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::SoftReLu,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           outputExpected);
}

LayerTestResult<float, 4> SoftReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SoftReLuTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> SoftReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SoftReLuTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                         tensorHandleFactory, 0.0625f, 64);
}

LayerTestResult<int16_t, 4> SoftReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SoftReLuTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> LeakyReLuTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    const float a = 0.01f;
    // Calculate output values for input.
    auto f = [a](float value)
    {
        return value > 0.0f ? value : (value * a);
    };
    std::vector<float> outputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), outputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::LeakyReLu,
                                           a,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           outputExpected);
}

LayerTestResult<float, 4> LeakyReLuTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LeakyReLuTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> LeakyReLuUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LeakyReLuTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                          tensorHandleFactory, 0.0625f, 64);
}

LayerTestResult<int16_t, 4> LeakyReLuInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return LeakyReLuTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> AbsTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return std::abs(value);
    };
    std::vector<float> outputExpected(inputData.size());
    std::transform(inputData.begin(), inputData.end(), outputExpected.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::Abs,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           outputExpected);
}

LayerTestResult<float, 4> AbsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AbsTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> AbsUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AbsTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.0625f, 64);
}

LayerTestResult<int16_t, 4> AbsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return AbsTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<float, 5> SqrtNNTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    const int inputDataSize = 120;
    std::vector<float> inputData(inputDataSize);

    for (unsigned int i = 0u; i < inputDataSize; ++i)
    {
        inputData[i] = static_cast<float>(i) / 10;
    }

    auto f = [](float value)
    {
        return std::sqrt(value);
    };
    std::vector<float> expectedOutput(inputDataSize);
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    armnn::TensorInfo inputTensorInfo(
        { 1u, 2u, 3u, 4u, 5u }, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo(
        { 1u, 2u, 3u, 4u, 5u }, armnn::DataType::Float32);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ActivationQueueDescriptor descriptor;
    armnn::WorkloadInfo workloadInfo;
    AddInputToWorkload(descriptor, workloadInfo, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, workloadInfo, outputTensorInfo, outputHandle.get());

    descriptor.m_Parameters.m_Function = armnn::ActivationFunction::Sqrt;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                descriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<float, 5>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SqrtTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            0.1f,  0.2f,  0.3f,  0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            1.0f,  2.0f,  3.0f,  4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return std::sqrt(value);
    };
    std::vector<float> expectedOutput(inputData.size());
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::Sqrt,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           expectedOutput);
}

LayerTestResult<float, 4> SqrtTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SqrtTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> SqrtUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SqrtTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.0625f, 64);
}

LayerTestResult<int16_t, 4> SqrtInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SqrtTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SquareTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    // Calculate output values for input.
    auto f = [](float value)
    {
        return std::pow(value,2);
    };
    std::vector<float> expectedOutput(inputData.size());
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::Square,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           expectedOutput);
}

LayerTestResult<float, 4> SquareTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SquareTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> SquareUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SquareTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                      tensorHandleFactory, 0.0625f, 64);
}

LayerTestResult<int16_t, 4> SquareInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SquareTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> TanhTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };

    const float a = 2.0f;
    const float b = 3.0f;
    // Calculate output values for input.
    auto f = [a, b](float value)
    {
        return a * tanhf(b * value);
    };
    std::vector<float> expectedOutput(inputData.size());
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::TanH,
                                           a,
                                           b,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           expectedOutput);
}

LayerTestResult<float, 4> TanhTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return TanhTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> TanhUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return TanhTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int16_t, 4> TanhInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return TanhTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> EluTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    std::vector<float> inputData = {
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.1f,  0.2f,  0.3f,  0.4f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            1.0f,  2.0f,  3.0f,  4.0f
    };


    const float a = 0.01f;
    // Calculate output values for input.
    auto f = [a](float value)
    {
        return (value >= 0) ? value : a * (expf(value) - 1);
    };
    std::vector<float> expectedOutput(inputData.size());
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::Elu,
                                           a,
                                           0.0f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           expectedOutput);
}

LayerTestResult<float, 4> EluTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return EluTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> EluUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return EluTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int16_t, 4> EluInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return EluTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> HardSwishTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset)
{
    std::vector<float> inputData = {
        -0.1f, -0.2f, -0.3f, -0.4f,
        0.1f,  0.2f,  0.3f,  0.4f,
        -1.0f, -2.0f, -3.0f, -4.0f,
        1.0f,  2.0f,  3.0f,  4.0f
    };
    // Calculate output values for input.
    auto f = [](float x)
        {
            // Break down the calculation to help with verification.
            // hard_swish(x) = x * relu6(x+3) / 6
            // relu6(x) = min(max(x,0),6)
            float reLu6_step1 = std::max((x + 3),0.0f);
            float reLu6Complete = std::min(reLu6_step1, 6.0f);
            float hardSwish_step1 = x * reLu6Complete;
            float result = hardSwish_step1 / 6;
            return result;
        };
    std::vector<float> expectedOutput(inputData.size());
    std::transform(inputData.begin(), inputData.end(), expectedOutput.begin(), f);

    return SimpleActivationTest<ArmnnType>(workloadFactory,
                                           memoryManager,
                                           tensorHandleFactory,
                                           armnn::ActivationFunction::HardSwish,
                                           0.f,
                                           0.f,
                                           qScale,
                                           qOffset,
                                           inputData,
                                           qScale,
                                           qOffset,
                                           expectedOutput);
}

LayerTestResult<float, 4> HardSwishTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return HardSwishTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}

LayerTestResult<uint8_t, 4> HardSwishUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return HardSwishTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager,
                                                          tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int16_t, 4> HardSwishInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return HardSwishTestCommon<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 0);
}


template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> CompareActivationTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::ActivationFunction f,
    unsigned int batchSize = 5,
    float qScale = 0.0f,
    int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    unsigned int width     = 17;
    unsigned int height    = 29;
    unsigned int channels  = 2;

    float a = 0.234f;
    float b = -12.345f;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    unsigned int shape[] = {batchSize, channels, height, width};

    inputTensorInfo = armnn::TensorInfo(4, shape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, shape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    float minVal = -10.f;
    if (f == armnn::ActivationFunction::Sqrt)
    {
        minVal = 0.f;
    }

    std::vector<T> input = MakeRandomTensor<T>(inputTensorInfo, 21453, minVal, 10.f);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandleRef = refTensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandleRef = refTensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ActivationQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Parameters.m_A        = a;
    data.m_Parameters.m_B        = b;
    data.m_Parameters.m_Function = f;

    armnn::ActivationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadInput(refData, refInfo, 0, inputTensorInfo, inputHandleRef.get());
    SetWorkloadOutput(refData, refInfo, 0, outputTensorInfo, outputHandleRef.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                data, info);
    ARMNN_ASSERT(workload != nullptr);
    std::unique_ptr<armnn::IWorkload> workloadRef = refWorkloadFactory.CreateWorkload(armnn::LayerType::Activation,
                                                                                      refData, refInfo);
    ARMNN_ASSERT(workloadRef != nullptr);

    inputHandle->Allocate();
    outputHandle->Allocate();
    inputHandleRef->Allocate();
    outputHandleRef->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());
    CopyDataToITensorHandle(inputHandleRef.get(), input.data());

    workload->Execute();
    workloadRef->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    CopyDataFromITensorHandle(expectedOutput.data(), outputHandleRef.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());

}

LayerTestResult<float, 4> CompareActivationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::ActivationFunction f,
    unsigned int batchSize)
{
    return CompareActivationTestImpl<armnn::DataType::Float32>(
        workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory,
        refTensorHandleFactory, f, batchSize);
}

LayerTestResult<uint8_t, 4> CompareActivationUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::IWorkloadFactory& refWorkloadFactory,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::ITensorHandleFactory& refTensorHandleFactory,
    armnn::ActivationFunction f)
{
    return CompareActivationTestImpl<armnn::DataType::QAsymmU8>(
        workloadFactory, memoryManager, refWorkloadFactory,
        tensorHandleFactory, refTensorHandleFactory, f, 5, 0.1f, 50);
}

LayerTestResult<int16_t, 4> CompareActivationInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        armnn::IWorkloadFactory& refWorkloadFactory,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::ITensorHandleFactory& refTensorHandleFactory,
        armnn::ActivationFunction f)
{
    return CompareActivationTestImpl<armnn::DataType::QSymmS16>(
            workloadFactory, memoryManager, refWorkloadFactory, tensorHandleFactory,
            refTensorHandleFactory, f, 5, 0.1f, 0);
}

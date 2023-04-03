//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnectedTestImpl.hpp"


#include <armnnUtils/QuantizeHelper.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

//
// Implementation templates
//

template<typename T, typename B>
LayerTestResult<T, 2> SimpleFullyConnectedTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::TensorInfo inputTensorInfo,
    armnn::TensorInfo outputTensorInfo,
    armnn::TensorInfo weightsTensorInfo,
    armnn::TensorInfo biasesTensorInfo,
    std::vector<T>& weights,
    std::vector<B>& bias,
    std::vector<T>& input,
    bool biasEnabled,
    bool transposeWeights,
    bool constantWeights)
{
    std::unique_ptr<armnn::ITensorHandle> input0Handle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> input1Handle = tensorHandleFactory.CreateTensorHandle(weightsTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    armnn::FullyConnectedQueueDescriptor data;
    armnn::WorkloadInfo info;

    AddInputToWorkload(data, info, inputTensorInfo, input0Handle.get());
    AddInputToWorkload(data, info, weightsTensorInfo, input1Handle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_TransposeWeightMatrix = transposeWeights;
    data.m_Parameters.m_ConstantWeights = constantWeights;

    std::unique_ptr<armnn::ITensorHandle> input2Handle = nullptr;
    if (biasEnabled)
    {
        input2Handle = tensorHandleFactory.CreateTensorHandle(biasesTensorInfo);
        AddInputToWorkload(data, info, biasesTensorInfo, input2Handle.get());
    }

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::FullyConnected,
                                                                                data,
                                                                                info);
    LayerTestResult<T, 2> result(outputTensorInfo);

    input0Handle->Allocate();
    input1Handle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(input0Handle.get(), input.data());
    CopyDataToITensorHandle(input1Handle.get(), weights.data());
    if (biasEnabled)
    {
        input2Handle->Allocate();
        CopyDataToITensorHandle(input2Handle.get(), bias.data());
    }

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());
    result.m_ActualData = actualOutput;

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> FullyConnectedTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled,
        bool constantWeights)
{
    constexpr static unsigned int inputWidth = 3u;
    constexpr static unsigned int inputHeight = 2u;
    constexpr static unsigned int inputChannels = 1u;

    constexpr static unsigned int inputSize = inputWidth * inputHeight * inputChannels;

    constexpr static unsigned int outputChannels = 2u;

    armnn::TensorInfo inputTensorInfo({ 1, inputChannels, inputHeight, inputWidth }, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(63);

    armnn::TensorInfo outputTensorInfo({ 1, outputChannels }, ArmnnType);
    outputTensorInfo.SetQuantizationScale(5.f);
    outputTensorInfo.SetQuantizationOffset(biasEnabled ? -50 : 10);

    armnn::TensorInfo weightsDesc({ outputChannels, inputSize }, ArmnnType);
    weightsDesc.SetQuantizationScale(0.2f);
    weightsDesc.SetQuantizationOffset(93);

    armnn::TensorInfo biasesDesc({ outputChannels }, GetBiasTypeFromWeightsType(weightsDesc.GetDataType()).value());
    biasesDesc.SetQuantizationScale(inputTensorInfo.GetQuantizationScale() * weightsDesc.GetQuantizationScale());
    biasesDesc.SetQuantizationOffset(0);

    LayerTestResult<T, 2> result(outputTensorInfo);

    std::vector<T> input = ConvertToDataType<ArmnnType>(
        {
            -1.2f, 6.1f, -3.5f,
            18.8f, -5.5f, 2.9f
        },
        inputTensorInfo);

    std::vector<T> weights = ConvertToDataType<ArmnnType>(
        {
            -8.4f, 20.0f, -10.4f, -8, 16.4f, -11.8f,
            23.4f, 10.4f, -14.0f, -3.8f, -11.8f, 11.4f
        },
        weightsDesc);

    std::vector<int32_t> bias = {9250, 67500};

    result = SimpleFullyConnectedTestImpl<T>(workloadFactory,
                                             memoryManager,
                                             tensorHandleFactory,
                                             inputTensorInfo,
                                             outputTensorInfo,
                                             weightsDesc,
                                             biasesDesc,
                                             weights,
                                             bias,
                                             input,
                                             biasEnabled,
                                             true,
                                             constantWeights);

    if (biasEnabled)
    {
        result.m_ExpectedData = ConvertToDataType<ArmnnType>({80.f, 1460.f}, outputTensorInfo);
    }
    else
    {
        result.m_ExpectedData = ConvertToDataType<ArmnnType>({-107.04f, 110.f}, outputTensorInfo);
    }

    return result;
}

//
// ArmNN variant of the AndroidNN fully_connected_float_large test.
//
// Tests the fully connected layer with large values, optionally transposing weights.
// Note this is templated for consistency, but the nature of this tests makes it unlikely to be useful in Uint8 mode.
//
template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> FullyConnectedLargeTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool transposeWeights,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    unsigned int inputWidth = 1;
    unsigned int inputHeight = 1;
    unsigned int inputChannels = 5;
    unsigned int inputNum = 1;

    unsigned int outputChannels = 1;
    unsigned int outputNum = 1;

    // Define the tensor descriptors.
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo weightsDesc;
    armnn::TensorInfo biasesDesc;

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels };
    unsigned int weightsShape[] = { inputChannels, outputChannels };
    if (transposeWeights)
    {
        std::swap(weightsShape[0], weightsShape[1]);
    }

    unsigned int biasShape[] = { outputChannels };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, ArmnnType);
    weightsDesc = armnn::TensorInfo(2, weightsShape, ArmnnType);
    biasesDesc = armnn::TensorInfo(1, biasShape, ArmnnType);

    // Set quantization parameters if the requested type is a quantized type.
    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    LayerTestResult<T, 2> result(outputTensorInfo);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
        {
            1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        },
        qScale, qOffset);

    std::vector<T> weights = armnnUtils::QuantizedVector<T>(
        {
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f
        },
        qScale, qOffset);

    std::vector<T> biasValues({900000.f});

    result = SimpleFullyConnectedTestImpl<T>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo, outputTensorInfo,
        weightsDesc, biasesDesc,
        weights, biasValues, input,
        true, transposeWeights, true
    );

    result.m_ExpectedData = armnnUtils::QuantizedVector<T>({ 965432.0f }, qScale, qOffset);

    return result;
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
FullyConnectedTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    bool constWeights);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
FullyConnectedTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    bool constWeights);

//
// Implementation functions
//

LayerTestResult<float, 2> FullyConnectedFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled,
    bool transposeWeights)
{
    unsigned int inputWidth = 1;
    unsigned int inputHeight = 1;
    unsigned int inputChannels = 5;
    unsigned int inputNum = 2;

    unsigned int outputChannels = 3;
    unsigned int outputNum = 2;

    // Define the tensor descriptors.
    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;
    armnn::TensorInfo weightsDesc;
    armnn::TensorInfo biasesDesc;

    unsigned int inputShape[]   = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[]  = { outputNum, outputChannels };
    unsigned int weightsShape[] = { inputChannels, outputChannels };

    if (transposeWeights)
    {
        std::swap(weightsShape[0], weightsShape[1]);
    }

    unsigned int biasShape[] = { outputChannels };

    inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
    outputTensorInfo = armnn::TensorInfo(2, outputShape, armnn::DataType::Float32);
    weightsDesc = armnn::TensorInfo(2, weightsShape, armnn::DataType::Float32);
    biasesDesc = armnn::TensorInfo(1, biasShape, armnn::DataType::Float32);

    LayerTestResult<float, 2> result(outputTensorInfo);

    std::vector<float> input =
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        5.0f, 4.0f, 3.0f, 2.0f, 1.0f
    };

    std::vector<float> weights =
    {
        .5f, 2.f, .5f,
        .5f, 2.f, 1.f,
        .5f, 2.f, 2.f,
        .5f, 2.f, 3.f,
        .5f, 2.f, 4.f
    };

    if (transposeWeights)
    {
        weights =
        {
            .5f, .5f, .5f, .5f, .5f,
            2.f, 2.f, 2.f, 2.f, 2.f,
            .5f, 1.f, 2.f, 3.f, 4.f
        };
    }

    std::vector<float> biasValues({0.f, 0.f, 0.f});
    if (biasEnabled)
    {
        biasValues = std::vector<float>({10.f, 20.f, 30.f});
    }

    result = SimpleFullyConnectedTestImpl<float>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo, outputTensorInfo,
        weightsDesc, biasesDesc,
        weights, biasValues, input,
        biasEnabled, transposeWeights, true
    );

    std::vector<float> expectedOutput =
    {
        0.5f + 1.0f + 1.5f + 2.0f + 2.5f + biasValues[0],
        2.0f + 4.0f + 6.0f + 8.0f + 10.f + biasValues[1],
        0.5f + 2.0f + 6.0f + 12.f + 20.f + biasValues[2],

        2.5f + 2.0f + 1.5f + 1.0f + 0.5f + biasValues[0],
        10.0f + 8.0f + 6.0f + 4.0f + 2.f + biasValues[1],
        2.5f + 4.0f + 6.0f + 6.f + 4.f   + biasValues[2]
    };
    result.m_ExpectedData = expectedOutput;

    return result;
}

LayerTestResult<float, 2> FullyConnectedLargeTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool transposeWeights)
{
    return FullyConnectedLargeTestCommon<armnn::DataType::Float32>(workloadFactory,
                                                                   memoryManager,
                                                                   tensorHandleFactory,
                                                                   transposeWeights);
}

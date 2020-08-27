//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FullyConnectedTestImpl.hpp"


#include <QuantizeHelper.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>

#include <backendsCommon/test/DataTypeUtils.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

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
        armnn::TensorInfo weightsDesc,
        armnn::TensorInfo biasesDesc,
        boost::multi_array<T, 2>& weights,
        boost::multi_array<B, 1>& bias,
        boost::multi_array<T, 4>& input,
        bool biasEnabled,
        bool transposeWeights)
{
    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::FullyConnectedQueueDescriptor data;
    armnn::WorkloadInfo info;
    armnn::ScopedCpuTensorHandle weightsTensor(weightsDesc);
    armnn::ScopedCpuTensorHandle biasTensor(biasesDesc);

    AllocateAndCopyDataToITensorHandle(&weightsTensor, &weights[0][0]);
    AllocateAndCopyDataToITensorHandle(&biasTensor, &bias[0]);

    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());
    data.m_Weight = &weightsTensor;
    data.m_Bias = &biasTensor;
    data.m_Parameters.m_BiasEnabled = biasEnabled;
    data.m_Parameters.m_TransposeWeightMatrix = transposeWeights;

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFullyConnected(data, info);
    LayerTestResult<T, 2> result(outputTensorInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();
    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> FullyConnectedTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        bool biasEnabled)
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

    auto input = MakeTensor<T, 4>(inputTensorInfo, ConvertToDataType<ArmnnType>(
        {
            -1.2f, 6.1f, -3.5f,
            18.8f, -5.5f, 2.9f
        },
        inputTensorInfo));

    auto weights = MakeTensor<T, 2>(weightsDesc, ConvertToDataType<ArmnnType>(
        {
            -8.4f, 20.0f, -10.4f, -8, 16.4f, -11.8f,
            23.4f, 10.4f, -14.0f, -3.8f, -11.8f, 11.4f
        },
        weightsDesc));

    auto bias = MakeTensor<int32_t, 1>(biasesDesc, std::vector<int32_t>{9250, 67500});

    result = SimpleFullyConnectedTestImpl<T>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            inputTensorInfo, outputTensorInfo,
            weightsDesc, biasesDesc,
            weights, bias, input,
            biasEnabled, true
    );

    if (biasEnabled)
    {
        result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                                 ConvertToDataType<ArmnnType>({80.f, 1460.f}, outputTensorInfo));
    }
    else
    {
        result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                                 ConvertToDataType<ArmnnType>({-107.04f, 110.f}, outputTensorInfo));
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
    float qScale = 0.0f,
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

    boost::multi_array<T, 4> input = MakeTensor<T, 4>(inputTensorInfo,
        armnnUtils::QuantizedVector<T>({
            1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        },
        qScale, qOffset)
    );

    boost::multi_array<T, 2> weights = MakeTensor<T, 2>(weightsDesc,
        armnnUtils::QuantizedVector<T>({
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f
        },
        qScale, qOffset)
    );

    std::vector<T> biasValues({900000.f});
    boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasesDesc, biasValues);

    result = SimpleFullyConnectedTestImpl<T>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo, outputTensorInfo,
        weightsDesc, biasesDesc,
        weights, bias, input,
        true, transposeWeights
    );

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
                                             armnnUtils::QuantizedVector<T>({ 965432.0f }, qScale, qOffset));

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
    bool biasEnabled);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
FullyConnectedTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    bool biasEnabled);

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

    boost::multi_array<float, 4> input = MakeTensor<float, 4>(inputTensorInfo, std::vector<float>(
        {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f,

            5.0f, 4.0f, 3.0f, 2.0f, 1.0f
        })
    );

    boost::multi_array<float, 2> weights = MakeTensor<float, 2>(weightsDesc, std::vector<float>(
        {
            .5f, 2.f, .5f,
            .5f, 2.f, 1.f,
            .5f, 2.f, 2.f,
            .5f, 2.f, 3.f,
            .5f, 2.f, 4.f
        }));

    if (transposeWeights)
    {
        weights = MakeTensor<float, 2>(weightsDesc, std::vector<float>(
        {
            .5f, .5f, .5f, .5f, .5f,
            2.f, 2.f, 2.f, 2.f, 2.f,
            .5f, 1.f, 2.f, 3.f, 4.f
        }));
    }


    std::vector<float> biasValues({0.f, 0.f, 0.f});
    if (biasEnabled)
    {
        biasValues =  std::vector<float>({10.f, 20.f, 30.f});
    }
    boost::multi_array<float, 1> bias = MakeTensor<float, 1>(biasesDesc, biasValues);

    result = SimpleFullyConnectedTestImpl<float>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo, outputTensorInfo,
        weightsDesc, biasesDesc,
        weights, bias, input,
        biasEnabled, transposeWeights
    );

    result.outputExpected = MakeTensor<float, 2>(outputTensorInfo, std::vector<float>(
        {
            0.5f + 1.0f + 1.5f + 2.0f + 2.5f + biasValues[0],
            2.0f + 4.0f + 6.0f + 8.0f + 10.f + biasValues[1],
            0.5f + 2.0f + 6.0f + 12.f + 20.f + biasValues[2],

            2.5f + 2.0f + 1.5f + 1.0f + 0.5f + biasValues[0],
            10.0f + 8.0f + 6.0f + 4.0f + 2.f + biasValues[1],
            2.5f + 4.0f + 6.0f + 6.f + 4.f   + biasValues[2]
        })
    );

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

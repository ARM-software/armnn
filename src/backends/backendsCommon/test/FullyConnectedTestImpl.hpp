//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>
#include "WorkloadTestUtils.hpp"
#include <backendsCommon/IBackendInternal.hpp>

LayerTestResult<float, 2> FullyConnectedFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
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

    unsigned int inputShape[] = { inputNum, inputChannels, inputHeight, inputWidth };
    unsigned int outputShape[] = { outputNum, outputChannels };
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
        QuantizedVector<T>(qScale, qOffset, {
            1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f,
        })
    );

    boost::multi_array<T, 2> weights = MakeTensor<T, 2>(weightsDesc,
        QuantizedVector<T>(qScale, qOffset, {
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f
        })
    );

    std::vector<T> biasValues({900000.f});
    boost::multi_array<T, 1> bias = MakeTensor<T, 1>(biasesDesc, biasValues);

    result = SimpleFullyConnectedTestImpl<T>(
        workloadFactory,
        memoryManager,
        inputTensorInfo, outputTensorInfo,
        weightsDesc, biasesDesc,
        weights, bias, input,
        true, transposeWeights
    );

    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo,
        QuantizedVector<T>(qScale, qOffset, {
            965432.0f,
        })
    );

    return result;
}

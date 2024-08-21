//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateMeanNetwork(const armnn::TensorShape& inputShape,
                                     const armnn::TensorShape& outputShape,
                                     const armnn::MeanDescriptor& descriptor,
                                     const float qScale = 1.0f,
                                     const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape,   DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);

    IConnectableLayer* mean   = network->AddMeanLayer(descriptor, "mean");
    IConnectableLayer* input  = network->AddInputLayer(0, "input");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, mean,   inputTensorInfo,  0, 0);
    Connect(mean,  output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void MeanEndToEnd(const std::vector<armnn::BackendId>& backends, bool keepDims = false)
{
    using namespace armnn;

    MeanDescriptor descriptor;
    descriptor.m_KeepDims = keepDims;
    descriptor.m_Axis     = { 0 };

    TensorShape inputShape  = { 2, 3 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        outputShape[descriptor.m_Axis[0]] = 1;
    }
    else
    {
        outputShape = { 3 };
    }

    INetworkPtr network = CreateMeanNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<float> floatOutputData = { 2.5f, 3.5f, 4.5f };

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void MeanEndToEndResNetV2(const std::vector<armnn::BackendId>& backends, bool keepDims = false)
{
    using namespace armnn;

    MeanDescriptor descriptor;
    descriptor.m_KeepDims = keepDims;
    descriptor.m_Axis     = { 1, 2 };

    TensorShape inputShape  = { 1, 7, 7, 2048 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        for (const auto axis : descriptor.m_Axis)
        {
            outputShape[axis] = 1;
        }
    }
    else
    {
        outputShape = { 1, 2048 };
    }

    INetworkPtr network = CreateMeanNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> dim_1 = { -3, -2, -1, 0, 1, 2, 3, 4 };

    std::vector<float> floatInputData;

    constexpr std::size_t copies = 7 * 7 * 256;
    floatInputData.reserve(dim_1.size() * copies); // { 1, 7, 7, (8 * 256) } -> { 1, 7, 7, 2048 }

    for (std::size_t i = 0; i != copies; ++i)
    {
        std::copy(dim_1.begin(), dim_1.end(), std::back_inserter(floatInputData));
    }

    std::vector<float> floatOutputData(floatInputData.begin(), floatInputData.begin() + 2048);

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}
} // anonymous namespace

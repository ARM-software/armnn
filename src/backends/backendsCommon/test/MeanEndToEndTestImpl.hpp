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

    TensorShape inputShape  = { 6 };
    TensorShape outputShape = inputShape;

    if (keepDims)
    {
        outputShape[descriptor.m_Axis[0]] = 1;
    }
    else
    {
        outputShape = { 1 };
    }

    INetworkPtr network = CreateMeanNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData =
    {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f
    };

    std::vector<float> floatOutputData = { 3.5f };

    std::vector<T> inputData  = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData    = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}
} // anonymous namespace

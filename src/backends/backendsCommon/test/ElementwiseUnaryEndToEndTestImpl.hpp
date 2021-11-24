//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <CommonTestUtils.hpp>

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

template<armnn::DataType ArmnnTypeInput>
INetworkPtr CreateElementwiseUnaryNetwork(const TensorShape& inputShape,
                                          const TensorShape& outputShape,
                                          UnaryOperation operation,
                                          const float qScale = 1.0f,
                                          const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr net(INetwork::Create());

    ElementwiseUnaryDescriptor descriptor(operation);
    IConnectableLayer* elementwiseUnaryLayer = net->AddElementwiseUnaryLayer(descriptor, "elementwiseUnary");

    TensorInfo inputTensorInfo(inputShape, ArmnnTypeInput, qScale, qOffset, true);
    IConnectableLayer* input = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(0));
    Connect(input, elementwiseUnaryLayer, inputTensorInfo, 0, 0);

    TensorInfo outputTensorInfo(outputShape, ArmnnTypeInput, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(elementwiseUnaryLayer, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnInType,
         typename TInput = armnn::ResolveType<ArmnnInType>>
void ElementwiseUnarySimpleEndToEnd(const std::vector<BackendId>& backends,
                                    UnaryOperation operation,
                                    const std::vector<float> expectedOutput)
{
    using namespace armnn;

    const float   qScale  = IsQuantizedType<TInput>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<TInput>() ? 50    : 0;

    const TensorShape& inputShape  = { 2, 2, 2, 2 };
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateElementwiseUnaryNetwork<ArmnnInType>(inputShape, outputShape, operation, qScale, qOffset);

    CHECK(net);

    const std::vector<float> input({ 1, -1, 1, 1,  5, -5, 5, 5,
                                       -3, 3, 3, 3,  4, 4, -4, 4 });

    // quantize data
    std::vector<TInput> qInputData      = armnnUtils::QuantizedVector<TInput>(input, qScale, qOffset);
    std::vector<TInput> qExpectedOutput = armnnUtils::QuantizedVector<TInput>(expectedOutput, qScale, qOffset);

    std::map<int, std::vector<TInput>> inputTensorData    = {{ 0, qInputData }};
    std::map<int, std::vector<TInput>> expectedOutputData = {{ 0, qExpectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, ArmnnInType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

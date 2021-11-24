//
// Copyright © 2017 Arm Ltd. All rights reserved.
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
INetworkPtr CreateComparisonNetwork(const std::vector<TensorShape>& inputShapes,
                                    const TensorShape& outputShape,
                                    ComparisonOperation operation,
                                    const float qScale = 1.0f,
                                    const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr net(INetwork::Create());

    ComparisonDescriptor descriptor(operation);
    IConnectableLayer* comparisonLayer = net->AddComparisonLayer(descriptor, "comparison");

    for (unsigned int i = 0; i < inputShapes.size(); ++i)
    {
        TensorInfo inputTensorInfo(inputShapes[i], ArmnnTypeInput, qScale, qOffset, true);
        IConnectableLayer* input = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(i));
        Connect(input, comparisonLayer, inputTensorInfo, 0, i);
    }

    TensorInfo outputTensorInfo(outputShape, DataType::Boolean, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(comparisonLayer, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnInType,
         typename TInput = armnn::ResolveType<ArmnnInType>>
void ComparisonSimpleEndToEnd(const std::vector<BackendId>& backends,
                              ComparisonOperation operation,
                              const std::vector<uint8_t> expectedOutput)
{
    using namespace armnn;

    const std::vector<TensorShape> inputShapes{{ 2, 2, 2, 2 }, { 2, 2, 2, 2 }};
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateComparisonNetwork<ArmnnInType>(inputShapes, outputShape, operation);

    CHECK(net);

    const std::vector<TInput> input0({ 1, 1, 1, 1,  5, 5, 5, 5,
                                       3, 3, 3, 3,  4, 4, 4, 4 });

    const std::vector<TInput> input1({ 1, 1, 1, 1,  3, 3, 3, 3,
                                       5, 5, 5, 5,  4, 4, 4, 4 });

    std::map<int, std::vector<TInput>>  inputTensorData    = {{ 0, input0 }, { 1, input1 }};
    std::map<int, std::vector<uint8_t>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, DataType::Boolean>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnInType,
         typename TInput = armnn::ResolveType<ArmnnInType>>
void ComparisonBroadcastEndToEnd(const std::vector<BackendId>& backends,
                                 ComparisonOperation operation,
                                 const std::vector<uint8_t> expectedOutput)
{
    using namespace armnn;

    const std::vector<TensorShape> inputShapes{{ 1, 2, 2, 3 }, { 1, 1, 1, 3 }};
    const TensorShape& outputShape = { 1, 2, 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateComparisonNetwork<ArmnnInType>(inputShapes, outputShape, operation);

    const std::vector<TInput> input0({ 1, 2, 3, 1, 0, 6,
                                       7, 8, 9, 10, 11, 12 });

    const std::vector<TInput> input1({ 1, 1, 3 });

    std::map<int, std::vector<TInput>>  inputTensorData    = {{ 0, input0 }, { 1, input1 }};
    std::map<int, std::vector<uint8_t>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnInType, DataType::Boolean>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

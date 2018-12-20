//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/INetwork.hpp>

#include <backendsCommon/test/CommonTestUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

namespace
{

template<typename armnn::DataType DataType>
INetworkPtr CreateArithmeticNetwork(const std::vector<TensorShape>& inputShapes,
                                    const TensorShape& outputShape,
                                    const LayerType type,
                                    const float qScale = 1.0f,
                                    const int32_t qOffset = 0)
{
    using namespace armnn;

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* arithmeticLayer = nullptr;

    switch(type){
        case LayerType::Equal: arithmeticLayer = net->AddEqualLayer("equal"); break;
        case LayerType::Greater: arithmeticLayer = net->AddGreaterLayer("greater"); break;
        default: BOOST_TEST_FAIL("Non-Arithmetic layer type called.");
    }

    for (unsigned int i = 0; i < inputShapes.size(); ++i)
    {
        TensorInfo inputTensorInfo(inputShapes[i], DataType, qScale, qOffset);
        IConnectableLayer* input = net->AddInputLayer(boost::numeric_cast<LayerBindingId>(i));
        Connect(input, arithmeticLayer, inputTensorInfo, 0, i);
    }

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(arithmeticLayer, output, outputTensorInfo, 0, 0);

    return net;
}

template<typename T>
void ArithmeticSimpleEndToEnd(const std::vector<BackendId>& backends,
                              const LayerType type,
                              const std::vector<T> expectedOutput)
{
    using namespace armnn;

    const std::vector<TensorShape> inputShapes{{ 2, 2, 2, 2 }, { 2, 2, 2, 2 }};
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateArithmeticNetwork<GetDataType<T>()>(inputShapes, outputShape, type);

    BOOST_TEST_CHECKPOINT("create a network");

    const std::vector<T> input0({ 1, 1, 1, 1,  5, 5, 5, 5,
                                  3, 3, 3, 3,  4, 4, 4, 4 });

    const std::vector<T> input1({ 1, 1, 1, 1,  3, 3, 3, 3,
                                  5, 5, 5, 5,  4, 4, 4, 4 });

    std::map<int, std::vector<T>> inputTensorData = {{ 0, input0 }, { 1, input1 }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<T>(move(net), inputTensorData, expectedOutputData, backends);
}

template<typename T>
void ArithmeticBroadcastEndToEnd(const std::vector<BackendId>& backends,
                                 const LayerType type,
                                 const std::vector<T> expectedOutput)
{
    using namespace armnn;

    const std::vector<TensorShape> inputShapes{{ 1, 2, 2, 3 }, { 1, 1, 1, 3 }};
    const TensorShape& outputShape = { 1, 2, 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateArithmeticNetwork<GetDataType<T>()>(inputShapes, outputShape, type);

    BOOST_TEST_CHECKPOINT("create a network");

    const std::vector<T> input0({ 1, 2, 3, 1, 0, 6,
                                  7, 8, 9, 10, 11, 12 });

    const std::vector<T> input1({ 1, 1, 3 });

    std::map<int, std::vector<T>> inputTensorData = {{ 0, input0 }, { 1, input1 }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<T>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

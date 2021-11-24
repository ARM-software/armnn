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

template<typename armnn::DataType DataType>
INetworkPtr CreateConcatNetwork(const std::vector<TensorShape>& inputShapes,
                                const TensorShape &outputShape,
                                unsigned int concatAxis,
                                const float qScale = 1.0f,
                                const int32_t qOffset = 0)
{
    using namespace armnn;
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    OriginsDescriptor descriptor;

    descriptor = CreateDescriptorForConcatenation(inputShapes.begin(),
                                                  inputShapes.end(),
                                                  concatAxis);
    IConnectableLayer* concat = net->AddConcatLayer(descriptor, "concat");

    for (unsigned int i = 0; i < inputShapes.size(); ++i)
    {
        TensorInfo inputTensorInfo(inputShapes[i], DataType, qScale, qOffset, true);
        IConnectableLayer* input = net->AddInputLayer(armnn::numeric_cast<LayerBindingId>(i));
        Connect(input, concat, inputTensorInfo, 0, i);
    }

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
    Connect(concat, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType>
void ConcatDim0EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int concatAxis = 0;
    const std::vector<TensorShape> inputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};
    const TensorShape& outputShape = { 4, 3, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateConcatNetwork<ArmnnType>(inputShapes, outputShape, concatAxis);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<T> expectedOutput{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }, { 1,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0,expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void ConcatDim1EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int concatAxis = 1;
    const std::vector<TensorShape> inputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};
    const TensorShape& outputShape = { 2, 6, 2, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateConcatNetwork<ArmnnType>(inputShapes, outputShape, concatAxis);

    // Creates structures for input & output.
    std::vector<T> inputData{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<T> expectedOutput{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }, { 1,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0,expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void ConcatDim2EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int concatAxis = 2;
    const std::vector<TensorShape> inputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};
    const TensorShape& outputShape = { 2, 3, 4, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateConcatNetwork<ArmnnType>(inputShapes, outputShape, concatAxis);

    // Creates structures for input & output.
    std::vector<T> inputData{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<T> expectedOutput{
        1, 2,
        3, 4,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        9, 10,
        11, 12
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }, { 1,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0,expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ConcatDim3EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;

    unsigned int concatAxis = 3;
    const std::vector<TensorShape> inputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};
    const TensorShape& outputShape = { 2, 3, 2, 4 };

    // Builds up the structure of the network
    INetworkPtr net = CreateConcatNetwork<ArmnnType>(inputShapes, outputShape, concatAxis);

    // Creates structures for input & output.
    std::vector<T> inputData{
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12,
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
        11, 12
    };

    std::vector<T> expectedOutput{
        1, 2,
        1, 2,
        3, 4,
        3, 4,
        5, 6,
        5, 6,
        7, 8,
        7, 8,
        9, 10,
        9, 10,
        11, 12,
        11, 12,
        1, 2,
        1, 2,
        3, 4,
        3, 4,
        5, 6,
        5, 6,
        7, 8,
        7, 8,
        9, 10,
        9, 10,
        11, 12,
        11, 12
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }, { 1,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0,expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

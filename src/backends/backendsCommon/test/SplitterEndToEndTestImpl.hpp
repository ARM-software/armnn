//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

template<typename armnn::DataType DataType>
INetworkPtr CreateSplitterNetwork(const TensorShape& inputShape,
                                  const std::vector<TensorShape>& outputShapes,
                                  unsigned int splitAxis,
                                  unsigned int numSplit,
                                  const float qScale = 1.0f,
                                  const int32_t qOffset = 0)
{
    using namespace armnn;
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);

    std::vector<unsigned int> splitterDimSizes(inputShape.GetNumDimensions());

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitAxis] % numSplit != 0)
    {
        throw ParseException("Number of splits must evenly divide the dimension");
    }
    splitterDimSizes[splitAxis] /= numSplit;

    SplitterDescriptor splitDesc(numSplit, inputShape.GetNumDimensions());
    for (unsigned int g = 0; g < numSplit; ++g)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDesc.SetViewSize(g, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDesc.SetViewOriginCoord(g, splitAxis, splitterDimSizes[splitAxis] * g);
    }

    IConnectableLayer* splitter = net->AddSplitterLayer(splitDesc, "splitter");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    Connect(input, splitter, inputTensorInfo, 0, 0);

    for (unsigned int i = 0; i < outputShapes.size(); ++i)
    {
        TensorInfo outputTensorInfo(outputShapes[i], DataType, qScale, qOffset);
        IConnectableLayer* output = net->AddOutputLayer(armnn::numeric_cast<LayerBindingId>(i));
        Connect(splitter, output, outputTensorInfo, i, 0);
    }

    return net;
}

template<armnn::DataType ArmnnType>
void Splitter1dEndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 0;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 4 };
    const std::vector<TensorShape> outputShapes{{ 2 }, { 2 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{ 1, 2, 3, 4 };

    std::vector<T> expectedOutput0{ 1, 2 };
    std::vector<T> expectedOutput1{ 3, 4 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 }, {1, expectedOutput1} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter2dDim0EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 0;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 4, 3 };
    const std::vector<TensorShape> outputShapes{{ 2, 3 }, { 2, 3 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12
    };

    std::vector<T> expectedOutput0{ 1, 2, 3, 4, 5, 6 };
    std::vector<T> expectedOutput1{ 7, 8, 9, 10, 11, 12 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 }, {1, expectedOutput1} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter2dDim1EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 1;
    unsigned int numSplit = 3;
    const TensorShape& inputShape = { 4, 3 };
    const std::vector<TensorShape> outputShapes{{ 4, 1 }, { 4, 1 }, { 4, 1 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12
    };

    std::vector<T> expectedOutput0{ 1, 4, 7, 10 };
    std::vector<T> expectedOutput1{ 2, 5, 8, 11 };
    std::vector<T> expectedOutput2{ 3, 6, 9, 12 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 },
                                                         { 1, expectedOutput1 },
                                                         { 2, expectedOutput2 } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter3dDim0EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 0;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 2, 4, 3 };
    const std::vector<TensorShape> outputShapes{{ 1, 4, 3 }, { 1, 4, 3 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21,
            22, 23, 24
    };

    std::vector<T> expectedOutput0{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
    };
    std::vector<T> expectedOutput1{
            13, 14, 15,
            16, 17, 18,
            19, 20, 21,
            22, 23, 24
    };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 },
                                                         { 1, expectedOutput1 } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter3dDim1EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 1;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 2, 4, 3 };
    const std::vector<TensorShape> outputShapes{{ 2, 2, 3 }, { 2, 2, 3 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21,
            22, 23, 24
    };

    std::vector<T> expectedOutput0{
            1, 2, 3,
            4, 5, 6,
            13, 14, 15,
            16, 17, 18
    };
    std::vector<T> expectedOutput1{
            7, 8, 9,
            10, 11, 12,
            19, 20, 21,
            22, 23, 24
    };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 },
                                                         { 1, expectedOutput1 } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter3dDim2EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 2;
    unsigned int numSplit = 3;
    const TensorShape& inputShape = { 2, 4, 3 };
    const std::vector<TensorShape> outputShapes{{ 2, 4, 1 }, { 2, 4, 1 }, { 2, 4, 1 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
            19, 20, 21,
            22, 23, 24
    };

    std::vector<T> expectedOutput0{ 1, 4, 7, 10, 13, 16, 19, 22 };
    std::vector<T> expectedOutput1{ 2, 5, 8, 11, 14, 17, 20, 23 };
    std::vector<T> expectedOutput2{ 3, 6, 9, 12, 15, 18, 21, 24 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput0 },
                                                         { 1, expectedOutput1 },
                                                         { 2, expectedOutput2 } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter4dDim0EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 0;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 4, 3, 2, 2 };
    const std::vector<TensorShape> outputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::vector<T> expectedOutput0{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24
    };

    std::vector<T> expectedOutput1{
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter4dDim1EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 1;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 2, 6, 2, 2 };
    const std::vector<TensorShape> outputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::vector<T> expectedOutput0{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36
    };

    std::vector<T> expectedOutput1{
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void Splitter4dDim2EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    unsigned int splitAxis = 2;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 2, 3, 4, 2 };
    const std::vector<TensorShape> outputShapes{{ 2, 3, 2, 2 }, { 2, 3, 2, 2 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::vector<T> expectedOutput0{
            1, 2,
            3, 4,
            9, 10,
            11, 12,
            17, 18,
            19, 20,
            25, 26,
            27, 28,
            33, 34,
            35, 36,
            41, 42,
            43, 44
    };

    std::vector<T> expectedOutput1{
            5, 6,
            7, 8,
            13, 14,
            15, 16,
            21, 22,
            23, 24,
            29, 30,
            31, 32,
            37, 38,
            39, 40,
            45, 46,
            47, 48
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void Splitter4dDim3EndToEnd(const std::vector<BackendId>& backends)
{
    using namespace armnn;

    unsigned int splitAxis = 3;
    unsigned int numSplit = 2;
    const TensorShape& inputShape = { 2, 3, 4, 2 };
    const std::vector<TensorShape> outputShapes{{ 2, 3, 4, 1 }, { 2, 3, 4, 1 }};

    // Builds up the structure of the network
    INetworkPtr net = CreateSplitterNetwork<ArmnnType>(inputShape, outputShapes, splitAxis, numSplit);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
            1, 2,
            3, 4,
            5, 6,
            7, 8,
            9, 10,
            11, 12,
            13, 14,
            15, 16,
            17, 18,
            19, 20,
            21, 22,
            23, 24,
            25, 26,
            27, 28,
            29, 30,
            31, 32,
            33, 34,
            35, 36,
            37, 38,
            39, 40,
            41, 42,
            43, 44,
            45, 46,
            47, 48
    };

    std::vector<T> expectedOutput0{
            1, 3, 5, 7,
            9, 11, 13, 15,
            17, 19, 21, 23,
            25, 27, 29, 31,
            33, 35, 37, 39,
            41, 43, 45, 47
    };

    std::vector<T> expectedOutput1{
            2, 4, 6, 8,
            10, 12, 14, 16,
            18, 20, 22, 24,
            26, 28, 30, 32,
            34, 36, 38, 40,
            42, 44, 46, 48
    };

    std::map<int, std::vector<T>> inputTensorData = {{ 0,inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput0 }, { 1, expectedOutput1 }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

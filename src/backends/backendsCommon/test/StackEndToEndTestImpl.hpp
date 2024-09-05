//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace {

template <typename armnn::DataType DataType>
armnn::INetworkPtr CreateStackNetwork(const std::vector<armnn::TensorShape>& inputShapes,
                                      const armnn::TensorShape& outputShape,
                                      const armnn::StackDescriptor& descriptor,
                                      float qScale = 1.0f,
                                      int32_t qOffset = 0,
                                      bool qConst = true)
{
    using namespace armnn;

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    //TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    const TensorInfo outputTensorInfo(outputShape, DataType);

    IConnectableLayer* stack = net->AddStackLayer(descriptor, "Stack");

    std::vector<armnn::IConnectableLayer*> inputs;
    inputs.reserve(descriptor.m_NumInputs);
    for(unsigned int i = 0; i < descriptor.m_NumInputs; ++i)
    {
        inputs.emplace_back(net->AddInputLayer(static_cast<LayerBindingId>(i), "input"));
    }

    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    // Connect all inputs to  layer
    std::vector<armnn::TensorInfo> inputTensorInfos;
    inputTensorInfos.reserve(descriptor.m_NumInputs);
    for (unsigned int i = 0; i < descriptor.m_NumInputs; ++i)
    {
        inputTensorInfos.emplace_back(inputShapes[i], DataType, qScale, qOffset, qConst);

        Connect(inputs[i], stack, inputTensorInfos[i], 0, i);
    }

    Connect(stack, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank0Axis0EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    const TensorShape scalarShape (Dimensionality::Scalar);

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 0;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = scalarShape;

    const std::vector<armnn::TensorShape> inputShapes{ scalarShape,
                                                       scalarShape,
                                                       scalarShape };

    const TensorShape& outputShape{ 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = 1     // shape = (scalar)
    // input_2 = 2     // shape = (scalar)
    // input_3 = 3     // shape = (scalar)
    const std::map<int, std::vector<T>> inputTensorData{ {0, {1}}, {1, {2}}, {2, {3}} };

    // output = [1, 2, 3]     // shape = (3, )
    const std::map<int, std::vector<T>> expectedOutputTensorData{ {0, {1, 2, 3} } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank1Axis0EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 0;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2 },
                                                       { 2 },
                                                       { 2 } };

    const TensorShape& outputShape{ 3, 2 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [1, 2]     // shape = (2, )
    // input_2 = [3, 4]     // shape = (2, )
    // input_3 = [5, 6]     // shape = (2, )
    std::vector<std::vector<T>> inputData{ { 1, 2 },
                                           { 3, 4 },
                                           { 5, 6 } };

    // output = [[1, 2],    // shape = (3, 2)
    //           [3, 4],
    //           [5, 6]]
    std::vector<T> expectedOutputData{ 1, 2, 3, 4, 5, 6 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank1Axis1EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 1;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2 },
                                                       { 2 },
                                                       { 2 } };

    const TensorShape& outputShape{ 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [1, 2]     // shape = (2, )
    // input_2 = [3, 4]     // shape = (2, )
    // input_3 = [5, 6]     // shape = (2, )
    std::vector<std::vector<T>> inputData{ { 1, 2 },
                                           { 3, 4 },
                                           { 5, 6 } };

    // output = [[1, 3, 5], // shape = (2, 3)
    //           [2, 4, 6]]
    std::vector<T> expectedOutputData{ 1, 3, 5, 2, 4, 6 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank2Axis0EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 0;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 3 },
                                                       { 2, 3 },
                                                       { 2, 3 } };

    const TensorShape& outputShape{ 3, 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[1, 2, 3],     // shape = (2, 3)
    //            [4, 5, 6]]
    // input_2 = [[7, 8, 9],     // shape = (2, 3)
    //            [10, 11, 12]]
    // input_3 = [[13, 14, 15],  // shape = (2, 3)
    //            [16, 17, 18]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6 },
                                           {  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18 } };

    // output = [[[1, 2, 3],     // shape = (3, 2, 3)
    //            [4, 5, 6]],
    //           [[7, 8, 9],
    //            [10, 11, 12]],
    //           [[13, 14, 15],
    //            [16, 17, 18]]]
    std::vector<T> expectedOutputData{ 1,  2,  3,  4,  5,  6,
                                       7,  8,  9, 10, 11, 12,
                                      13, 14, 15, 16, 17, 18 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank2Axis1EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 1;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 3 },
                                                       { 2, 3 },
                                                       { 2, 3 } };

    const TensorShape& outputShape{ 2, 3, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[1, 2, 3],     // shape = (2, 3)
    //            [4, 5, 6]]
    // input_2 = [[7, 8, 9],     // shape = (2, 3)
    //            [10, 11, 12]]
    // input_3 = [[13, 14, 15],  // shape = (2, 3)
    //            [16, 17, 18]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6 },
                                           {  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18 } };

    // output = [[[1, 2, 3],     // shape = (2, 3, 3)
    //            [7, 8, 9],
    //            [13, 14, 15]],
    //           [[4, 5, 6],
    //            [10, 11, 12],
    //            [16, 17, 18]]]
    std::vector<T> expectedOutputData{ 1, 2, 3,  7,  8,  9, 13, 14, 15,
                                       4, 5, 6, 10, 11, 12, 16, 17, 18 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank2Axis2EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 2;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 3 },
                                                       { 2, 3 },
                                                       { 2, 3 } };

    const TensorShape& outputShape{ 2, 3, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[1, 2, 3],     // shape = (2, 3)
    //            [4, 5, 6]]
    // input_2 = [[7, 8, 9],     // shape = (2, 3)
    //            [10, 11, 12]]
    // input_3 = [[13, 14, 15],  // shape = (2, 3)
    //            [16, 17, 18]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6 },
                                           {  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18 } };

    // output = [[[1, 7, 13],     // shape = (2, 3, 3)
    //            [2, 8, 14],
    //            [3, 9, 15]],
    //           [[4, 10, 16],
    //            [5, 11, 17],
    //            [6, 12, 18]]]
    std::vector<T> expectedOutputData{ 1,  7, 13, 2,  8, 14, 3,  9, 15,
                                       4, 10, 16, 5, 11, 17, 6, 12, 18 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank3Axis0EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 0;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 2, 3 },
                                                       { 2, 2, 3 },
                                                       { 2, 2, 3 } };

    const TensorShape& outputShape{ 3, 2, 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[[ 1,  2,  3],    // shape = (2, 2, 3)
    //             [ 4,  5,  6]],
    //            [[ 7,  8,  9],
    //             [10, 11, 12]]]
    // input_2 = [[[13, 14, 15],    // shape = (2, 2, 3)
    //             [16, 17, 18]],
    //            [[19, 20, 21],
    //             [22, 23, 24]]]
    // input_3 = [[[25, 26, 27],    // shape = (2, 2, 3)
    //             [28, 29, 30]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
                                           { 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 } };

    // output = [[[[ 1,  2,  3],    // shape = (3, 2, 2, 3)
    //             [ 4,  5,  6]],
    //            [[ 7,  8,  9],
    //             [10, 11, 12]]],
    //           [[[13, 14, 15],
    //             [16, 17, 18]],
    //            [[19, 20, 21],
    //             [22, 23, 24]]],
    //           [[[25, 26, 27],
    //             [28, 29, 30]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]]
    std::vector<T> expectedOutputData{ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                      25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank3Axis1EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 1;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 2, 3 },
                                                       { 2, 2, 3 },
                                                       { 2, 2, 3 } };

    const TensorShape& outputShape{ 2, 3, 2, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[[ 1,  2,  3],    // shape = (2, 2, 3)
    //             [ 4,  5,  6]],
    //            [[ 7,  8,  9],
    //             [10, 11, 12]]]
    // input_2 = [[[13, 14, 15],    // shape = (2, 2, 3)
    //             [16, 17, 18]],
    //            [[19, 20, 21],
    //             [22, 23, 24]]]
    // input_3 = [[[25, 26, 27],    // shape = (2, 2, 3)
    //             [28, 29, 30]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
                                           { 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 } };

    // output = [[[[ 1,  2,  3],    // shape = (2, 3, 2, 3)
    //             [ 4,  5,  6]],
    //            [[13, 14, 15],
    //             [16, 17, 18]],
    //            [[25, 26, 27],
    //             [28, 29, 30]]],
    //           [[[ 7,  8,  9],
    //             [10, 11, 12]],
    //            [[19, 20, 21],
    //             [22, 23, 24]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]]
    std::vector<T> expectedOutputData{ 1, 2, 3,  4,  5,  6, 13, 14, 15, 16, 17, 18, 25, 26, 27, 28, 29, 30,
                                       7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank3Axis2EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 2;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 2, 3 },
                                                       { 2, 2, 3 },
                                                       { 2, 2, 3 } };

    const TensorShape& outputShape{ 2, 2, 3, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[[ 1,  2,  3],    // shape = (2, 2, 3)
    //             [ 4,  5,  6]],
    //            [[ 7,  8,  9],
    //             [10, 11, 12]]]
    // input_2 = [[[13, 14, 15],    // shape = (2, 2, 3)
    //             [16, 17, 18]],
    //            [[19, 20, 21],
    //             [22, 23, 24]]]
    // input_3 = [[[25, 26, 27],    // shape = (2, 2, 3)
    //             [28, 29, 30]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
                                           { 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 } };

    // output = [[[[ 1,  2,  3],    // shape = (2, 2, 3, 3)
    //             [13, 14, 15],
    //             [25, 26, 27]],
    //            [[ 4,  5,  6],
    //             [16, 17, 18],
    //             [28, 29, 30]]],
    //           [[[ 7,  8,  9],
    //             [19, 20, 21],
    //             [31, 32, 33]],
    //            [[10, 11, 12],
    //             [22, 23, 24],
    //             [34, 35, 36]]]]
    std::vector<T> expectedOutputData{ 1, 2, 3, 13, 14, 15, 25, 26, 27,  4,  5,  6, 16, 17, 18, 28, 29, 30,
                                       7, 8, 9, 19, 20, 21, 31, 32, 33, 10, 11, 12, 22, 23, 24, 34, 35, 36 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void StackRank3Axis3EndToEndTest(const std::vector<armnn::BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    StackDescriptor stackDesc;
    stackDesc.m_Axis = 3;
    stackDesc.m_NumInputs = 3;
    stackDesc.m_InputShape = armnn::TensorShape( { 2, 2, 3 } );

    const std::vector<armnn::TensorShape> inputShapes{ { 2, 2, 3 },
                                                       { 2, 2, 3 },
                                                       { 2, 2, 3 } };

    const TensorShape& outputShape{ 2, 2, 3, 3 };

    // Builds up the structure of the network
    INetworkPtr net = CreateStackNetwork<ArmnnType>(inputShapes,
                                                    outputShape,
                                                    stackDesc,
                                                    qScale,
                                                    qOffset,
                                                    qConst);

    CHECK(net);

    // input_1 = [[[ 1,  2,  3],    // shape = (2, 2, 3)
    //             [ 4,  5,  6]],
    //            [[ 7,  8,  9],
    //             [10, 11, 12]]]
    // input_2 = [[[13, 14, 15],    // shape = (2, 2, 3)
    //             [16, 17, 18]],
    //            [[19, 20, 21],
    //             [22, 23, 24]]]
    // input_3 = [[[25, 26, 27],    // shape = (2, 2, 3)
    //             [28, 29, 30]],
    //            [[31, 32, 33],
    //             [34, 35, 36]]]
    std::vector<std::vector<T>> inputData{ {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12 },
                                           { 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
                                           { 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 } };

    // output = [[[[ 1, 13, 25],    // shape = (2, 2, 3, 3)
    //             [ 2, 14, 26],
    //             [ 3, 15, 27]],
    //            [[ 4, 16, 28],
    //             [ 5, 17, 29],
    //             [ 6, 18, 30]]],
    //           [[[ 7, 19, 31],
    //             [ 8, 20, 32],
    //             [ 9, 21, 33]],
    //            [[10, 22, 34],
    //             [11, 23, 35],
    //             [12, 24, 36]]]]
    std::vector<T> expectedOutputData{ 1, 13, 25, 2, 14, 26, 3, 15, 27,  4, 16, 28,  5, 17, 29,  6, 18, 30,
                                       7, 19, 31, 8, 20, 32, 9, 21, 33, 10, 22, 34, 11, 23, 35, 12, 24, 36 };

    std::map<int, std::vector<T>> inputTensorData{ {0, inputData[0]}, {1, inputData[1]}, {2, inputData[2]} };
    std::map<int, std::vector<T>> expectedOutputTensorData{ {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

} // anonymous namespace
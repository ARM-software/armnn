//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace {

template <typename armnn::DataType DataType>
armnn::INetworkPtr CreateSoftmaxNetwork(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& outputShape,
                                        const armnn::SoftmaxDescriptor& descriptor,
                                        const float qScale = 1.0f,
                                        const int32_t qOffset = 0)
{
    using namespace armnn;

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType);

    IConnectableLayer* Softmax = net->AddSoftmaxLayer(descriptor, "Softmax");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(input, Softmax, inputTensorInfo, 0, 0);
    Connect(Softmax, output, outputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void SoftmaxEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape  = { 2, 2 };
    const TensorShape& outputShape = { 2, 2 };

    SoftmaxDescriptor softmaxDesc;
    softmaxDesc.m_Beta = 1.0f;
    softmaxDesc.m_Axis = 1;

    // Builds up the structure of the network
    INetworkPtr net = CreateSoftmaxNetwork<ArmnnType>(inputShape,
                                                      outputShape,
                                                      softmaxDesc);

    CHECK(net);

    std::vector<T> inputData
    {
            17.0f, 16.0f, 5.0f, 14.0f
    };

    std::vector<T> expectedOutputData
    {
            0.731059f, 0.268941f, 0.000123f, 0.999877f
    };

    std::map<int, std::vector<T>> inputTensorData = { {0, inputData} };
    std::map<int, std::vector<T>> expectedOutputTensorData = { {0, expectedOutputData} };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void QSoftmax3DEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape  = { 1, 2, 2, 9 };
    const TensorShape& outputShape = { 1, 2, 2, 9 };

    // Beta & Scale Used to generate Softmax Lookup Table
    SoftmaxDescriptor softmaxDesc;
    softmaxDesc.m_Beta = 1.0f;
    softmaxDesc.m_Axis = 1;
    float qScale = 0.065034f;
    int32_t qOffset = 6;

    // Builds up the structure of the network
    INetworkPtr net = CreateSoftmaxNetwork<ArmnnType>(inputShape,
                                                      outputShape,
                                                      softmaxDesc,
                                                      qScale,
                                                      qOffset);
    CHECK(net);

    std::vector<T> inputData = { 113, 86,  1, 101, 127, 45, 41, 124, 10,
                                   1,  8,  1,  10,   1,  4,  4,  14, 10,
                                   3,  6,  1,   1,   2,  5,  7,   4, 10,
                                 113,116, 97, 100, 107, 45, 41, 124, 10};

    std::vector<T> expectedData = { 41,   7,   0,  19, 103,   0,   0,  85,   0,
                                    20,  31,  20,  36,  20,  24,  24,  46,  36,
                                    26,  31,  23,  23,  24,  29,  33,  27,  40,
                                    45,  54,  16,  19,  30,   1,   0,  91,   0};

    std::map<int, std::vector<T>> inputTensorData = { {0, inputData} };
    std::map<int, std::vector<T>> expectedTensorData = { {0, expectedData} };

    EndToEndLayerTestImpl<ArmnnType,ArmnnType>(std::move(net),
                                               inputTensorData,
                                               expectedTensorData,
                                               backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void QSoftmax1DEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape  = { 1, 9 };
    const TensorShape& outputShape = { 1, 9 };

    // Beta & Scale Used to generate Softmax Lookup Table
    SoftmaxDescriptor softmaxDesc;
    softmaxDesc.m_Beta = 1.0f;
    softmaxDesc.m_Axis = 1;
    float qScale = 0.065034f;
    int32_t qOffset = 6;

    // Builds up the structure of the network
    INetworkPtr net = CreateSoftmaxNetwork<ArmnnType>(inputShape,
                                                      outputShape,
                                                      softmaxDesc,
                                                      qScale,
                                                      qOffset);
    CHECK(net);

    std::vector<T> inputData = { 113, 86,  1, 101, 127, 45, 41, 124, 10};

    std::vector<T> expectedData = { 41,   7,   0,  19, 103,   0,   0,  85,   0};

    std::map<int, std::vector<T>> inputTensorData = { {0, inputData} };
    std::map<int, std::vector<T>> expectedTensorData = { {0, expectedData} };

    EndToEndLayerTestImpl<ArmnnType,ArmnnType>(std::move(net),
                                               inputTensorData,
                                               expectedTensorData,
                                               backends);
}

} // anonymous namespace
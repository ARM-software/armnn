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

    IConnectableLayer* Softmax = net->AddSoftmaxLayer(descriptor, "Softmax");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    Connect(input, Softmax, inputTensorInfo, 0, 0);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);
    IConnectableLayer* output = net->AddOutputLayer(0, "output");
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

} // anonymous namespace
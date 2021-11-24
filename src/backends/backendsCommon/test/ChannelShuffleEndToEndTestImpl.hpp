//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace{

armnn::INetworkPtr CreateChannelShuffleNetwork(const armnn::TensorInfo& inputInfo,
                                               const armnn::TensorInfo& outputInfo,
                                               const armnn::ChannelShuffleDescriptor& descriptor)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer = net->AddInputLayer(0);
    armnn::IConnectableLayer* channelShuffleLayer = net->AddChannelShuffleLayer(descriptor, "channelShuffle");
    armnn::IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");
    Connect(inputLayer, channelShuffleLayer, inputInfo, 0, 0);
    Connect(channelShuffleLayer, outputLayer, outputInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ChannelShuffleEndToEnd(const std::vector<BackendId>& backends)
{
    armnn::TensorInfo inputInfo({ 3,12 }, ArmnnType);
    armnn::TensorInfo outputInfo({ 3,12 }, ArmnnType);

    inputInfo.SetQuantizationScale(1.0f);
    inputInfo.SetQuantizationOffset(0);
    inputInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    // Creates structures for input & output.
    std::vector<T> inputData{
        0, 1, 2, 3,        4, 5, 6, 7,       8, 9, 10, 11,
        12, 13, 14, 15,   16, 17, 18, 19,   20, 21, 22, 23,
        24, 25, 26, 27,   28, 29, 30, 31,   32, 33, 34, 35
    };

    std::vector<T> expectedOutput{
        0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11,
        12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23,
        24, 28, 32, 25, 29, 33, 26, 30, 34, 27, 31, 35
    };
    ChannelShuffleDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumGroups = 3;

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateChannelShuffleNetwork(inputInfo, outputInfo, descriptor);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

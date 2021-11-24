//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace{

armnn::INetworkPtr CreateGatherNetwork(const armnn::TensorInfo& paramsInfo,
                                       const armnn::TensorInfo& indicesInfo,
                                       const armnn::TensorInfo& outputInfo,
                                       const std::vector<int32_t>& indicesData)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::GatherDescriptor descriptor;
    armnn::IConnectableLayer* paramsLayer = net->AddInputLayer(0);
    armnn::IConnectableLayer* indicesLayer = net->AddConstantLayer(armnn::ConstTensor(indicesInfo, indicesData));
    armnn::IConnectableLayer* gatherLayer = net->AddGatherLayer(descriptor, "gather");
    armnn::IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");
    Connect(paramsLayer, gatherLayer, paramsInfo, 0, 0);
    Connect(indicesLayer, gatherLayer, indicesInfo, 0, 1);
    Connect(gatherLayer, outputLayer, outputInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void GatherEndToEnd(const std::vector<BackendId>& backends)
{
    armnn::TensorInfo paramsInfo({ 8 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 3 }, ArmnnType);

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    // Creates structures for input & output.
    std::vector<T> paramsData{
        1, 2, 3, 4, 5, 6, 7, 8
    };

    std::vector<int32_t> indicesData{
        7, 6, 5
    };

    std::vector<T> expectedOutput{
        8, 7, 6
    };

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNetwork(paramsInfo, indicesInfo, outputInfo, indicesData);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, paramsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void GatherMultiDimEndToEnd(const std::vector<BackendId>& backends)
{
    armnn::TensorInfo paramsInfo({ 3, 2, 3}, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 3, 2, 3 }, ArmnnType);

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    // Creates structures for input & output.
    std::vector<T> paramsData{
         1,  2,  3,
         4,  5,  6,

         7,  8,  9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    };

    std::vector<int32_t> indicesData{
        1, 2, 1,
        2, 1, 0
    };

    std::vector<T> expectedOutput{
         7,  8,  9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
         7,  8,  9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18,
         7,  8,  9,
        10, 11, 12,
         1,  2,  3,
         4,  5,  6
    };

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNetwork(paramsInfo, indicesInfo, outputInfo, indicesData);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, paramsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

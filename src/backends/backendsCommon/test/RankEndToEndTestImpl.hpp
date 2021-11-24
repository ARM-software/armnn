//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>

#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

armnn::INetworkPtr CreateRankNetwork(const armnn::TensorInfo& inputTensorInfo,
                                     const armnn::TensorInfo& outputTensorInfo)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* rankLayer   = network->AddRankLayer("Rank");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, rankLayer, inputTensorInfo, 0, 0);
    Connect(rankLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void RankEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    std::vector<float> floatInputData{
         1,  2,  3,  4,  5,
        11, 12, 13, 14, 15,
        21, 22, 23, 24, 25
    };
    std::vector<T> inputData = armnnUtils::QuantizedVector<T>(floatInputData);

    std::vector<int32_t> expectedOutputData{ 4 };

    TensorInfo inputInfo ({ 1, 1, 5, 3 }, ArmnnType, 0.0f, 0, true);
    TensorShape outputShape (Dimensionality::Scalar);
    TensorInfo outputInfo(outputShape, DataType::Signed32);

    armnn::INetworkPtr network = CreateRankNetwork(inputInfo, outputInfo);

    CHECK(network);

    std::map<int, std::vector<T>> inputTensorData   = {{ 0, inputData }};
    std::map<int, std::vector<int32_t>> expectedOutputTensorData = {{ 0, expectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, DataType::Signed32>(move(network),
                                                         inputTensorData,
                                                         expectedOutputTensorData,
                                                         backends);
}

} // anonymous namespace
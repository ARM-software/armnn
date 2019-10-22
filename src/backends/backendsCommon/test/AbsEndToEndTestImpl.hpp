//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommonTestUtils.hpp"

#include <QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <armnn/ArmNN.hpp>

namespace
{

armnn::INetworkPtr CreateAbsNetwork(const armnn::TensorInfo& tensorInfo)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* absLayer    = network->AddAbsLayer("abs");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");

    Connect(inputLayer, absLayer, tensorInfo, 0, 0);
    Connect(absLayer, outputLayer, tensorInfo, 0, 0);

    return network;
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void AbsEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const float   qScale  = IsQuantizedType<T>() ? 0.25f : 1.0f;
    const int32_t qOffset = IsQuantizedType<T>() ? 50    : 0;

    TensorInfo tensorInfo({ 1, 1, 2, 3 }, ArmnnType, qScale, qOffset);

    std::vector<float> inputData =
    {
       -1.f,  2.f, -3.f,
        4.f, -5.f,  6.f
    };

    std::vector<float> expectedOutputData =
    {
        1.f, 2.f, 3.f,
        4.f, 5.f, 6.f
    };

    // quantize data
    std::vector<T> qInputData          = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    INetworkPtr network = CreateAbsNetwork(tensorInfo);

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                { { 0, qInputData } },
                                                { { 0, qExpectedOutputData } },
                                                backends);
}

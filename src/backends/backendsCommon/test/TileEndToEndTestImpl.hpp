//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "armnn/INetwork.hpp"
#include "armnnUtils/QuantizeHelper.hpp"
#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>
#include <doctest/doctest.h>

namespace
{
using namespace armnn;
armnn::INetworkPtr CreateTileNetwork(TileDescriptor& descriptor,
                                     const armnn::TensorInfo& inputInfo,
                                     const armnn::TensorInfo& outputInfo)
{
    INetworkPtr network(INetwork::Create());
    IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
    IConnectableLayer* tileLayer   = network->AddTileLayer(descriptor, "tile");
    IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");
    Connect(inputLayer,   tileLayer, inputInfo,  0, 0);
    Connect(tileLayer, outputLayer,  outputInfo, 0, 0);
    return network;
}

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void TileEndToEnd(const std::vector<BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    const TensorShape inputTensorShape =  { 2, 3 };
    const TensorShape outputTensorShape = { 4, 6 };

    TensorInfo inputInfo  (inputTensorShape, ArmnnType, qScale, qOffset, qConst);
    TensorInfo outputInfo (outputTensorShape, ArmnnType,qScale, qOffset);

    std::vector<T> inputData = armnnUtils::QuantizedVector<T>({
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f
    }, qScale, qOffset);

    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>({
        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f,
        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f
    }, qScale, qOffset);

    auto descriptor = armnn::TileDescriptor(std::vector<uint32_t>{ 2, 2 });
    INetworkPtr network = CreateTileNetwork(descriptor, inputInfo, outputInfo);

    std::map<int, std::vector<T>> inputTensor          = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputTensor = { { 0, expectedOutputData } };
    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutputTensor, backends);
}

} // anonymous namespace
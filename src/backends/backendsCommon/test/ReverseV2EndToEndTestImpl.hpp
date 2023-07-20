//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

namespace
{

armnn::INetworkPtr CreateReverseV2Network(const armnn::TensorInfo& inputInfo,
                                          const armnn::TensorInfo& axisInfo,
                                          const armnn::TensorInfo& outputInfo,
                                          const std::vector<int32_t>& axisData)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    IConnectableLayer* inputLayer   = network->AddInputLayer(0, "input");
    IConnectableLayer* axisLayer    = network->AddConstantLayer(ConstTensor(axisInfo, axisData));
    IConnectableLayer* reverseLayer = network->AddReverseV2Layer("reverseV2");
    IConnectableLayer* outputLayer  = network->AddOutputLayer(0, "output");

    Connect(inputLayer,   reverseLayer, inputInfo,  0, 0);
    Connect(axisLayer,    reverseLayer, axisInfo,   0, 1);
    Connect(reverseLayer, outputLayer,  outputInfo, 0, 0);

    return network;
}

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReverseV2EndToEnd(const std::vector<BackendId>& backends)
{
    float   qScale  = 1.0f;
    int32_t qOffset = 0;
    bool    qConst  = true;

    TensorInfo inputInfo  ( { 2, 3, 4 }, ArmnnType,          qScale, qOffset, qConst);
    TensorInfo axisInfo   ( { 1 }      , DataType::Signed32, qScale, qOffset, qConst);
    TensorInfo outputInfo ( { 2, 3, 4 }, ArmnnType,          qScale, qOffset);

    std::vector<T> inputData = armnnUtils::QuantizedVector<T>({
         1,  2,  3,  4,
         5,  6 , 7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    }, qScale, qOffset);

    std::vector<int32_t> axisData = { 1 };

    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>({
         9, 10, 11, 12,
         5,  6,  7 , 8,
         1,  2,  3,  4,
        21, 22, 23, 24,
        17, 18, 19, 20,
        13, 14, 15, 16
    }, qScale, qOffset);

    INetworkPtr network = CreateReverseV2Network(inputInfo, axisInfo, outputInfo, axisData);

    std::map<int, std::vector<T>> inputTensor          = { { 0, inputData  } };
    std::map<int, std::vector<T>> expectedOutputTensor = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutputTensor, backends);
}

} // anonymous namespace


//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename T>
armnn::INetworkPtr CreateDequantizeNetwork(const armnn::TensorInfo& inputInfo,
                                           const armnn::TensorInfo& outputInfo)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer = net->AddInputLayer(0);
    armnn::IConnectableLayer* dequantizeLayer = net->AddDequantizeLayer("Dequantize");
    armnn::IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");
    Connect(inputLayer, dequantizeLayer, inputInfo, 0, 0);
    Connect(dequantizeLayer, outputLayer, outputInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DequantizeEndToEndLayerTestImpl(const std::vector<BackendId>& backends,
                                     const armnn::TensorShape& tensorShape,
                                     const std::vector<T>& input,
                                     const std::vector<float>& expectedOutput,
                                     float scale,
                                     int32_t offset)
{
    armnn::TensorInfo inputInfo(tensorShape, ArmnnType);
    armnn::TensorInfo outputInfo(tensorShape, armnn::DataType::Float32);

    inputInfo.SetQuantizationScale(scale);
    inputInfo.SetQuantizationOffset(offset);
    inputInfo.SetConstant(true);

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateDequantizeNetwork<T>(inputInfo, outputInfo);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = { { 0, input } };
    std::map<int, std::vector<float>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, armnn::DataType::Float32>(
            move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DequantizeEndToEndSimple(const std::vector<BackendId>& backends)
{
    const armnn::TensorShape tensorShape({ 1, 2, 2, 4 });
    std::vector<T> inputData = std::vector<T>(
    {
        2, 4, 6, 8,
        10, 12, 14, 16,
        18, 20, 22, 24,
        26, 28, 30, 32
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    DequantizeEndToEndLayerTestImpl<ArmnnType>(backends, tensorShape, inputData, expectedOutputData, 0.5f, 0);
};

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void DequantizeEndToEndOffset(const std::vector<BackendId>& backends)
{
    const armnn::TensorShape tensorShape({ 1, 2, 2, 4 });
    std::vector<T> inputData = std::vector<T>(
    {
        3, 5, 7, 9,
        11, 13, 15, 17,
        19, 21, 23, 25,
        27, 29, 31, 33
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    });
    DequantizeEndToEndLayerTestImpl<ArmnnType>(backends, tensorShape, inputData, expectedOutputData, 0.5f, 1);
};

} // anonymous namespace

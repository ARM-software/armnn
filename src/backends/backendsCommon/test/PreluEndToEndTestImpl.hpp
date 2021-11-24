//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>

namespace
{
template<typename armnn::DataType DataType>
INetworkPtr CreatePreluNetwork(const armnn::TensorInfo& inputInfo,
                               const armnn::TensorInfo& alphaInfo,
                               const armnn::TensorInfo& outputInfo)
{
    using namespace armnn;

    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* alpha = net->AddInputLayer(1, "alpha");
    IConnectableLayer* prelu = net->AddPreluLayer("Prelu");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(input, prelu, inputInfo, 0, 0);
    Connect(alpha, prelu, alphaInfo, 0, 1);
    Connect(prelu, output, outputInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void PreluEndToEnd(const std::vector<BackendId>& backends,
                   const std::vector<T>& inputData,
                   const std::vector<T>& alphaData,
                   const std::vector<T>& expectedOutputData,
                   const float qScale ,
                   const int32_t qOffset)
{
    using namespace armnn;

    armnn::TensorInfo inputInfo({ 2, 2, 2, 1 }, ArmnnType);
    armnn::TensorInfo alphaInfo({ 1, 2, 2, 1 }, ArmnnType);
    armnn::TensorInfo outputInfo({ 2, 2, 2, 1 }, ArmnnType);

    inputInfo.SetQuantizationOffset(qOffset);
    inputInfo.SetQuantizationScale(qScale);
    inputInfo.SetConstant(true);
    alphaInfo.SetQuantizationOffset(qOffset);
    alphaInfo.SetQuantizationScale(qScale);
    alphaInfo.SetConstant(true);
    outputInfo.SetQuantizationOffset(qOffset);
    outputInfo.SetQuantizationScale(qScale);

    INetworkPtr net = CreatePreluNetwork<ArmnnType>(inputInfo, alphaInfo, outputInfo);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData          = { { 0, inputData }, { 1, alphaData} };
    std::map<int, std::vector<T>> expectedOutputTensorData = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void PreluEndToEndPositiveTest(const std::vector<BackendId>& backends, const float qScale = 1.0f,
                               const int32_t qOffset = 2)
{
    std::vector<T> inputData{ 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<T> alphaData{ 2, 1, 1, 1 };

    std::vector<T> expectedOutputData{ 2, 2, 3, 4, 5, 6, 7, 8 };

    PreluEndToEnd<ArmnnType>(backends, inputData, alphaData, expectedOutputData, qScale, qOffset);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void PreluEndToEndNegativeTest(const std::vector<BackendId>& backends, const float qScale = 1.0f,
                               const int32_t qOffset = 0)
{
    std::vector<T> inputData{ 1, -2, 3, 4, 5, 6, 7, 8 };
    std::vector<T> alphaData{ 1, 2, 1, 1 };

    std::vector<T> expectedOutputData{ 1, -4, 3, 4, 5, 6, 7, 8 };

    PreluEndToEnd<ArmnnType>(backends, inputData, alphaData, expectedOutputData, qScale, qOffset);
}

} // anonymous namespace
//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace{

armnn::INetworkPtr CreateGatherNdNetwork(const armnn::TensorInfo& paramsInfo,
                                         const armnn::TensorInfo& indicesInfo,
                                         const armnn::TensorInfo& outputInfo,
                                         const std::vector<int32_t>& indicesData)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* paramsLayer = net->AddInputLayer(0);
    armnn::IConnectableLayer* indicesLayer = net->AddConstantLayer(armnn::ConstTensor(indicesInfo, indicesData));
    armnn::IConnectableLayer* gatherNdLayer = net->AddGatherNdLayer("gatherNd");
    armnn::IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");
    Connect(paramsLayer, gatherNdLayer, paramsInfo, 0, 0);
    Connect(indicesLayer, gatherNdLayer, indicesInfo, 0, 1);
    Connect(gatherNdLayer, outputLayer, outputInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void GatherNdEndToEnd(const std::vector<BackendId>& backends)
{
    armnn::TensorInfo paramsInfo({ 2, 3, 8, 4 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 2 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 8, 4 }, ArmnnType);

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    // Creates structures for input & output.
    std::vector<T> paramsData{
             0,   1,   2,   3, 4,   5,   6,   7, 8,   9,  10,  11, 12,  13,  14,  15,
            16,  17,  18,  19, 20,  21,  22,  23, 24,  25,  26,  27, 28,  29,  30,  31,

            32,  33,  34,  35, 36,  37,  38,  39, 40,  41,  42,  43, 44,  45,  46,  47,
            48,  49,  50,  51, 52,  53,  54,  55, 56,  57,  58,  59, 60,  61,  62,  63,

            64,  65,  66,  67, 68,  69,  70,  71, 72,  73,  74,  75, 76,  77,  78,  79,
            80,  81,  82,  83, 84,  85,  86,  87, 88,  89,  90,  91, 92,  93,  94,  95,

            96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,

            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,

            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191
    };

    std::vector<int32_t> indicesData{
            { 1, 2, 1, 1},
    };

    std::vector<T> expectedOutput{
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,

        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159
    };

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNdNetwork(paramsInfo, indicesInfo, outputInfo, indicesData);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, paramsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void GatherNdMultiDimEndToEnd(const std::vector<BackendId>& backends)
{
    armnn::TensorInfo paramsInfo({ 5, 5, 2 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 2, 3, 2 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 2, 3, 2 }, ArmnnType);

    paramsInfo.SetQuantizationScale(1.0f);
    paramsInfo.SetQuantizationOffset(0);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(1.0f);
    outputInfo.SetQuantizationOffset(0);

    // Creates structures for input & output.
    std::vector<T> paramsData{
            0,  1,    2,  3,    4,  5,    6,  7,    8,  9,
            10, 11,   12,  13,   14, 15,   16, 17,   18, 19,
            20, 21,   22,  23,   24, 25,   26, 27,   28, 29,
            30, 31,   32,  33,   34, 35,   36, 37,   38, 39,
            40, 41,   42,  43,   44, 45,   46, 47,   48, 49
    };

    std::vector<int32_t> indicesData{
            0, 0,
            3, 3,
            4, 4,

            0, 0,
            1, 1,
            2, 2,

            4, 4,
            3, 3,
            0, 0,

            2, 2,
            1, 1,
            0, 0
    };

    std::vector<T> expectedOutput{
            0,  1,
            36, 37,
            48, 49,

            0,  1,
            12, 13,
            24, 25,

            48, 49,
            36, 37,
            0,  1,

            24, 25,
            12, 13,
            0,  1
    };

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNdNetwork(paramsInfo, indicesInfo, outputInfo, indicesData);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, paramsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

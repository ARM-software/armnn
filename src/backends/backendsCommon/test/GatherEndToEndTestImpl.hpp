//
// Copyright © 2019-2021,2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

armnn::INetworkPtr CreateGatherNetwork(const armnn::TensorInfo& paramsInfo,
                                       const armnn::TensorInfo& indicesInfo,
                                       const armnn::TensorInfo& outputInfo,
                                       const std::vector<int32_t>& indicesData,
                                       const int axis = 0)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::GatherDescriptor descriptor;
    descriptor.m_Axis = axis;
    armnn::IConnectableLayer* paramsLayer = net->AddInputLayer(0, "params");
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

    const float qScale  = armnn::IsQuantizedType<T>() ? 0.5f : 1.0f;
    const int32_t qOffset = armnn::IsQuantizedType<T>() ? 2 : 0;
    paramsInfo.SetQuantizationScale(qScale);
    paramsInfo.SetQuantizationOffset(qOffset);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(qScale);
    outputInfo.SetQuantizationOffset(qOffset);

    // Creates structures for input & output.
    std::vector<T> paramsData
    {
        1, 2, 3, 4, 5, 6, 7, 8
    };

    std::vector<int32_t> indicesData
    {
        7, 6, 5
    };

    std::vector<T> expectedOutput
    {
        8, 7, 6
    };

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNetwork(paramsInfo, indicesInfo, outputInfo, indicesData);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, paramsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void GatherMultiDimEndToEnd(const std::vector<BackendId>& backends, int axis = 0)
{
    armnn::TensorInfo paramsInfo({ 3, 2, 3 }, ArmnnType);
    armnn::TensorInfo indicesInfo = (axis == 1 ) ? TensorInfo({ 1, 3 }, armnn::DataType::Signed32)
                              /* axis 0 and 2 */ : TensorInfo({ 2, 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo = (axis == 0) ? TensorInfo({ 2, 3, 2, 3 }, ArmnnType) :
                                   (axis == 1) ? TensorInfo({ 3, 1, 3, 3 }, ArmnnType)
                                  /* axis 2 */ : TensorInfo({ 3, 2, 2, 3 }, ArmnnType);

    const float_t qScale  = armnn::IsQuantizedType<T>() ? 0.9f : 1.0f;
    const int32_t qOffset = armnn::IsQuantizedType<T>() ? 2 : 0;
    paramsInfo.SetQuantizationScale(qScale);
    paramsInfo.SetQuantizationOffset(qOffset);
    paramsInfo.SetConstant(true);
    indicesInfo.SetConstant(true);
    outputInfo.SetQuantizationScale(qScale);
    outputInfo.SetQuantizationOffset(qOffset);

    // Creates structures for input & output.
    std::vector<float_t> paramsData
    {
         1,  2,  3,
         4,  5,  6,

         7,  8,  9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    };

    std::vector<int32_t> indicesData = (axis == 1) ? std::vector<int32_t>({ 1, 0, 1 })
                                /* axis 0 and 2 */ : std::vector<int32_t>({ 1, 0, 2,
                                                                            2, 1, 0 });

    std::vector<float_t> expectedOutput = (axis == 0) ? std::vector<float_t>({ 7,  8,  9,
                                                                              10, 11, 12,
                                                                              1,  2,  3,
                                                                              4,  5,  6,
                                                                              13, 14, 15,
                                                                              16, 17, 18,
                                                                              13, 14, 15,
                                                                              16, 17, 18,
                                                                              7,  8,  9,
                                                                              10, 11, 12,
                                                                              1,  2,  3,
                                                                              4,  5,  6 }) :
                                         (axis == 1) ? std::vector<float_t>({ 4,  5,  6,
                                                                              1,  2,  3,
                                                                              4,  5,  6,
                                                                              10, 11, 12,
                                                                              7,  8,  9,
                                                                              10, 11, 12,
                                                                              16, 17, 18,
                                                                              13, 14, 15,
                                                                              16, 17, 18 })
                                        /* axis 2 */ : std::vector<float_t>({ 2,  1,  3,
                                                                              3,  2,  1,
                                                                              5,  4,  6,
                                                                              6,  5,  4,
                                                                              8,  7,  9,
                                                                              9,  8,  7,
                                                                              11, 10, 12,
                                                                              12, 11, 10,
                                                                              14, 13, 15,
                                                                              15, 14, 13,
                                                                              17, 16, 18,
                                                                              18, 17, 16 });

    std::vector<T> qParamsData = armnnUtils::QuantizedVector<T>(paramsData, qScale, qOffset);
    std::vector<T> qExpectedOutputData = armnnUtils::QuantizedVector<T>(expectedOutput, qScale, qOffset);

    // Builds up the structure of the network
    armnn::INetworkPtr net = CreateGatherNetwork(paramsInfo, indicesInfo, outputInfo, indicesData, axis);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, qParamsData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, qExpectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

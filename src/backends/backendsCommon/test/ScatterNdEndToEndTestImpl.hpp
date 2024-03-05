//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{

template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
INetworkPtr CreateScatterNdNetwork(const TensorInfo& shapeInfo,
                                   const TensorInfo& indicesInfo,
                                   const TensorInfo& updatesInfo,
                                   const TensorInfo& outputInfo,
                                   const std::vector<int32_t>& indicesData,
                                   const std::vector<T>& updatesData,
                                   const ScatterNdDescriptor& descriptor)
{
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* shapeLayer = net->AddInputLayer(0);
    IConnectableLayer* indicesLayer = net->AddConstantLayer(ConstTensor(indicesInfo, indicesData));
    IConnectableLayer* updatesLayer = net->AddConstantLayer(ConstTensor(updatesInfo, updatesData));
    IConnectableLayer* scatterNdLayer = net->AddScatterNdLayer(descriptor, "scatterNd");
    IConnectableLayer* outputLayer = net->AddOutputLayer(0, "output");
    Connect(shapeLayer, scatterNdLayer, shapeInfo, 0, 0);
    Connect(indicesLayer, scatterNdLayer, indicesInfo, 0, 1);
    Connect(updatesLayer, scatterNdLayer, updatesInfo, 0, 2);
    Connect(scatterNdLayer, outputLayer, outputInfo, 0, 0);

    return net;
}

template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
void ScatterNd1DimUpdateWithInputEndToEnd(const std::vector<BackendId>& backends)
{
    float_t qScale = 1.f;
    int32_t qOffset = 0;

    TensorInfo inputInfo({ 5 }, ArmnnType, qScale, qOffset, true);
    TensorInfo indicesInfo({ 3, 1 }, DataType::Signed32, 1.0f, 0, true);
    TensorInfo updatesInfo({ 3 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 5 }, ArmnnType, qScale, qOffset, false);

    std::vector<T> inputData = armnnUtils::QuantizedVector<T>({ 0, 0, 0, 0, 0 }, qScale, qOffset);
    std::vector<int32_t> indicesData{0, 1, 2};
    std::vector<T> updatesData = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>({ 1, 2, 3, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    INetworkPtr net = CreateScatterNdNetwork<ArmnnType>(inputInfo, indicesInfo, updatesInfo, outputInfo,
                                                        indicesData, updatesData, descriptor);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
void ScatterNd1DimUpdateNoInputEndToEnd(const std::vector<BackendId>& backends)
{
    float_t qScale = 1.f;
    int32_t qOffset = 0;

    TensorInfo shapeInfo({ 1 }, DataType::Signed32, 1.0f, 0, true);
    TensorInfo indicesInfo({ 3, 1 }, DataType::Signed32,  1.0f, 0, true);
    TensorInfo updatesInfo({ 3 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 5 }, ArmnnType, qScale, qOffset, false);

    std::vector<int32_t> shapeData{ 5 };
    std::vector<int32_t> indicesData{ 0, 1, 2 };
    std::vector<T> updatesData = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>({ 1, 2, 3, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    INetworkPtr net = CreateScatterNdNetwork<ArmnnType>(shapeInfo, indicesInfo, updatesInfo, outputInfo,
                                                        indicesData, updatesData, descriptor);

    CHECK(net);

    std::map<int, std::vector<int32_t>> inputTensorData = {{ 0, shapeData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<DataType::Signed32, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
void ScatterNd2DimUpdateWithInputEndToEnd(const std::vector<BackendId>& backends)
{
    float_t qScale = 1.f;
    int32_t qOffset = 0;

    TensorInfo inputInfo({ 3, 3 }, ArmnnType, qScale, qOffset, true);
    TensorInfo indicesInfo({ 3, 2 }, DataType::Signed32, 1.0f, 0, true);
    TensorInfo updatesInfo({ 3 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 3, 3 }, ArmnnType, qScale, qOffset, false);

    std::vector<T> inputData = armnnUtils::QuantizedVector<T>({ 1, 1, 1, 1, 1, 1, 1, 1, 1 }, qScale, qOffset);
    std::vector<int32_t> indicesData{0, 0, 1, 1, 2, 2};
    std::vector<T> updatesData = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>({ 1, 1, 1, 1, 2, 1, 1, 1, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    INetworkPtr net = CreateScatterNdNetwork<ArmnnType>(inputInfo, indicesInfo, updatesInfo, outputInfo,
                                                        indicesData, updatesData, descriptor);

    CHECK(net);

    std::map<int, std::vector<T>> inputTensorData = {{ 0, inputData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

template<DataType ArmnnType, typename T = ResolveType<ArmnnType>>
void ScatterNd2DimUpdateNoInputEndToEnd(const std::vector<BackendId>& backends)
{
    float_t qScale = 1.f;
    int32_t qOffset = 0;

    TensorInfo shapeInfo({ 2 }, DataType::Signed32, 1.0f, 0, true);
    TensorInfo indicesInfo({ 3, 2 }, DataType::Signed32, 1.0f, 0, true);
    TensorInfo updatesInfo({ 3 }, ArmnnType, qScale, qOffset, true);
    TensorInfo outputInfo({ 3, 3 }, ArmnnType, qScale, qOffset, false);

    std::vector<int32_t> shapeData{ 3, 3 };
    std::vector<int32_t> indicesData{0, 0, 1, 1, 2, 2};
    std::vector<T> updatesData = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>({ 1, 0, 0, 0, 2, 0, 0, 0, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    INetworkPtr net = CreateScatterNdNetwork<ArmnnType>(shapeInfo, indicesInfo, updatesInfo, outputInfo,
                                                        indicesData, updatesData, descriptor);

    CHECK(net);

    std::map<int, std::vector<int32_t>> inputTensorData = {{ 0, shapeData }};
    std::map<int, std::vector<T>> expectedOutputData = {{ 0, expectedOutput }};

    EndToEndLayerTestImpl<DataType::Signed32, ArmnnType>(std::move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>
#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateSubtractionNetwork(const armnn::TensorShape& inputXShape,
                                            const armnn::TensorShape& inputYShape,
                                            const armnn::TensorShape& outputShape,
                                            const float qScale = 1.0f,
                                            const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputXTensorInfo(inputXShape, DataType, qScale, qOffset, true);
    TensorInfo inputYTensorInfo(inputYShape, DataType, qScale, qOffset, true);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    IConnectableLayer* subtraction = network->AddSubtractionLayer("subtraction");
    ARMNN_NO_DEPRECATE_WARN_END
    IConnectableLayer* inputX = network->AddInputLayer(0, "inputX");
    IConnectableLayer* inputY = network->AddInputLayer(1, "inputY");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(inputX, subtraction, inputXTensorInfo, 0, 0);
    Connect(inputY, subtraction, inputYTensorInfo, 0, 1);
    Connect(subtraction, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void SubtractionEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputXShape = { 2, 2 };
    const TensorShape& inputYShape = { 2, 2 };
    const TensorShape& outputShape = { 2, 2 };

    INetworkPtr network = CreateSubtractionNetwork<ArmnnType>(inputXShape, inputYShape, outputShape);

    CHECK(network);

    std::vector<T> inputXData{ 10, 11, 12, 13 };
    std::vector<T> inputYData{ 5, 7, 6, 8 };
    std::vector<T> expectedOutput{ 5, 4, 6, 5 };

    std::map<int, std::vector<T>> inputTensorData = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void SubtractionEndToEndFloat16(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape& inputXShape = { 2, 2 };
    const TensorShape& inputYShape = { 2, 2 };
    const TensorShape& outputShape = { 2, 2 };

    INetworkPtr network = CreateSubtractionNetwork<ArmnnType>(inputXShape, inputYShape, outputShape);
    CHECK(network);

    std::vector<Half> inputXData{ 11._h, 12._h,
                                  13._h, 14._h };
    std::vector<Half> inputYData{ 5._h, 7._h,
                                  6._h, 8._h };
    std::vector<Half> expectedOutput{ 6._h, 5._h,
                                      7._h, 6._h };

    std::map<int, std::vector<Half>> inputTensorData = {{ 0, inputXData }, { 1, inputYData }};
    std::map<int, std::vector<Half>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

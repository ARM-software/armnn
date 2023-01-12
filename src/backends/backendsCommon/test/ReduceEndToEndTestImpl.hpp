//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
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
armnn::INetworkPtr CreateReduceNetwork(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& outputShape,
                                        const armnn::ReduceDescriptor& descriptor,
                                        const float qScale = 1.0f,
                                        const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);


    IConnectableLayer* reduce = network->AddReduceLayer(descriptor, "reduce");
    IConnectableLayer* input   = network->AddInputLayer(0, "input");
    IConnectableLayer* output  = network->AddOutputLayer(0, "output");

    Connect(input, reduce, inputTensorInfo, 0, 0);
    Connect(reduce, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReduceEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape = { 1, 1, 1, 5 };
    const TensorShape& outputShape = { 1, 1, 1 };

    ReduceDescriptor descriptor;
    descriptor.m_KeepDims = false;
    descriptor.m_vAxis = { 3 };
    descriptor.m_ReduceOperation = ReduceOperation::Sum;

    INetworkPtr network = CreateReduceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<float> floatInputData({ 5.0f, 2.0f, 8.0f, 10.0f, 9.0f });
    std::vector<float> floatOutputData({ 34.0f });

    std::vector<T> inputData = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> outputData = armnnUtils::QuantizedVector<T>(floatOutputData);

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, outputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}
} // anonymous namespace

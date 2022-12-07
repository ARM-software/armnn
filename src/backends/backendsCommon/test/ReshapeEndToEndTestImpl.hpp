//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
armnn::INetworkPtr CreateReshapeNetwork(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& outputShape,
                                        const armnn::ReshapeDescriptor& descriptor,
                                        const float qScale = 1.0f,
                                        const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);


    IConnectableLayer* reshape = network->AddReshapeLayer(descriptor, "reshape");
    IConnectableLayer* input   = network->AddInputLayer(0, "input");
    IConnectableLayer* output  = network->AddOutputLayer(0, "output");

    Connect(input, reshape, inputTensorInfo, 0, 0);
    Connect(reshape, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void ReshapeEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape = { 2, 3 };
    const TensorShape& outputShape = { 6 };

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = outputShape;

    INetworkPtr network = CreateReshapeNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<T> data{ 1, 2, 3,
                         4, 5, 6 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, data } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, data } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void ReshapeEndToEndFloat16(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape& inputShape = { 2, 3 };
    const TensorShape& outputShape = { 6 };

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = outputShape;

    INetworkPtr network = CreateReshapeNetwork<ArmnnType>(inputShape, outputShape, descriptor);
    CHECK(network);

    std::vector<Half> data{ 1._h, 2._h, 3._h,
                            4._h, 5._h, 6._h };

    std::map<int, std::vector<Half>> inputTensorData = { { 0, data } };
    std::map<int, std::vector<Half>> expectedOutputData = { { 0, data } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

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
armnn::INetworkPtr CreateSliceNetwork(const armnn::TensorShape& inputShape,
                                      const armnn::TensorShape& outputShape,
                                      const armnn::SliceDescriptor& descriptor,
                                      const float qScale = 1.0f,
                                      const int32_t qOffset = 0)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);


    IConnectableLayer* slice  = network->AddSliceLayer(descriptor, "slice");
    IConnectableLayer* input  = network->AddInputLayer(0, "input");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(input, slice, inputTensorInfo, 0, 0);
    Connect(slice, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void SliceEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputShape  = { 3, 2, 3 };
    const TensorShape& outputShape = { 2, 1, 3 };

    SliceDescriptor descriptor;
    descriptor.m_Begin = { 1, 0, 0 };
    descriptor.m_Size  = { 2, 1, 3 };

    INetworkPtr network = CreateSliceNetwork<ArmnnType>(inputShape, outputShape, descriptor);

    CHECK(network);

    std::vector<T> inputData{ 1, 1, 1, 2, 2, 2,
                              3, 3, 3, 4, 4, 4,
                              5, 5, 5, 6, 6, 6 };
    std::vector<T> expectedOutput{ 3, 3, 3,
                                   5, 5, 5 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void SliceEndToEndFloat16(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;
    using namespace half_float::literal;
    using Half = half_float::half;

    const TensorShape& inputShape = { 3, 2, 3 };
    const TensorShape& outputShape = { 2, 1, 3 };

    SliceDescriptor descriptor;
    descriptor.m_Begin = { 1, 0, 0 };
    descriptor.m_Size  = { 2, 1, 3 };

    INetworkPtr network = CreateSliceNetwork<ArmnnType>(inputShape, outputShape, descriptor);
    CHECK(network);

    std::vector<Half> inputData{ 1._h, 1._h, 1._h, 2._h, 2._h, 2._h,
                                 3._h, 3._h, 3._h, 4._h, 4._h, 4._h,
                                 5._h, 5._h, 5._h, 6._h, 6._h, 6._h };
    std::vector<Half> expectedOutput{ 3._h, 3._h, 3._h,
                                      5._h, 5._h, 5._h };

    std::map<int, std::vector<Half>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<Half>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace
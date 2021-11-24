//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <CommonTestUtils.hpp>

#include <doctest/doctest.h>

#include <vector>

namespace
{

template<typename armnn::DataType DataType>
INetworkPtr CreateBatchToSpaceNdNetwork(const armnn::TensorShape& inputShape,
                                        const armnn::TensorShape& outputShape,
                                        std::vector<unsigned int>& blockShape,
                                        std::vector<std::pair<unsigned int, unsigned int>>& crops,
                                        armnn::DataLayout dataLayout,
                                        const float qScale = 1.0f,
                                        const int32_t qOffset = 0)
{
    using namespace armnn;
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo inputTensorInfo(inputShape, DataType, qScale, qOffset, true);
    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);

    BatchToSpaceNdDescriptor batchToSpaceNdDesc(blockShape, crops);
    batchToSpaceNdDesc.m_DataLayout = dataLayout;

    IConnectableLayer* batchToSpaceNd = net->AddBatchToSpaceNdLayer(batchToSpaceNdDesc, "batchToSpaceNd");
    IConnectableLayer* input = net->AddInputLayer(0, "input");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    Connect(batchToSpaceNd, output, outputTensorInfo, 0, 0);
    Connect(input, batchToSpaceNd, inputTensorInfo, 0, 0);

    return net;
}

template<armnn::DataType ArmnnType>
void BatchToSpaceNdEndToEnd(const std::vector<BackendId>& backends, armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};
    const TensorShape& inputShape  = { 4, 1, 1, 1 };
    const TensorShape& outputShape = (dataLayout == DataLayout::NCHW)
                                     ? std::initializer_list<unsigned int>({ 1, 1, 2, 2 })
                                     : std::initializer_list<unsigned int>({ 1, 2, 2, 1 });

    // Builds up the structure of the network
    INetworkPtr net = CreateBatchToSpaceNdNetwork<ArmnnType>(inputShape, outputShape, blockShape, crops, dataLayout);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{ 1, 2, 3, 4 };

    std::vector<T> expectedOutput{ 1, 2, 3, 4 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

template<armnn::DataType ArmnnType>
void BatchToSpaceNdComplexEndToEnd(const std::vector<BackendId>& backends, armnn::DataLayout dataLayout)
{
    using namespace armnn;
    using T = ResolveType<ArmnnType>;

    std::vector<unsigned int> blockShape {2, 2};
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {2, 0}};
    const TensorShape& inputShape  = (dataLayout == DataLayout::NCHW)
                                     ? std::initializer_list<unsigned int>({ 8, 1, 1, 3 })
                                     : std::initializer_list<unsigned int>({ 8, 1, 3, 1 });
    const TensorShape& outputShape = (dataLayout == DataLayout::NCHW)
                                     ? std::initializer_list<unsigned int>({ 2, 1, 2, 4 })
                                     : std::initializer_list<unsigned int>({ 2, 2, 4, 1 });

    // Builds up the structure of the network
    INetworkPtr net = CreateBatchToSpaceNdNetwork<ArmnnType>(inputShape, outputShape, blockShape, crops, dataLayout);

    CHECK(net);

    // Creates structures for input & output.
    std::vector<T> inputData{
                              0, 1, 3, 0,  9, 11,
                              0, 2, 4, 0, 10, 12,
                              0, 5, 7, 0, 13, 15,
                              0, 6, 8, 0, 14, 16
                            };

    std::vector<T> expectedOutput{
                                   1,   2,  3,  4,
                                   5,   6,  7,  8,
                                   9,  10, 11, 12,
                                   13, 14, 15, 16
                                 };

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputData } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutput } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(net), inputTensorData, expectedOutputData, backends);
}

} // anonymous namespace

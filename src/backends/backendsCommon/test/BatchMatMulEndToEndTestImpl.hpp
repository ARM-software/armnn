//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>
#include <CommonTestUtils.hpp>

namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreateBatchMatMulNetwork(const armnn::TensorShape& inputXShape,
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

    BatchMatMulDescriptor batchMatMulDesc;
    batchMatMulDesc.m_TransposeX = false;
    batchMatMulDesc.m_TransposeY = true;

    IConnectableLayer* batchMatMul = network->AddBatchMatMulLayer(batchMatMulDesc, "batchMatMul");
    IConnectableLayer* inputX = network->AddInputLayer(0, "inputX");
    IConnectableLayer* inputY = network->AddInputLayer(1, "inputY");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(inputX, batchMatMul, inputXTensorInfo, 0, 0);
    Connect(inputY, batchMatMul, inputYTensorInfo, 0, 1);
    Connect(batchMatMul, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchMatMulEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputXShape = { 2, 2, 2 };
    const TensorShape& inputYShape = { 2, 2, 2 };
    const TensorShape& outputShape = { 2, 2, 2 };

    constexpr float qScale    = 1.0f;
    constexpr int32_t qOffset = 0;

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape, inputYShape, outputShape, qScale, qOffset);

    CHECK(network);

    std::vector<float> floatInputXData{ 1., 2.,
                                        3., 4.,

                                        9., 10.,
                                        11., 12. };
    std::vector<T> inputXData = armnnUtils::QuantizedVector<T>(floatInputXData, qScale, qOffset);

    std::vector<float> floatInputYData{ 5., 7.,
                                        6., 8.,

                                        13., 15.,
                                        14., 16. };
    std::vector<T> inputYData = armnnUtils::QuantizedVector<T>(floatInputYData, qScale, qOffset);

    std::vector<float> floatExpectedOutputData{ 19., 22.,
                                                43., 50.,

                                                267., 286.,
                                                323., 346. };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensor = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutput = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutput, backends);
}

} // anonymous namespace
//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
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
                                     const int32_t qOffset = 0,
                                     const bool transposeX = true,
                                     const bool transposeY = true)
{
    using namespace armnn;

    INetworkPtr network(INetwork::Create());

    TensorInfo inputXTensorInfo(inputXShape, DataType, qScale, qOffset, true);
    TensorInfo inputYTensorInfo(inputYShape, DataType, qScale, qOffset, true);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset);

    BatchMatMulDescriptor batchMatMulDesc;
    batchMatMulDesc.m_TransposeX = transposeX;
    batchMatMulDesc.m_TransposeY = transposeY;

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

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape,
                                                              inputYShape,
                                                              outputShape,
                                                              qScale,
                                                              qOffset,
                                                              false);

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

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchMatMulNotSquareEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;
    const TensorShape& inputXShape = { 2, 5, 3 };
    const TensorShape& inputYShape = { 2, 3, 4 };
    const TensorShape& outputShape = { 2, 5, 4 };

    constexpr float qScale    = 1.0f;
    constexpr int32_t qOffset = 0;

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape,
                                                              inputYShape,
                                                              outputShape, qScale, qOffset, false, false);

    CHECK(network);

    std::vector<float> floatInputXData{ 8, 8, 4,
                                        6, 1, 3,
                                        8, 8, 3,
                                        8, 9, 8,
                                        5, 4, 4,

                                        1, 8, 5,
                                        7, 1, 1,
                                        8, 7, 9,
                                        3, 2, 7,
                                        8, 5, 3 };
    std::vector<T> inputXData = armnnUtils::QuantizedVector<T>(floatInputXData, qScale, qOffset);

    std::vector<float> floatInputYData{
            6, 2, 3, 2,
            6, 2, 2, 8,
            3, 7, 8, 1,

            7, 2, 9, 5,
            2, 3, 1, 3,
            2, 7, 7, 5 };
    std::vector<T> inputYData = armnnUtils::QuantizedVector<T>(floatInputYData, qScale, qOffset);

    std::vector<float> floatExpectedOutputData{
            108, 60, 72, 84,
            51, 35, 44, 23,
            105, 53, 64, 83,
            126, 90, 106, 96,
            66, 46, 55, 46,

            33, 61, 52, 54,
            53, 24, 71, 43,
            88, 100, 142, 106,
            39, 61, 78, 56,
            72, 52, 98, 70
    };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensor = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutput = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutput, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchMatMulNoTransposeEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputXShape = { 2, 2, 2 };
    const TensorShape& inputYShape = { 2, 2, 2 };
    const TensorShape& outputShape = { 2, 2, 2 };

    constexpr float qScale    = 1.0f;
    constexpr int32_t qOffset = 0;

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape,
                                                              inputYShape,
                                                              outputShape,
                                                              qScale,
                                                              qOffset,
                                                              false,
                                                              false);

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

    std::vector<float> floatExpectedOutputData{ 17., 23.,
                                                39., 53.,

                                                257., 295.,
                                                311., 357. };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensor = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutput = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutput, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchMatMul4DEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputXShape = { 2, 2, 2, 2 };
    const TensorShape& inputYShape = { 2, 2, 2, 2 };
    const TensorShape& outputShape = { 2, 2, 2, 2 };

    constexpr float qScale    = 1.0f;
    constexpr int32_t qOffset = 0;

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape,
                                                              inputYShape,
                                                              outputShape,
                                                              qScale,
                                                              qOffset,
                                                              false,
                                                              false);

    CHECK(network);

    std::vector<float> floatInputXData{ 1., 2.,
                                        3., 4.,

                                        5., 6.,
                                        7., 8.,

                                        1., 2.,
                                        3., 4.,

                                        5., 6.,
                                        7., 8.};
    std::vector<T> inputXData = armnnUtils::QuantizedVector<T>(floatInputXData, qScale, qOffset);

    std::vector<float> floatInputYData{ 10., 11.,
                                        12., 13.,

                                        14., 15.,
                                        16., 17.,

                                        10., 11.,
                                        12., 13.,

                                        14., 15.,
                                        16., 17. };
    std::vector<T> inputYData = armnnUtils::QuantizedVector<T>(floatInputYData, qScale, qOffset);

    std::vector<float> floatExpectedOutputData{ 34., 37.,
                                                78., 85.,

                                                166., 177.,
                                                226., 241.,

                                                34., 37.,
                                                78., 85.,

                                                166., 177.,
                                                226., 241. };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensor = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutput = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutput, backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void BatchMatMulSimple4DEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const TensorShape& inputXShape = { 1, 1, 2, 2 };
    const TensorShape& inputYShape = { 1, 1, 2, 2 };
    const TensorShape& outputShape = { 1, 1, 2, 2 };

    constexpr float qScale    = 1.0f;
    constexpr int32_t qOffset = 0;

    INetworkPtr network = CreateBatchMatMulNetwork<ArmnnType>(inputXShape,
                                                              inputYShape,
                                                              outputShape,
                                                              qScale,
                                                              qOffset,
                                                              false,
                                                              false);

    CHECK(network);

    std::vector<float> floatInputXData{ 1., 2.,
                                        3., 4. };
    std::vector<T> inputXData = armnnUtils::QuantizedVector<T>(floatInputXData, qScale, qOffset);

    std::vector<float> floatInputYData{ 5., 7.,
                                        6., 8. };
    std::vector<T> inputYData = armnnUtils::QuantizedVector<T>(floatInputYData, qScale, qOffset);

    std::vector<float> floatExpectedOutputData{ 17., 23.,
                                                39., 53. };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensor = {{ 0, inputXData }, {1, inputYData}};
    std::map<int, std::vector<T>> expectedOutput = { { 0, expectedOutputData } };

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network), inputTensor, expectedOutput, backends);
}
} // anonymous namespace
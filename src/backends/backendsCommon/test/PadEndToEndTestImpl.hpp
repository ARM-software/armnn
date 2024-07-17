//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>
#include <CommonTestUtils.hpp>

using namespace armnn;
namespace
{

template<typename armnn::DataType DataType>
armnn::INetworkPtr CreatePadNetwork(const armnn::TensorShape& inputXShape,
                                    const armnn::TensorShape& outputShape,
                                    const armnn::PadDescriptor padDesc,
                                    const float qScale = 1.0f,
                                    const int32_t qOffset = 0)
{

    INetworkPtr network(INetwork::Create());

    TensorInfo inputTensorInfo(inputXShape, DataType, qScale, qOffset, true);

    TensorInfo outputTensorInfo(outputShape, DataType, qScale, qOffset, true);

    IConnectableLayer* padLayer = network->AddPadLayer(padDesc, "pad");
    IConnectableLayer* inputX = network->AddInputLayer(0, "input");
    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    Connect(inputX, padLayer, inputTensorInfo, 0, 0);
    Connect(padLayer, output, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void PadEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 7 };

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::PadDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    descriptor.m_PadList = padList;
    descriptor.m_PadValue = 3;
    armnn::WorkloadInfo info;

    INetworkPtr network = CreatePadNetwork<ArmnnType>(inputShape,
                                                      outputShape,
                                                      descriptor,
                                                      qScale,
                                                      qOffset);
    CHECK(network);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>({
                    // Height (3) x Width (3)
                    4, 8, 6,
                    7, 4, 4,
                    3, 2, 4
            }, qScale, qOffset);

    float p = 3;
    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>({
                    p, p, p, p, p, p, p,
                    p, p, p, p, p, p, p,
                    p, p, 4, 8, 6, p, p,
                    p, p, 7, 4, 4, p, p,
                    p, p, 3, 2, 4, p, p,
                    p, p, p, p, p, p, p,
                    p, p, p, p, p, p, p
            }, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputValues } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutputValues } };
    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                inputTensorData,
                                                expectedOutputData,
                                                backends);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void Pad4dEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    const armnn::TensorShape inputShape{ 2, 2, 3, 2 };
    const armnn::TensorShape outputShape{ 4, 5, 7, 4 };

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::PadDescriptor padDesc;
    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(3,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    padDesc.m_PadList = padList;
    padDesc.m_PadValue = 0;
    INetworkPtr network = CreatePadNetwork<ArmnnType>(inputShape,
                                                      outputShape,
                                                      padDesc,
                                                      qScale,
                                                      qOffset);
    CHECK(network);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>({ // Batch 0, Channel 0, Height (3) x Width (2)
                                                                  0, 1,
                                                                  2, 3,
                                                                  4, 5,

                                                                  // Batch 0, Channel 1, Height (3) x Width (2)
                                                                  6, 7,
                                                                  8, 9,
                                                                  10, 11,

                                                                  // Batch 1, Channel 0, Height (3) x Width (2)
                                                                  12, 13,
                                                                  14, 15,
                                                                  16, 17,

                                                                  // Batch 1, Channel 1, Height (3) x Width (2)
                                                                  18, 19,
                                                                  20, 21,
                                                                  22, 23
                                                                }, qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>({ 0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 1, 0,
                                                                           0, 2, 3, 0,
                                                                           0, 4, 5, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 6, 7, 0,
                                                                           0, 8, 9, 0,
                                                                           0, 10, 11, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 12, 13, 0,
                                                                           0, 14, 15, 0,
                                                                           0, 16, 17, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 18, 19, 0,
                                                                           0, 20, 21, 0,
                                                                           0, 22, 23, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,

                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0,
                                                                           0, 0, 0, 0
                                                                         }, qScale, qOffset);

    std::map<int, std::vector<T>> inputTensorData = { { 0, inputValues } };
    std::map<int, std::vector<T>> expectedOutputData = { { 0, expectedOutputValues } };
    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(std::move(network),
                                                inputTensorData,
                                                expectedOutputData,
                                                backends);
}

} // anonymous namespace
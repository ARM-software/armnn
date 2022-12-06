//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <CommonTestUtils.hpp>

#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>

#include <ResolveType.hpp>

#include <doctest/doctest.h>

namespace
{

armnn::INetworkPtr CreateTransposeNetwork(const armnn::TensorInfo& inputTensorInfo,
                                          const armnn::TensorInfo& outputTensorInfo,
                                          const armnn::PermutationVector& mappings)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    const armnn::TransposeDescriptor transposeDescriptor(mappings);

    armnn::IConnectableLayer* inputLayer     = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* transposeLayer = network->AddTransposeLayer(transposeDescriptor, "Transpose");
    armnn::IConnectableLayer* outputLayer    = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, transposeLayer, inputTensorInfo, 0, 0);
    Connect(transposeLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void TransposeEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    std::vector<float> floatInputData{
            1,  2,  3,  4,  5,
            11, 12, 13, 14, 15,
            21, 22, 23, 24, 25
    };
    std::vector<T> inputData = armnnUtils::QuantizedVector<T>(floatInputData);
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatInputData);

    const armnn::PermutationVector mappings{0, 2, 1 ,3};

    TensorInfo inputInfo ({ 1, 1, 5, 3 }, ArmnnType, 0.0f, 0, true);
    TensorInfo outputInfo({ 1, 5, 1, 3 }, ArmnnType, 0.0f, 0, true);

    armnn::INetworkPtr network = CreateTransposeNetwork(inputInfo, outputInfo, mappings);

    CHECK(network);

    std::map<int, std::vector<T>> inputTensorData   = {{ 0, inputData }};
    std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, DataType::Signed32>(std::move(network),
                                                         inputTensorData,
                                                         expectedOutputTensorData,
                                                         backends);
}

} // anonymous namespace
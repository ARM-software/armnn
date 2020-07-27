//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CommonTestUtils.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/TypesUtils.hpp>

#include <ResolveType.hpp>

namespace
{

armnn::INetworkPtr CreateFillNetwork(const armnn::TensorInfo& inputTensorInfo,
                                     const armnn::TensorInfo& outputTensorInfo,
                                     armnn::FillDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* fillLayer   = network->AddFillLayer(descriptor, "Fill");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fillLayer, inputTensorInfo, 0, 0);
    Connect(fillLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void FillEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    FillDescriptor descriptor;
    descriptor.m_Value = 9;

    std::vector<T> inputData{
            1, 1, 5, 3
    };

    std::vector<T> expectedOutputData{
            9, 9, 9, 9, 9,
            9, 9, 9, 9, 9,
            9, 9, 9, 9, 9
    };

    TensorInfo inputInfo ({ 4 }, ArmnnType);
    TensorInfo outputInfo({ 1, 1, 5, 3 }, ArmnnType);

    armnn::INetworkPtr network = CreateFillNetwork(inputInfo, outputInfo, descriptor);

    BOOST_TEST_CHECKPOINT("create a network");

    std::map<int, std::vector<T>> inputTensorData          = {{ 0, inputData }};
    std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(network), inputTensorData, expectedOutputTensorData, backends);
}

} // anonymous namespace
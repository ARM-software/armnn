//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "CommonTestUtils.hpp"

#include <ResolveType.hpp>

#include <armnn/INetwork.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

namespace
{

armnn::INetworkPtr CreateFullyConnectedNetworkNonConstWeights(const armnn::TensorInfo& inputTensorInfo,
                                                              const armnn::TensorInfo& outputTensorInfo,
                                                              const armnn::TensorInfo& weightsTensorInfo,
                                                              armnn::FullyConnectedDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "Input");
    armnn::IConnectableLayer* weightsInputLayer   = network->AddInputLayer(1, "Weights_Input");
    armnn::IConnectableLayer* fullyConnectedLayer = network->AddFullyConnectedLayer(descriptor,
                                                                                    armnn::EmptyOptional(),
                                                                                    armnn::EmptyOptional(),
                                                                                    "Fully_Connected");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "Output");

    Connect(inputLayer, fullyConnectedLayer, inputTensorInfo, 0, 0);
    Connect(weightsInputLayer, fullyConnectedLayer, weightsTensorInfo, 0, 1);
    Connect(fullyConnectedLayer, outputLayer, outputTensorInfo, 0, 0);

    return network;
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
void FullyConnectedWithDynamicWeightsEndToEnd(const std::vector<armnn::BackendId>& backends)
{
    using namespace armnn;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 2, 3 }, ArmnnType);
    inputTensorInfo.SetQuantizationScale(0.1f);
    inputTensorInfo.SetQuantizationOffset(63);

    armnn::TensorInfo outputTensorInfo({ 1, 2 }, ArmnnType);
    outputTensorInfo.SetQuantizationScale(5.f);
    outputTensorInfo.SetQuantizationOffset(10);

    armnn::TensorInfo weightsTensorInfo({ 2, 6 }, ArmnnType);
    weightsTensorInfo.SetQuantizationScale(0.2f);
    weightsTensorInfo.SetQuantizationOffset(93);

    FullyConnectedDescriptor descriptor;
    descriptor.m_ConstantWeights = false;
    descriptor.m_BiasEnabled     = false;
    descriptor.m_TransposeWeightMatrix = true;

    std::vector<T> inputData {
        -1.2f, 6.1f, -3.5f,
        18.8f, -5.5f, 2.9f
    };

    std::vector<T> weightsData {
        -8.4f, 20.0f, -10.4f, -8, 16.4f, -11.8f,
        23.4f, 10.4f, -14.0f, -3.8f, -11.8f, 11.4f
    };

    std::vector<T> floatExpectedOutputData {
        -107.04f, 110.f
    };
    std::vector<T> expectedOutputData = armnnUtils::QuantizedVector<T>(floatExpectedOutputData);

    armnn::INetworkPtr network = CreateFullyConnectedNetworkNonConstWeights(inputTensorInfo,
                                                                            outputTensorInfo,
                                                                            weightsTensorInfo,
                                                                            descriptor);

    BOOST_TEST_CHECKPOINT("create a network");

    std::map<int, std::vector<T>> inputTensorData    = {{ 0, inputData }, {1, weightsData}};
    std::map<int, std::vector<T>> expectedOutputTensorData = {{ 0, expectedOutputData }};

    EndToEndLayerTestImpl<ArmnnType, ArmnnType>(move(network),
                                                inputTensorData,
                                                expectedOutputTensorData,
                                                backends,
                                                1.0f);
}
} // anonymous namespace

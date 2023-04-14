//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/INetwork.hpp>

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <doctest/doctest.h>

TEST_SUITE("TosaReferenceOptimizedNetwork")
{

TEST_CASE("SimpleSupportedOptimizedNetwork")
{
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));
    armnn::INetworkPtr network(armnn::INetwork::Create());

    auto inputLayer1 = network->AddInputLayer(0, "input_1");
    auto inputLayer2 = network->AddInputLayer(1, "input_2");
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    auto addLayer = network->AddAdditionLayer("add");
    ARMNN_NO_DEPRECATE_WARN_END
    auto outputLayer = network->AddOutputLayer(2, "output");

    armnn::TensorInfo tensorInfo{{4}, armnn::DataType::Float32};

    inputLayer1->GetOutputSlot(0).Connect(addLayer->GetInputSlot(0));
    inputLayer1->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    inputLayer2->GetOutputSlot(0).Connect(addLayer->GetInputSlot(1));
    inputLayer2->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    addLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));
    addLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    std::vector<armnn::BackendId> backends = { "TosaRef" };

    armnn::OptimizerOptionsOpaque optimizedOptions;
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network, backends, runtime->GetDeviceSpec(), optimizedOptions);
    CHECK(optNet);

    armnn::Graph& graph = GetGraphForTesting(optNet.get());

    // Check graph layer sequence to ensure that the network has been replaced with a PreCompiledLayer
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<armnn::InputLayer>,
                        &IsLayerOfType<armnn::InputLayer>,
                        &IsLayerOfType<armnn::PreCompiledLayer>,
                        &IsLayerOfType<armnn::OutputLayer>));
}

}

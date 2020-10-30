//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include <cl/ClBackend.hpp>
#include <neon/NeonBackend.hpp>

#include <Network.hpp>

#include <vector>
#include <string>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(BackendsCompatibility, * boost::unit_test::disabled())

BOOST_AUTO_TEST_CASE(Neon_Cl_DirectCompatibility_Test)
{
    auto neonBackend = std::make_unique<NeonBackend>();
    auto clBackend = std::make_unique<ClBackend>();

    TensorHandleFactoryRegistry registry;
    neonBackend->RegisterTensorHandleFactories(registry);
    clBackend->RegisterTensorHandleFactories(registry);

    const BackendId& neonBackendId = neonBackend->GetId();
    const BackendId& clBackendId = clBackend->GetId();

    BackendsMap backends;
    backends[neonBackendId] = std::move(neonBackend);
    backends[clBackendId] = std::move(clBackend);

    armnn::Graph graph;

    armnn::InputLayer* const inputLayer = graph.AddLayer<armnn::InputLayer>(0, "input");

    inputLayer->SetBackendId(neonBackendId);

    armnn::SoftmaxDescriptor smDesc;
    armnn::SoftmaxLayer* const softmaxLayer1 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax1");
    softmaxLayer1->SetBackendId(clBackendId);

    armnn::SoftmaxLayer* const softmaxLayer2 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax2");
    softmaxLayer2->SetBackendId(neonBackendId);

    armnn::SoftmaxLayer* const softmaxLayer3 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax3");
    softmaxLayer3->SetBackendId(clBackendId);

    armnn::SoftmaxLayer* const softmaxLayer4 = graph.AddLayer<armnn::SoftmaxLayer>(smDesc, "softmax4");
    softmaxLayer4->SetBackendId(neonBackendId);

    armnn::OutputLayer* const outputLayer = graph.AddLayer<armnn::OutputLayer>(0, "output");
    outputLayer->SetBackendId(clBackendId);

    inputLayer->GetOutputSlot(0).Connect(softmaxLayer1->GetInputSlot(0));
    softmaxLayer1->GetOutputSlot(0).Connect(softmaxLayer2->GetInputSlot(0));
    softmaxLayer2->GetOutputSlot(0).Connect(softmaxLayer3->GetInputSlot(0));
    softmaxLayer3->GetOutputSlot(0).Connect(softmaxLayer4->GetInputSlot(0));
    softmaxLayer4->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    graph.TopologicalSort();

    std::vector<std::string> errors;
    auto result = SelectTensorHandleStrategy(graph, backends, registry, true, errors);

    BOOST_TEST(result.m_Error == false);
    BOOST_TEST(result.m_Warning == false);

    OutputSlot& inputLayerOut = inputLayer->GetOutputSlot(0);
    OutputSlot& softmaxLayer1Out = softmaxLayer1->GetOutputSlot(0);
    OutputSlot& softmaxLayer2Out = softmaxLayer2->GetOutputSlot(0);
    OutputSlot& softmaxLayer3Out = softmaxLayer3->GetOutputSlot(0);
    OutputSlot& softmaxLayer4Out = softmaxLayer4->GetOutputSlot(0);

    // Check that the correct factory was selected
    BOOST_TEST(inputLayerOut.GetTensorHandleFactoryId()    == "Arm/Cl/TensorHandleFactory");
    BOOST_TEST(softmaxLayer1Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    BOOST_TEST(softmaxLayer2Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    BOOST_TEST(softmaxLayer3Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    BOOST_TEST(softmaxLayer4Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");

    // Check that the correct strategy was selected
    BOOST_TEST((inputLayerOut.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    BOOST_TEST((softmaxLayer1Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    BOOST_TEST((softmaxLayer2Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    BOOST_TEST((softmaxLayer3Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    BOOST_TEST((softmaxLayer4Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));

    graph.AddCompatibilityLayers(backends, registry);

    // Test for copy layers
    int copyCount= 0;
    graph.ForEachLayer([&copyCount](Layer* layer)
    {
        if (layer->GetType() == LayerType::MemCopy)
        {
            copyCount++;
        }
    });
    BOOST_TEST(copyCount == 0);

    // Test for import layers
    int importCount= 0;
    graph.ForEachLayer([&importCount](Layer *layer)
    {
        if (layer->GetType() == LayerType::MemImport)
        {
            importCount++;
        }
    });
    BOOST_TEST(importCount == 0);
}

BOOST_AUTO_TEST_SUITE_END()

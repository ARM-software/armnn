//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if defined(ARMCOMPUTECL_ENABLED)
#include <cl/ClBackend.hpp>
#endif
#if defined(ARMCOMPUTENEON_ENABLED)
#include <neon/NeonBackend.hpp>
#endif
#include <reference/RefBackend.hpp>
#include <armnn/BackendHelper.hpp>

#include <Network.hpp>

#include <doctest/doctest.h>

#include <vector>
#include <string>

using namespace armnn;

#if defined(ARMCOMPUTENEON_ENABLED) && defined(ARMCOMPUTECL_ENABLED)

TEST_SUITE("BackendsCompatibility")
{
// Partially disabled Test Suite
TEST_CASE("Neon_Cl_DirectCompatibility_Test")
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

    CHECK(result.m_Error == false);
    CHECK(result.m_Warning == false);

    // OutputSlot& inputLayerOut = inputLayer->GetOutputSlot(0);
    // OutputSlot& softmaxLayer1Out = softmaxLayer1->GetOutputSlot(0);
    // OutputSlot& softmaxLayer2Out = softmaxLayer2->GetOutputSlot(0);
    // OutputSlot& softmaxLayer3Out = softmaxLayer3->GetOutputSlot(0);
    // OutputSlot& softmaxLayer4Out = softmaxLayer4->GetOutputSlot(0);

    // // Check that the correct factory was selected
    // CHECK(inputLayerOut.GetTensorHandleFactoryId()    == "Arm/Cl/TensorHandleFactory");
    // CHECK(softmaxLayer1Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    // CHECK(softmaxLayer2Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    // CHECK(softmaxLayer3Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");
    // CHECK(softmaxLayer4Out.GetTensorHandleFactoryId() == "Arm/Cl/TensorHandleFactory");

    // // Check that the correct strategy was selected
    // CHECK((inputLayerOut.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    // CHECK((softmaxLayer1Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    // CHECK((softmaxLayer2Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    // CHECK((softmaxLayer3Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));
    // CHECK((softmaxLayer4Out.GetEdgeStrategyForConnection(0) == EdgeStrategy::DirectCompatibility));

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
    // CHECK(copyCount == 0);

    // Test for import layers
    int importCount= 0;
    graph.ForEachLayer([&importCount](Layer *layer)
    {
        if (layer->GetType() == LayerType::MemImport)
        {
            importCount++;
        }
    });
    // CHECK(importCount == 0);
}

}
#endif

TEST_SUITE("BackendCapability")
{

namespace
{

#if defined(ARMNNREF_ENABLED) || defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
void CapabilityTestHelper(BackendCapabilities &capabilities,
                          std::vector<std::pair<std::string, bool>> capabilityVector)
{
    for (auto pair : capabilityVector)
    {
        CHECK_MESSAGE(armnn::HasCapability(pair.first, capabilities),
                        pair.first << " capability was not been found");
        CHECK_MESSAGE(armnn::HasCapability(BackendOptions::BackendOption{pair.first, pair.second}, capabilities),
                        pair.first << " capability set incorrectly");
    }
}
#endif

#if defined(ARMNNREF_ENABLED)

TEST_CASE("Ref_Backends_Unknown_Capability_Test")
{
    auto refBackend = std::make_unique<RefBackend>();
    auto refCapabilities = refBackend->GetCapabilities();

    armnn::BackendOptions::BackendOption AsyncExecutionFalse{"AsyncExecution", false};
    CHECK(!armnn::HasCapability(AsyncExecutionFalse, refCapabilities));

    armnn::BackendOptions::BackendOption AsyncExecutionInt{"AsyncExecution", 50};
    CHECK(!armnn::HasCapability(AsyncExecutionFalse, refCapabilities));

    armnn::BackendOptions::BackendOption AsyncExecutionFloat{"AsyncExecution", 0.0f};
    CHECK(!armnn::HasCapability(AsyncExecutionFloat, refCapabilities));

    armnn::BackendOptions::BackendOption AsyncExecutionString{"AsyncExecution", "true"};
    CHECK(!armnn::HasCapability(AsyncExecutionString, refCapabilities));

    CHECK(!armnn::HasCapability("Telekinesis", refCapabilities));
    armnn::BackendOptions::BackendOption unknownCapability{"Telekinesis", true};
    CHECK(!armnn::HasCapability(unknownCapability, refCapabilities));
}

TEST_CASE ("Ref_Backends_Capability_Test")
{
    auto refBackend = std::make_unique<RefBackend>();
    auto refCapabilities = refBackend->GetCapabilities();

    CapabilityTestHelper(refCapabilities,
                         {{"NonConstWeights", true},
                          {"AsyncExecution", true},
                          {"ProtectedContentAllocation", false},
                          {"ConstantTensorsAsInputs", true},
                          {"PreImportIOTensors", true},
                          {"ExternallyManagedMemory", true},
                          {"MultiAxisPacking", false}});
}

#endif

#if defined(ARMCOMPUTENEON_ENABLED)

TEST_CASE ("Neon_Backends_Capability_Test")
{
    auto neonBackend = std::make_unique<NeonBackend>();
    auto neonCapabilities = neonBackend->GetCapabilities();

    CapabilityTestHelper(neonCapabilities,
                         {{"NonConstWeights", false},
                          {"AsyncExecution", false},
                          {"ProtectedContentAllocation", false},
                          {"ConstantTensorsAsInputs", false},
                          {"PreImportIOTensors", false},
                          {"ExternallyManagedMemory", true},
                          {"MultiAxisPacking", false}});
}

#endif

#if defined(ARMCOMPUTECL_ENABLED)

TEST_CASE ("Cl_Backends_Capability_Test")
{
    auto clBackend = std::make_unique<ClBackend>();
    auto clCapabilities = clBackend->GetCapabilities();

    CapabilityTestHelper(clCapabilities,
                         {{"NonConstWeights", false},
                          {"AsyncExecution", false},
                          {"ProtectedContentAllocation", true},
                          {"ConstantTensorsAsInputs", false},
                          {"PreImportIOTensors", false},
                          {"ExternallyManagedMemory", true},
                          {"MultiAxisPacking", false}});
}

#endif
}
}

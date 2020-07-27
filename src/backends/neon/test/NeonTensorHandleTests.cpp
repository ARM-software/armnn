//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonTensorHandleFactory.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonTensorHandleTests)
using namespace armnn;

BOOST_AUTO_TEST_CASE(NeonTensorHandleGetCapabilitiesNoPadding)
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    SoftmaxDescriptor descriptor;
    descriptor.m_Beta = 1.0f;
    IConnectableLayer* softmax = network->AddSoftmaxLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(softmax->GetInputSlot(0));
    softmax->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         softmax,
                                                                         CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.empty());

    // No padding required for Softmax
    capabilities = handleFactory.GetCapabilities(softmax, output, CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.empty());
}

BOOST_AUTO_TEST_CASE(NeonTensorHandleGetCapabilitiesPadding)
{
    std::shared_ptr<NeonMemoryManager> memoryManager = std::make_shared<NeonMemoryManager>();
    NeonTensorHandleFactory handleFactory(memoryManager);

    INetworkPtr network(INetwork::Create());

    // Add the layers
    IConnectableLayer* input = network->AddInputLayer(0);
    Pooling2dDescriptor descriptor;
    IConnectableLayer* pooling = network->AddPooling2dLayer(descriptor);
    IConnectableLayer* output = network->AddOutputLayer(2);

    // Establish connections
    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // No padding required for input
    std::vector<Capability> capabilities = handleFactory.GetCapabilities(input,
                                                                         pooling,
                                                                         CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.empty());

    // No padding required for output
    capabilities = handleFactory.GetCapabilities(output, nullptr, CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.empty());

    // Padding required for Pooling2d
    capabilities = handleFactory.GetCapabilities(pooling, output, CapabilityClass::PaddingRequired);
    BOOST_TEST(capabilities.size() == 1);
    BOOST_TEST((capabilities[0].m_CapabilityClass == CapabilityClass::PaddingRequired));
    BOOST_TEST(capabilities[0].m_Value);
}

BOOST_AUTO_TEST_SUITE_END()

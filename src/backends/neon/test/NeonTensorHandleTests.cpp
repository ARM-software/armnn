//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Graph.hpp>
#include <Network.hpp>

#include <neon/NeonTensorHandle.hpp>
#include <neon/NeonTensorHandleFactory.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <test/GraphUtils.hpp>

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

BOOST_AUTO_TEST_CASE(ConcatOnXorYSubTensorsNoPaddinRequiredTest)
{
    armnn::INetworkPtr net(armnn::INetwork::Create());

    // Set up tensor infos
    const armnn::TensorInfo inputInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo intermediateInfo = armnn::TensorInfo({2, 3, 2, 2}, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo = armnn::TensorInfo({2, 3, 4, 2}, armnn::DataType::Float32);

    armnn::ElementwiseUnaryDescriptor descriptor(armnn::UnaryOperation::Abs);

    // Create the network
    armnn::IConnectableLayer* const input0Layer = net->AddInputLayer(0, "input_0");
    input0Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* elementwiseUnaryLayer0 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseUnary_0");
    elementwiseUnaryLayer0->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input0Layer->GetOutputSlot(0).Connect(elementwiseUnaryLayer0->GetInputSlot(0));

    armnn::IConnectableLayer* const input1Layer = net->AddInputLayer(1, "input_1");
    input1Layer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    armnn::IConnectableLayer* elementwiseUnaryLayer1 = net->AddElementwiseUnaryLayer(descriptor, "elementwiseUnary_1");
    elementwiseUnaryLayer1->GetOutputSlot(0).SetTensorInfo(intermediateInfo);
    input1Layer->GetOutputSlot(0).Connect(elementwiseUnaryLayer1->GetInputSlot(0));

    std::array<armnn::TensorShape, 2> concatInputShapes = { intermediateInfo.GetShape(), intermediateInfo.GetShape() };
    armnn::IConnectableLayer* const concatLayer = net->AddConcatLayer(armnn::CreateDescriptorForConcatenation(
        concatInputShapes.begin(), concatInputShapes.end(), 2), "concatenation");
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    elementwiseUnaryLayer0->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    elementwiseUnaryLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));

    armnn::IConnectableLayer* const outputLayer = net->AddOutputLayer(0, "output");
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    armnn::IOptimizedNetworkPtr optimizedNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());

    const armnn::Graph& theGraph = static_cast<armnn::OptimizedNetwork*>(optimizedNet.get())->GetGraph();

    // Load graph into runtime
    armnn::NetworkId networkIdentifier;
    runtime->LoadNetwork(networkIdentifier, std::move(optimizedNet));

    // now check the concat how many sub-tensors it is using..
    auto TraceSubTensorHandleAncestry = [](armnn::ITensorHandle* const subTensorHandle)
    {
        if (subTensorHandle && subTensorHandle->GetParent())
        {
            return true;
        }
        return false;
    };

    for (auto&& layer : theGraph)
    {
        if(layer->GetType() == armnn::LayerType::Concat)
        {
            unsigned int numberOfSubTensors = 0;
            for (unsigned int i = 0; i < layer->GetNumInputSlots(); ++i)
            {
                const armnn::OutputSlot* slot = layer->GetInputSlot(i).GetConnectedOutputSlot();
                if (TraceSubTensorHandleAncestry(slot->GetOutputHandler().GetData()))
                {
                    ++numberOfSubTensors;
                }
            }
            // sub-tensors should be supported in this configuration
            BOOST_CHECK(numberOfSubTensors > 0);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

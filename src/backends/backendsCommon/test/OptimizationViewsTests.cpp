//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <CommonTestUtils.hpp>
#include "MockBackend.hpp"

#include <Graph.hpp>
#include <Network.hpp>
#include <SubgraphViewSelector.hpp>

#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/SubgraphView.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <doctest/doctest.h>

using namespace armnn;

void CheckLayers(Graph& graph)
{
    unsigned int m_inputLayerCount = 0, m_outputLayerCount = 0, m_addLayerCount = 0;
    for(auto layer : graph)
    {
        switch(layer->GetType())
        {
            case LayerType::Input:
                ++m_inputLayerCount;
                CHECK((layer->GetName() == std::string("inLayer0") ||
                            layer->GetName() == std::string("inLayer1")));
                break;
            // The Addition layer should become a PreCompiled Layer after Optimisation
            case LayerType::PreCompiled:
                ++m_addLayerCount;
                CHECK(std::string(layer->GetName()) == "pre-compiled");
                break;
            case LayerType::Output:
                ++m_outputLayerCount;
                CHECK(std::string(layer->GetName()) == "outLayer");
                break;
            default:
                //Fail for anything else
                CHECK(false);
        }
    }
    CHECK(m_inputLayerCount == 2);
    CHECK(m_outputLayerCount == 1);
    CHECK(m_addLayerCount == 1);
}

TEST_SUITE("OptimizationViewsTestSuite")
{
TEST_CASE("OptimizedViewsSubgraphLayerCount")
{
    OptimizationViews view;
    // Construct a graph with 3 layers
    Graph baseGraph;

    Layer* const inputLayer = baseGraph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    PreCompiledDescriptor substitutionLayerDescriptor(1, 1);
    Layer* const convLayer1 = baseGraph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = baseGraph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");
    Layer* const substitutableCompiledLayer =
            baseGraph.AddLayer<PreCompiledLayer>(substitutionLayerDescriptor, "pre-compiled");

    Layer* const outputLayer = baseGraph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Subgraph for a failed layer
    SubgraphViewSelector::SubgraphViewPtr failedSubgraph =
        CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                               CreateOutputsFrom({convLayer1}),
                               {convLayer1});
    // Subgraph for an untouched layer
    SubgraphViewSelector::SubgraphViewPtr untouchedSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer2}),
                                   CreateOutputsFrom({convLayer2}),
                                   {convLayer2});
    // Subgraph for a substitutable layer
    SubgraphViewSelector::SubgraphViewPtr substitutableSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {substitutableCompiledLayer});
    // Create a Graph containing a layer to substitute in
    Graph substitutableGraph;
    Layer* const substitutionpreCompiledLayer =
            substitutableGraph.AddLayer<PreCompiledLayer>(substitutionLayerDescriptor, "pre-compiled");

    // Subgraph for a substitution layer
    SubgraphViewSelector::SubgraphViewPtr substitutionSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({substitutionpreCompiledLayer}),
                                   CreateOutputsFrom({substitutionpreCompiledLayer}),
                                   {substitutionpreCompiledLayer});

    // Sub in the graph
    baseGraph.SubstituteSubgraph(*substitutableSubgraph, *substitutionSubgraph);

    view.AddFailedSubgraph(SubgraphView(*failedSubgraph));
    view.AddUntouchedSubgraph(SubgraphView(*untouchedSubgraph));

    SubgraphViewSelector::SubgraphViewPtr baseSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {substitutionpreCompiledLayer});
    view.AddSubstitution({*baseSubgraph, *substitutionSubgraph});

    // Construct original subgraph to compare against
    SubgraphViewSelector::SubgraphViewPtr originalSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
            CreateOutputsFrom({convLayer2}),
            {convLayer1, convLayer2, substitutionpreCompiledLayer});

    CHECK(view.Validate(*originalSubgraph));
}


TEST_CASE("OptimizedViewsSubgraphLayerCountUsingGetINetwork")
{
    OptimizationViews view;

    IConnectableLayer* const inputLayer = view.GetINetwork()->AddInputLayer(0, "input");

    DepthwiseConvolution2dDescriptor convDescriptor;
    PreCompiledDescriptor substitutionLayerDescriptor(1, 1);
    CompiledBlobPtr blobPtr;
    BackendId backend = Compute::CpuRef;

    Layer* convLayer1 = PolymorphicDowncast<Layer*>(
        view.GetINetwork()->AddDepthwiseConvolution2dLayer(convDescriptor,
                                                           ConstTensor(),
                                                           Optional<ConstTensor>(),
                                                           "conv1"));

    Layer* convLayer2 = PolymorphicDowncast<Layer*>(
        view.GetINetwork()->AddDepthwiseConvolution2dLayer(convDescriptor,
                                                           ConstTensor(),
                                                           Optional<ConstTensor>(),
                                                           "conv2"));

    IConnectableLayer* const outputLayer = view.GetINetwork()->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Subgraph for a failed layer
    SubgraphViewSelector::SubgraphViewPtr failedSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                                  CreateOutputsFrom({convLayer1}),
                                                                                  {convLayer1});
    // Subgraph for an untouched layer
    SubgraphViewSelector::SubgraphViewPtr untouchedSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer2}),
                                                                                     CreateOutputsFrom({convLayer2}),
                                                                                     {convLayer2});

    // Create a Network containing a layer to substitute in
    NetworkImpl net;
    Layer* substitutionpreCompiledLayer = PolymorphicDowncast<Layer*>(
        net.AddPrecompiledLayer(substitutionLayerDescriptor, std::move(blobPtr), backend));

    // Subgraph for a substitution layer
    SubgraphViewSelector::SubgraphViewPtr substitutionSubgraph =
        CreateSubgraphViewFrom(CreateInputsFrom({substitutionpreCompiledLayer}),
                                                CreateOutputsFrom({substitutionpreCompiledLayer}),
                                                {substitutionpreCompiledLayer});

    view.AddFailedSubgraph(SubgraphView(*failedSubgraph));
    view.AddUntouchedSubgraph(SubgraphView(*untouchedSubgraph));

    SubgraphViewSelector::SubgraphViewPtr baseSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                                CreateOutputsFrom({convLayer2}),
                                                                                {substitutionpreCompiledLayer});
    view.AddSubstitution({*baseSubgraph, *substitutionSubgraph});

    // Construct original subgraph to compare against
    SubgraphViewSelector::SubgraphViewPtr originalSubgraph =
        CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                CreateOutputsFrom({convLayer2}),
                                                {convLayer1, convLayer2, substitutionpreCompiledLayer});

    CHECK(view.Validate(*originalSubgraph));
}

TEST_CASE("OptimizedViewsSubgraphLayerCountFailValidate")
{
    OptimizationViews view;
    // Construct a graph with 3 layers
    Graph baseGraph;

    Layer* const inputLayer = baseGraph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    PreCompiledDescriptor substitutionLayerDescriptor(1, 1);
    Layer* const convLayer1 = baseGraph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = baseGraph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");
    Layer* const substitutableCompiledLayer =
            baseGraph.AddLayer<PreCompiledLayer>(substitutionLayerDescriptor, "pre-compiled");

    Layer* const outputLayer = baseGraph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Subgraph for an untouched layer
    SubgraphViewSelector::SubgraphViewPtr untouchedSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer2}),
                                   CreateOutputsFrom({convLayer2}),
                                   {convLayer2});
    // Subgraph for a substitutable layer
    SubgraphViewSelector::SubgraphViewPtr substitutableSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {substitutableCompiledLayer});
    // Create a Graph containing a layer to substitute in
    Graph substitutableGraph;
    Layer* const substitutionpreCompiledLayer =
            substitutableGraph.AddLayer<PreCompiledLayer>(substitutionLayerDescriptor, "pre-compiled");

    // Subgraph for a substitution layer
    SubgraphViewSelector::SubgraphViewPtr substitutionSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({substitutionpreCompiledLayer}),
                                   CreateOutputsFrom({substitutionpreCompiledLayer}),
                                   {substitutionpreCompiledLayer});

    // Sub in the graph
    baseGraph.SubstituteSubgraph(*substitutableSubgraph, *substitutionSubgraph);

    view.AddUntouchedSubgraph(SubgraphView(*untouchedSubgraph));

    SubgraphViewSelector::SubgraphViewPtr baseSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {substitutionpreCompiledLayer});
    view.AddSubstitution({*baseSubgraph, *substitutionSubgraph});

    // Construct original subgraph to compare against
    SubgraphViewSelector::SubgraphViewPtr originalSubgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {convLayer1, convLayer2, substitutionpreCompiledLayer});

    // Validate should fail as convLayer1 is not counted
    CHECK(!view.Validate(*originalSubgraph));
}

TEST_CASE("OptimizeViewsValidateDeviceMockBackend")
{
    // build up the structure of the network
    armnn::INetworkPtr net(armnn::INetwork::Create());

    armnn::IConnectableLayer* input = net->AddInputLayer(0, "inLayer0");
    armnn::IConnectableLayer* input1 = net->AddInputLayer(1, "inLayer1");

    armnn::IConnectableLayer* addition = net->AddAdditionLayer("addLayer");

    armnn::IConnectableLayer* output = net->AddOutputLayer(0, "outLayer");

    input->GetOutputSlot(0).Connect(addition->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(addition->GetInputSlot(1));
    addition->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    input1->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));
    addition->GetOutputSlot(0).SetTensorInfo(armnn::TensorInfo({ 1, 1, 4, 4 }, armnn::DataType::Float32));

    armnn::MockBackendInitialiser initialiser;
    armnn::IRuntime::CreationOptions options;
    armnn::IRuntimePtr runtime(armnn::IRuntime::Create(options));

    std::vector<armnn::BackendId> backends = { MockBackend().GetIdStatic() };
    armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*net, backends, runtime->GetDeviceSpec());
    CHECK(optNet);

    // Check the optimised graph
    armnn::Graph& graph = GetGraphForTesting(optNet.get());
    CheckLayers(graph);
}

}
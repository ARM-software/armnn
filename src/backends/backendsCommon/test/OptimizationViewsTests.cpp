//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include <armnn/ArmNN.hpp>
#include <Graph.hpp>
#include <SubgraphView.hpp>
#include <SubgraphViewSelector.hpp>
#include <backendsCommon/OptimizationViews.hpp>
#include <Network.hpp>

#include "CommonTestUtils.hpp"

using namespace armnn;

BOOST_AUTO_TEST_SUITE(OptimizationViewsTestSuite)

BOOST_AUTO_TEST_CASE(OptimizedViewsSubgraphLayerCount)
{
    OptimizationViews view;
    // Construct a graph with 3 layers
    Graph& baseGraph = view.GetGraph();

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

    BOOST_CHECK(view.Validate(*originalSubgraph));
}

BOOST_AUTO_TEST_CASE(OptimizedViewsSubgraphLayerCountFailValidate)
{
    OptimizationViews view;
    // Construct a graph with 3 layers
    Graph& baseGraph = view.GetGraph();

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
    BOOST_CHECK(!view.Validate(*originalSubgraph));
}

BOOST_AUTO_TEST_SUITE_END()
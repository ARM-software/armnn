//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Graph.hpp>
#include <SubgraphViewSelector.hpp>

#include <armnn/backends/OptimizationViews.hpp>
#include <armnn/backends/SubgraphView.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <doctest/doctest.h>

#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <chrono>
#include <numeric>

using namespace armnn;

namespace
{

bool AreAnySubgraphLayersPresentInGraph(const SubgraphView::IConnectableLayers &subgraphLayers, const Graph &graph)
{
    for(auto&& layer : subgraphLayers)
    {
        auto posInGraph = std::find(graph.begin(), graph.end(), layer);
        if(posInGraph != graph.end())
        {
            return true;
        }
    }

    return false;
}

//
// this helper only works if all layers where the inputs connect to are not selected
//
SubgraphView::InputSlots CreateInputsFrom(const std::vector<Layer*>& layers)
{
    SubgraphView::InputSlots result;
    for (auto&& layer : layers)
    {
        for (auto&& it = layer->BeginInputSlots(); it != layer->EndInputSlots(); ++it)
        {
            result.push_back(&(*it));
        }
    }
    return result;
}

/// Duplication for IConnectableLayer
SubgraphView::IInputSlots CreateIInputsFrom(const std::vector<armnn::IConnectableLayer*>& layers)
{
    SubgraphView::IInputSlots result;
    for (auto&& layer : layers)
    {
        for (unsigned int i = 0 ; i < layer->GetNumInputSlots(); ++i)
        {
            result.push_back(&(layer->GetInputSlot(i)));
        }
    }
    return result;
}

//
// this helper only works if all layers where the outputs connect to are not selected
//
SubgraphView::OutputSlots CreateOutputsFrom(const std::vector<Layer*>& layers)
{
    SubgraphView::OutputSlots result;
    for (auto && layer : layers)
    {
        for (auto&& it = layer->BeginOutputSlots(); it != layer->EndOutputSlots(); ++it)
        {
            result.push_back(&(*it));
        }
    }
    return result;
}

/// Duplication for IConnectableLayer
SubgraphView::IOutputSlots CreateIOutputsFrom(const std::vector<armnn::IConnectableLayer*>& layers)
{
    SubgraphView::IOutputSlots result;
    for (auto&& layer: layers)
    {
        for (unsigned int i = 0; i < layer->GetNumOutputSlots(); ++i)
        {
            result.push_back(&(layer->GetOutputSlot(i)));
        }
    }
    return result;
}

//
// this takes the inputs, outputs and layers as a copy and the move these copies into the
// resulting subgraph, so the pass by value is intentional
//
SubgraphViewSelector::SubgraphViewPtr CreateSubgraphViewFrom(SubgraphView::InputSlots&& inputs,
                                                             SubgraphView::OutputSlots&& outputs,
                                                             SubgraphView::Layers&& layers)
{
    return std::make_unique<SubgraphView>(std::move(inputs), std::move(outputs), std::move(layers));
}

SubgraphViewSelector::SubgraphViewPtr CreateSubgraphViewFrom(SubgraphView::IConnectableLayers&& layers,
                                                             SubgraphView::IInputSlots&& inputs,
                                                             SubgraphView::IOutputSlots&& outputs)
{
    return std::make_unique<SubgraphView>(std::move(layers), std::move(inputs), std::move(outputs));
}

template <typename T, typename Iterator>
std::vector<T> ToSortedArray(Iterator begin, Iterator end)
{
    std::vector<T> result(begin, end);
    std::sort(result.begin(), result.end());
    return result;
}

template <typename T>
void CompareVectors(const std::vector<T>& result, const std::vector<T>& expected)
{
    CHECK(std::equal(result.begin(), result.end(), expected.begin(), expected.end()));
}

void CompareSubgraphViews(SubgraphViewSelector::SubgraphViewPtr& result,
                          SubgraphViewSelector::SubgraphViewPtr& expected)
{
    // expect both to be valid subgraphs
    CHECK((result.get() != nullptr));
    CHECK((expected.get() != nullptr));

    if (result.get() != nullptr && expected.get() != nullptr)
    {
        CHECK(result->GetIInputSlots().size() == expected->GetIInputSlots().size());
        CHECK(result->GetIOutputSlots().size() == expected->GetIOutputSlots().size());
        CHECK(result->GetIConnectableLayers().size() == expected->GetIConnectableLayers().size());

        auto resultLayers = ToSortedArray<IConnectableLayer*>(result->GetIConnectableLayers().begin(),
                                                   result->GetIConnectableLayers().end());
        auto expectedLayers = ToSortedArray<IConnectableLayer*>(expected->GetIConnectableLayers().begin(),
                                                     expected->GetIConnectableLayers().end());
        CompareVectors(resultLayers, expectedLayers);

        auto resultInputs = ToSortedArray<IInputSlot *>(result->GetIInputSlots().begin(),
                                                       result->GetIInputSlots().end());
        auto expectedInputs = ToSortedArray<IInputSlot *>(expected->GetIInputSlots().begin(),
                                                         expected->GetIInputSlots().end());
        CompareVectors(resultInputs, expectedInputs);

        auto resultOutputs = ToSortedArray<IOutputSlot *>(result->GetIOutputSlots().begin(),
                                                         result->GetIOutputSlots().end());
        auto expectedOutputs = ToSortedArray<IOutputSlot *>(expected->GetIOutputSlots().begin(),
                                                           expected->GetIOutputSlots().end());
        CompareVectors(resultOutputs, expectedOutputs);
    }
}

} // namespace <anonymous>

TEST_SUITE("SubgraphViewBackwardCompatibilityTests")
{
// Test that SubraphView has been converted to using IConnectableLayer/IInputSlot/IOutputSlot
// in a backward compatible manner from  ILayer/InputSlot/OutputSlot
TEST_CASE("SubgraphViewIterators")
{
    INetworkPtr net(INetwork::Create());
    IConnectableLayer* layer = net->AddInputLayer(1, "input");

    SubgraphView subgraph{layer};

    // cbeginIConnectable() and cendIConnectable()
    bool found = false;
    if (std::find(subgraph.cbeginIConnectable(), subgraph.cendIConnectable(), layer)
        != subgraph.cendIConnectable())
    {
        found = true;
    }
    CHECK(found);
    found = false;

    // beginIConnectable() and endIConnectable()
    if (std::find(subgraph.beginIConnectable(), subgraph.endIConnectable(), layer)
        != subgraph.endIConnectable())
    {
        found = true;
    }
    CHECK(found);
    found = false;

    // GetIConnectableLayers returns IConnectableLayers initialized when calling constructor given IConnectableLayers
    const SubgraphView::IConnectableLayers& subgraphLayers = subgraph.GetIConnectableLayers();
    for (auto& iConnectableLayer : subgraphLayers)
    {
        if (std::string(iConnectableLayer->GetName()) == "input")
        {
            found = true;
        }
    }
    CHECK(found);
    found = false;

    // Test GetLayers returns layers initialized when calling constructor given IConnectableLayers
    ARMNN_NO_DEPRECATE_WARN_BEGIN
    const SubgraphView::Layers& subgraphLayersOld = subgraph.GetLayers();
    ARMNN_NO_DEPRECATE_WARN_END
    for (auto& layerOld : subgraphLayersOld)
    {
        if (std::string(layerOld->GetName()) == "input")
        {
            found = true;
        }
    }
    CHECK(found);
}

TEST_CASE("SubgraphViewSlots")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom({},
                                                                            CreateIInputsFrom({convLayer1}),
                                                                            CreateIOutputsFrom({convLayer2}));

    // Test that both old and new are initialized
    CHECK(subgraph->GetIInputSlots().size() == 1);
    CHECK(subgraph->GetIOutputSlots().size() == 1);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    CHECK(subgraph->GetInputSlots().size() == 1);
    CHECK(subgraph->GetOutputSlots().size() == 1);

    // Check old and new pointing to same address
    CHECK(subgraph->GetOutputSlot(0) == subgraph->GetIOutputSlot(0));
    CHECK(subgraph->GetInputSlot(0) == subgraph->GetIInputSlot(0));
    ARMNN_NO_DEPRECATE_WARN_END

}

TEST_CASE("SubgraphViewConstructors")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph =
            CreateSubgraphViewFrom({inputLayer, convLayer1, convLayer2, outputLayer},
                                   CreateIInputsFrom({convLayer1}),
                                   CreateIOutputsFrom({convLayer2}));

    // Copy Constructor
    SubgraphView subgraph2(*subgraph.get());
    CHECK(subgraph->GetIConnectableLayers() == subgraph2.GetIConnectableLayers());
    CHECK(subgraph->GetIInputSlots() == subgraph2.GetIInputSlots());
    CHECK(subgraph->GetIOutputSlots() == subgraph2.GetIOutputSlots());

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    CHECK(subgraph->GetLayers() == subgraph2.GetLayers());
    CHECK(subgraph->GetInputSlots() == subgraph2.GetInputSlots());
    CHECK(subgraph->GetOutputSlots() == subgraph2.GetOutputSlots());
    ARMNN_NO_DEPRECATE_WARN_END

    // Move Constructor
    SubgraphView subgraph3(std::move(subgraph2));
    CHECK(subgraph->GetIConnectableLayers() == subgraph3.GetIConnectableLayers());
    CHECK(subgraph->GetIInputSlots() == subgraph3.GetIInputSlots());
    CHECK(subgraph->GetIOutputSlots() == subgraph3.GetIOutputSlots());

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    CHECK(subgraph->GetLayers() == subgraph3.GetLayers());
    CHECK(subgraph->GetInputSlots() == subgraph3.GetInputSlots());
    CHECK(subgraph->GetOutputSlots() == subgraph3.GetOutputSlots());
    ARMNN_NO_DEPRECATE_WARN_END

    // Clear
    subgraph.get()->Clear();
    CHECK(subgraph->GetIConnectableLayers().size() == 0);
    CHECK(subgraph->GetIInputSlots().size() == 0);
    CHECK(subgraph->GetIOutputSlots().size() == 0);
}

} // SubgraphViewBackwardCompatibilityTests Test Suite end

TEST_SUITE("SubgraphSubstitution")
{
TEST_CASE("SingleInputSingleOutput")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph =
            CreateSubgraphViewFrom({},
                                   CreateIInputsFrom({convLayer1}),
                                   CreateIOutputsFrom({convLayer2}));

    // Save sub-graph connections for comparison after substitution
    // Using GetIInputSlot/GetIIOutputSlot functions
    IOutputSlot* subgraphInputConn = subgraph->GetIInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetIOutputSlot(0)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(1, 1);

    IConnectableLayer* const preCompiledLayer =
            graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

TEST_CASE("SingleInputSingleOutputAddPrecompiledLayerSubstituteSubgraph1")
{
    // Construct graph.
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                            CreateOutputsFrom({convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn = subgraph->GetIInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetIOutputSlot(0)->GetConnection(0);

    PreCompiledDescriptor preCompiledDescriptor(1, 1);
    CompiledBlobPtr compiledBlobPtr;
    BackendId backend = Compute::CpuRef;

    // Construct dummy pre-compiled layer
    INetworkPtr network = INetwork::Create();
    IConnectableLayer* preCompiledLayer = network->AddPrecompiledLayer(preCompiledDescriptor,
                                                                       std::move(compiledBlobPtr),
                                                                       backend);

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

TEST_CASE("SingleInputSingleOutputAddPrecompiledLayerSubstituteSubgraph2")
{
    // Construct graph.
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                            CreateOutputsFrom({convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn = subgraph->GetIInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetIOutputSlot(0)->GetConnection(0);

    PreCompiledDescriptor preCompiledDescriptor(1, 1);
    CompiledBlobPtr compiledBlobPtr;
    BackendId backend = Compute::CpuRef;

    // Construct dummy pre-compiled layer
    INetworkPtr network = INetwork::Create();
    IConnectableLayer* preCompiledLayer = network->AddPrecompiledLayer(preCompiledDescriptor,
                                                                       std::move(compiledBlobPtr),
                                                                       backend);
    SubgraphView substituteSubgraph(preCompiledLayer);

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, substituteSubgraph);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

TEST_CASE("SingleInputSingleOutputSubstituteGraph")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                   CreateOutputsFrom({convLayer2}),
                                   {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn = subgraph->GetIInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetIOutputSlot(0)->GetConnection(0);

    // Construct second graph with a single pre-compiled layer
    Graph substituteGraph;
    PreCompiledDescriptor preCompiledDescriptor(1, 1);
    Layer* const preCompiledLayer = substituteGraph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    SubgraphViewSelector::SubgraphViewPtr substituteSubgraph =
                                          CreateSubgraphViewFrom(CreateInputsFrom({preCompiledLayer}),
                                                                 CreateOutputsFrom({preCompiledLayer}),
                                                                 {preCompiledLayer});
    // Substitute subgraph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, *substituteSubgraph);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

TEST_CASE("MultiInputSingleOutput")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    ViewsDescriptor splitterDescriptor(2);
    Layer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    OriginsDescriptor concatDescriptor(2);
    Layer* const concatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(convLayer2->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    auto subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1, convLayer2}),
                                                                  CreateOutputsFrom({concatLayer}),
                                                                   {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetIInputSlot(0)->GetConnection();
    IOutputSlot* subgraphInputConn2 = subgraph->GetIInputSlot(1)->GetConnection();

    IInputSlot* subgraphOutputConn = subgraph->GetIOutputSlot(0)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(2, 1);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);
    CHECK_EQ(preCompiledLayer->GetInputSlot(1).GetConnection(), subgraphInputConn2);

    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

TEST_CASE("SingleInputMultiOutput")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");
    OriginsDescriptor concatDescriptor(2);
    Layer* const concatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");
    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    ViewsDescriptor splitterDescriptor(2);
    Layer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(convLayer2->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({splitterLayer}),
                                   CreateOutputsFrom({convLayer1, convLayer2}),
                                   {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetIInputSlot(0)->GetConnection();

    IInputSlot* subgraphOutputConn1 = subgraph->GetIOutputSlot(0)->GetConnection(0);
    IInputSlot* subgraphOutputConn2 = subgraph->GetIOutputSlot(1)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(1, 2);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);

    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(1).GetConnection(0), subgraphOutputConn2);
}

TEST_CASE("MultiInputMultiOutput")
{
    // Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    ViewsDescriptor splitterDescriptor(2);
    Layer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    OriginsDescriptor concatDescriptor(2);
    Layer* const concatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(convLayer2->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(1));
    concatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph =
            CreateSubgraphViewFrom(CreateInputsFrom({convLayer1, convLayer2}),
                                   CreateOutputsFrom({convLayer1, convLayer2}),
                                   {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetIInputSlot(0)->GetConnection();
    IOutputSlot* subgraphInputConn2 = subgraph->GetIInputSlot(1)->GetConnection();

    IInputSlot* subgraphOutputConn1 = subgraph->GetIOutputSlot(0)->GetConnection(0);
    IInputSlot* subgraphOutputConn2 = subgraph->GetIOutputSlot(1)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(2, 2);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);
    CHECK_EQ(preCompiledLayer->GetInputSlot(1).GetConnection(), subgraphInputConn2);

    CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
    CHECK_EQ(preCompiledLayer->GetOutputSlot(1).GetConnection(0), subgraphOutputConn2);
}

TEST_CASE("EraseReplacedIConnectableLayers")
{
    // Construct graph
    Graph graph;

    graph.AddLayer<InputLayer>(0, "input");

    ViewsDescriptor splitterDescriptor(2);
    IConnectableLayer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");

    Convolution2dDescriptor convDescriptor;
    IConnectableLayer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    IConnectableLayer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    OriginsDescriptor concatDescriptor(2);
    IConnectableLayer* const concatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");

    graph.AddLayer<OutputLayer>(0, "output");

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom({splitterLayer,
                                                                             convLayer1,
                                                                             convLayer2,
                                                                             concatLayer},
                                                                            {},
                                                                            {});

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(0, 0);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Save sub-graph layers for later verification
    const SubgraphView::IConnectableLayers subgraphLayers = subgraph->GetIConnectableLayers();

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that the layers belonging to the sub-graph have been erased from the graph after substitution
    CHECK(!AreAnySubgraphLayersPresentInGraph(subgraphLayers, graph));
}

}

TEST_SUITE("SubgraphSelection")
{
TEST_CASE("SubgraphForEmptyGraph")
{
    Graph graph;
    SubgraphView subgraph(graph);

    CHECK(subgraph.GetIInputSlots().empty());
    CHECK(subgraph.GetIOutputSlots().empty());
    CHECK(subgraph.GetIConnectableLayers().empty());
}

TEST_CASE("SubgraphForEntireGraph")
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    auto mid0 = graph.InsertNewLayer<ActivationLayer>(output->GetInputSlot(0),
                                                      ActivationDescriptor{},
                                                      "mid0");
    auto mid1 = graph.InsertNewLayer<ActivationLayer>(mid0->GetInputSlot(0),
                                                      ActivationDescriptor{},
                                                      "mid1");
    graph.InsertNewLayer<InputLayer>(mid1->GetInputSlot(0), 0, "input");

    SubgraphView subgraph(graph);

    CHECK(subgraph.GetIInputSlots().empty());
    CHECK(subgraph.GetIOutputSlots().empty());
    CHECK(subgraph.GetIConnectableLayers().size() == graph.GetNumLayers());
}

TEST_CASE("NoSubgraphsForNoMatch")
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    graph.InsertNewLayer<InputLayer>(output->GetInputSlot(0), 0, "input");

    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(graph, [](const Layer &) { return false; });

    CHECK(subgraphs.empty());
}

TEST_CASE("OneSubgraphsSelectedASingleMatch")
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    graph.InsertNewLayer<InputLayer>(output->GetInputSlot(0), 0, "input");

    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the output layer only
            [](const Layer & l)
            {
                bool isOutput = l.GetNameStr().compare("output") == 0;
                return isOutput;
            });

    CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({output}),
                                               // outputs of 'output' will be empty
                                               CreateOutputsFrom({output}),
                                               {output});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("MultipleLayersSelectedInTheMiddle")
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    auto mid0 = graph.InsertNewLayer<ActivationLayer>(output->GetInputSlot(0),
                                                      ActivationDescriptor{},
                                                      "mid0");
    auto mid1 = graph.InsertNewLayer<ActivationLayer>(mid0->GetInputSlot(0),
                                                      ActivationDescriptor{},
                                                      "mid1");
    graph.InsertNewLayer<InputLayer>(mid1->GetInputSlot(0), 0, "input");

    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation);
                return toSelect;
            });

    CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({mid1}),
                                               CreateOutputsFrom({mid0}),
                                               {mid1, mid0});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("DisjointGraphs")
{
    // The input graph has two disjoint sections and all layers are selected.
    // This should result in two subgraphs being produced.
    Graph graph;

    // the graph is constructed in reverse order
    auto o0 = graph.AddLayer<OutputLayer>(0, "output0");
    auto n0 = graph.InsertNewLayer<ActivationLayer>(o0->GetInputSlot(0), ActivationDescriptor{}, "intermediate0");
    auto i0 = graph.InsertNewLayer<InputLayer>(n0->GetInputSlot(0), 0, "input0");

    auto o1 = graph.AddLayer<OutputLayer>(1, "output1");
    auto n1 = graph.InsertNewLayer<ActivationLayer>(o1->GetInputSlot(0), ActivationDescriptor{}, "intermediate1");
    auto i1 = graph.InsertNewLayer<InputLayer>(n1->GetInputSlot(0), 1, "input1");

    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(graph,
                                              // select the middle layers only
                                              [](const Layer&) {
                                                  return true;
                                              });

    // expected results to test against
    auto expected1 = CreateSubgraphViewFrom({}, {}, { o0, n0, i0 });
    auto expected2 = CreateSubgraphViewFrom({}, {}, { o1, n1, i1 });
    CHECK(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));
        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            if (std::find(subgraphs[0]->GetIConnectableLayers().begin(),
                          subgraphs[0]->GetIConnectableLayers().end(), i0) !=
                          subgraphs[0]->GetIConnectableLayers().end())
            {
                CompareSubgraphViews(subgraphs[0], expected1);
                CompareSubgraphViews(subgraphs[1], expected2);
            }
            else
            {
                CompareSubgraphViews(subgraphs[0], expected2);
                CompareSubgraphViews(subgraphs[1], expected1);
            }
        }
    }
}

TEST_CASE("IslandInTheMiddle")
{
    // This case represent the scenario when a non-selected X1 node placed in the middle
    // of the selected M* nodes.
    // This checks that we don't merge M6 and M3 and create a dependency loop.
    /*
          M0
          / \
        M1   M4
         |   |
        M2   X1 < the island in the middle !
         |   |
        M3   M5
          \ /
          M6
    */
    Graph graph;

    OriginsDescriptor concatDescriptor(2);
    auto m6 = graph.AddLayer<ConcatLayer>(concatDescriptor, "m6");
    auto m3 = graph.InsertNewLayer<ActivationLayer>(m6->GetInputSlot(0),
        ActivationDescriptor{},
        "m3");
    auto m2 = graph.InsertNewLayer<ActivationLayer>(m3->GetInputSlot(0),
        ActivationDescriptor{},
        "m2");
    auto m1 = graph.InsertNewLayer<ActivationLayer>(m2->GetInputSlot(0),
        ActivationDescriptor{},
        "m1");
    auto m0 = graph.InsertNewLayer<InputLayer>(m1->GetInputSlot(0), 0, "m0");

    auto m5 = graph.InsertNewLayer<ActivationLayer>(m6->GetInputSlot(1),
        ActivationDescriptor{},
        "m5");
    auto x1 = graph.InsertNewLayer<ActivationLayer>(m5->GetInputSlot(0),
        ActivationDescriptor{},
        "x1");
    auto m4 = graph.InsertNewLayer<ActivationLayer>(x1->GetInputSlot(0),
        ActivationDescriptor{},
        "m4");

    // Connect the other branch to the input layer
    m0->GetOutputSlot(0).Connect(m4->GetInputSlot(0));

    // All selected 'M*' layers will be of Activation type
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the middle layers only
            [](const Layer& l)
    {
        bool toSelect = std::string(l.GetName())[0] == 'm';
        return toSelect;
    });

    // expected results to test against
    auto largerSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({ m0 }),
        CreateOutputsFrom({ m3, m4 }),
        { m0, m1, m2, m3, m4 });

    auto smallerSubgraph =
        CreateSubgraphViewFrom(std::vector<InputSlot*>{ &m5->GetInputSlot(0), & m6->GetInputSlot(0) },
            std::vector<OutputSlot*>{},
            { m5, m6 });

    CHECK(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr& lhs, SubgraphViewSelector::SubgraphViewPtr& rhs)
            {
                return (lhs->GetIConnectableLayers().size() < rhs->GetIConnectableLayers().size());
            }
            );

            CHECK(subgraphs[0]->GetIConnectableLayers().size() == 2);
            CHECK(subgraphs[1]->GetIConnectableLayers().size() == 5);

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], largerSubgraph);
        }
    }
}

TEST_CASE("MultipleSimpleSubgraphs")
{
    // This test case represents the scenario when we have two distinct subgraphs
    // in a simple linear network. The selected nodes are the M* and the
    // non-selected ones are the X*
    //
    // X1 -> M1 -> M2 -> X2 -> M3 -> X3
    //
    // The expected results is two subgraphs, one with {M1, M2} and another one
    // with {M3}
    //
    Graph graph;

    // the graph is constructed in reverse order
    auto x3 = graph.AddLayer<OutputLayer>(0, "output");
    auto m3 = graph.InsertNewLayer<ActivationLayer>(x3->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m3");
    auto x2 = graph.InsertNewLayer<Convolution2dLayer>(m3->GetInputSlot(0),
                                                       Convolution2dDescriptor{},
                                                       "x2");
    auto m2 = graph.InsertNewLayer<ActivationLayer>(x2->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m2");
    auto m1 = graph.InsertNewLayer<ActivationLayer>(m2->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m1");
    graph.InsertNewLayer<InputLayer>(m1->GetInputSlot(0), 0, "x1");

    // All selected 'M*' layers will be of Activation type
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation);
                return toSelect;
            });

    // expected results to test against
    auto largerSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({m1}),
                                                 CreateOutputsFrom({m2}),
                                                 {m1, m2});

    auto smallerSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({m3}),
                                                  CreateOutputsFrom({m3}),
                                                  {m3});

    CHECK(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                {
                    return (lhs->GetIConnectableLayers().size() < rhs->GetIConnectableLayers().size());
                }
            );

            CHECK(subgraphs[0]->GetIConnectableLayers().size() == 1);
            CHECK(subgraphs[1]->GetIConnectableLayers().size() == 2);

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], largerSubgraph);
        }
    }
}

TEST_CASE("SimpleLinearTest")
{
    //X1 -> M1 -> M2 -> X2
    //Where the input slots of M1 and the output slots of M2 are to be the sub graph boundaries.
    Graph graph;

    ActivationDescriptor activationDefaults;

    auto layerX1 = graph.AddLayer<InputLayer>(0, "layerX1");
    auto layerX2 = graph.AddLayer<OutputLayer>(0, "layerX2");
    auto layerM1 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM1");
    auto layerM2 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM2");

    //      X1
    //      |
    //      M1
    //      |
    //      M2
    //      |
    //      X2

    layerX1->GetOutputSlot(0).Connect(layerM1->GetInputSlot(0));
    layerM1->GetOutputSlot(0).Connect(layerM2->GetInputSlot(0));
    layerM2->GetOutputSlot(0).Connect(layerX2->GetInputSlot(0));

    SubgraphViewSelector::Subgraphs subgraphs =
            SubgraphViewSelector::SelectSubgraphs(
                    graph,
                    // select the activation layers M1 and M2
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation);
                        return toSelect;
                    });

    CHECK(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1}),
                                               CreateOutputsFrom({layerM2}),
                                               {layerM1, layerM2});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("MultiInputSingleOutput")
{
    //X1 -> M1 -> M3 -> X3
    //X2 -> M2 -> M3 -> X3
    //Where the input slots of {M1, M2} and the output slots of M3 are to be the subgraph boundaries.
    Graph graph;

    ActivationDescriptor activationDefaults;

    auto layerX1 = graph.AddLayer<InputLayer>(0, "layerX1");
    auto layerX2 = graph.AddLayer<InputLayer>(1, "layerX2");
    auto layerM1 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM1");
    auto layerM2 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM2");
    auto layerM3 = graph.AddLayer<AdditionLayer>("layerM3");
    auto layerX3 = graph.AddLayer<OutputLayer>(0, "layerX3");

    //  X1  X2
    //  |   |
    //  M1  M2
    //   \  |
    //    \ |
    //     \|
    //      M3
    //      |
    //      |
    //      X3

    layerX1->GetOutputSlot(0).Connect(layerM1->GetInputSlot(0));
    layerX2->GetOutputSlot(0).Connect(layerM2->GetInputSlot(0));
    layerM1->GetOutputSlot(0).Connect(layerM3->GetInputSlot(0));
    layerM2->GetOutputSlot(0).Connect(layerM3->GetInputSlot(1));
    layerM3->GetOutputSlot(0).Connect(layerX3->GetInputSlot(0));

    SubgraphViewSelector::Subgraphs subgraphs =
            SubgraphViewSelector::SelectSubgraphs(
                    graph,
                    // select Activation and Addition Layers M1, M2 and M3
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation
                                         || l.GetType() == LayerType::Addition);
                        return toSelect;
                    });

    CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1, layerM2}),
                                               CreateOutputsFrom({layerM3}),
                                               {layerM1, layerM2, layerM3});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("SingleInputMultiOutput")
{
    //X1 -> M1 -> M2 -> X2
    //X1 -> M1 -> M3 -> X3
    //Where the input slots of M1 and the output slots of {M2, M3} are to be the subgraph boundaries.
    Graph graph;

    ActivationDescriptor activationDefaults;
    ViewsDescriptor viewDefaults(2,4);

    Layer* layerX1 = graph.AddLayer<InputLayer>(0, "layerX1");
    Layer* layerM1 = graph.AddLayer<SplitterLayer>(viewDefaults, "layerM1");
    Layer* layerM2 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM2");
    Layer* layerM3 = graph.AddLayer<ActivationLayer>(activationDefaults, "layerM3");
    Layer* layerX2 = graph.AddLayer<OutputLayer>(0, "layerX2");
    Layer* layerX3 = graph.AddLayer<OutputLayer>(1, "layerX3");

    //      X1
    //      |
    //      M1
    //     /|
    //    / |
    //   /  |
    //  M2  M3
    //  |   |
    //  |   |
    //  X2  X3

    layerX1->GetOutputSlot(0).Connect(layerM1->GetInputSlot(0));
    layerM1->GetOutputSlot(0).Connect(layerM2->GetInputSlot(0));
    layerM1->GetOutputSlot(1).Connect(layerM3->GetInputSlot(0));
    layerM2->GetOutputSlot(0).Connect(layerX2->GetInputSlot(0));
    layerM3->GetOutputSlot(0).Connect(layerX3->GetInputSlot(0));

    SubgraphViewSelector::Subgraphs subgraphs =
            SubgraphViewSelector::SelectSubgraphs(
                    graph,
                    // select Activation and Splitter Layers M1, M2 and M3
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation
                                         || l.GetType() == LayerType::Splitter);
                        return toSelect;
                    });

    CHECK(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1}),
                                               CreateOutputsFrom({layerM2, layerM3}),
                                               {layerM1, layerM2, layerM3});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("MultiInputMultiOutput")
{
    // This case represents the scenario with multiple inputs and multiple outputs
    //
    // X1 -> M1 -> M3 -> M4 -> X3
    // X2 -> M2 -> M3 -> M5 -> X4
    //
    // Where the input slots of {M1, M2} and the output slots of {M4, M5} are to be the subgraph
    // boundaries.

    Graph graph;

    ActivationDescriptor activationDefaults;
    OriginsDescriptor concatDescriptor(2);

    auto x1 = graph.AddLayer<InputLayer>(0, "x1");
    auto x2 = graph.AddLayer<InputLayer>(1, "x2");

    auto m1 = graph.AddLayer<ActivationLayer>(activationDefaults, "m1");
    auto m2 = graph.AddLayer<ActivationLayer>(activationDefaults, "m2");
    auto m3 = graph.AddLayer<ConcatLayer>(concatDescriptor, "m3");

    auto m4 = graph.AddLayer<ActivationLayer>(activationDefaults, "m4");
    auto m5 = graph.AddLayer<ActivationLayer>(activationDefaults, "m5");

    auto x3 = graph.AddLayer<OutputLayer>(0, "x3");
    auto x4 = graph.AddLayer<OutputLayer>(1, "x4");

    x1->GetOutputSlot(0).Connect(m1->GetInputSlot(0));
    x2->GetOutputSlot(0).Connect(m2->GetInputSlot(0));

    m1->GetOutputSlot(0).Connect(m3->GetInputSlot(0));
    m2->GetOutputSlot(0).Connect(m3->GetInputSlot(1));

    m3->GetOutputSlot(0).Connect(m4->GetInputSlot(0));
    m3->GetOutputSlot(0).Connect(m5->GetInputSlot(0));

    m4->GetOutputSlot(0).Connect(x3->GetInputSlot(0));
    m5->GetOutputSlot(0).Connect(x4->GetInputSlot(0));


    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select Activation and Concat Layers M1, M2, M3, M4, M5
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation
                                 || l.GetType() == LayerType::Concat);
                return toSelect;
            });


    CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({m1, m2}),
                                               CreateOutputsFrom({m4, m5}),
                                               {m1, m2, m3, m4, m5});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

TEST_CASE("ValidMerge")
{
    // Checks that a node that has multiple choices for merge candidates (M3 in this case) correctly merges with the
    // one that it can (M0), and doesn't merge with the ones it can't (X2 and M2).
    //
    //      X1
    //      |
    //      M1
    //     / \'
    //    X2  M2   M0
    //     \  |  /
    //        M3
    //
    Graph graph;

    ActivationDescriptor activationDefaults;
    OriginsDescriptor concatDescriptor(3);

    auto x1 = graph.AddLayer<InputLayer>(0, "x1");
    auto x2 = graph.AddLayer<ActivationLayer>(activationDefaults, "x2");
    auto m0 = graph.AddLayer<InputLayer>(1, "m0");
    auto m1 = graph.AddLayer<ActivationLayer>(activationDefaults, "m1");
    auto m2 = graph.AddLayer<ActivationLayer>(activationDefaults, "m2");
    auto m3 = graph.AddLayer<ConcatLayer>(concatDescriptor, "m3");

    x1->GetOutputSlot(0).Connect(m1->GetInputSlot(0));
    m1->GetOutputSlot(0).Connect(x2->GetInputSlot(0));
    m1->GetOutputSlot(0).Connect(m2->GetInputSlot(0));
    x2->GetOutputSlot(0).Connect(m3->GetInputSlot(0));
    m2->GetOutputSlot(0).Connect(m3->GetInputSlot(1));
    m0->GetOutputSlot(0).Connect(m3->GetInputSlot(2));

    SubgraphViewSelector::Subgraphs subgraphs = SubgraphViewSelector::SelectSubgraphs(
        graph,
        [](const Layer& l) {
            return std::string(l.GetName())[0] == 'm';
        });

    // expected results to test against
    auto expectedSubgraph0 =
        CreateSubgraphViewFrom(
            CreateInputsFrom({ m1 }),
            std::vector<OutputSlot*>{ &m1->GetOutputSlot(0), &m2->GetOutputSlot(0) },
            { m1, m2 });

    auto expectedSubgraph1 = CreateSubgraphViewFrom(
        std::vector<InputSlot*>{ &m3->GetInputSlot(0), & m3->GetInputSlot(1) },
        CreateOutputsFrom({ }),
        { m0, m3 });

    CHECK(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            if (subgraphs[0]->GetIInputSlots().size() == 1)
            {
                CompareSubgraphViews(subgraphs[0], expectedSubgraph0);
                CompareSubgraphViews(subgraphs[1], expectedSubgraph1);
            }
            else
            {
                CompareSubgraphViews(subgraphs[0], expectedSubgraph1);
                CompareSubgraphViews(subgraphs[1], expectedSubgraph0);
            }
        }
    }
}

TEST_CASE("PropagatedDependencies")
{
    // Version of IslandInTheMiddle with longer chain
    // to make sure antecedents are propagated.
    /*
          M0
          / \
        M1   M4
         |   |
        M2   X1 < the island in the middle !
         |   |
         |   M10
         |   |
         |   X2 < another island in the middle !
         |   |
         M3  M5
          \ /
          M6
    */
    Graph graph;

    OriginsDescriptor concatDescriptor(2);
    auto m6 = graph.AddLayer<ConcatLayer>(concatDescriptor, "m6");
    auto m3 = graph.InsertNewLayer<ActivationLayer>(m6->GetInputSlot(0),
        ActivationDescriptor{},
        "m3");
    auto m2 = graph.InsertNewLayer<ActivationLayer>(m3->GetInputSlot(0),
        ActivationDescriptor{},
        "m2");
    auto m1 = graph.InsertNewLayer<ActivationLayer>(m2->GetInputSlot(0),
        ActivationDescriptor{},
        "m1");
    auto m0 = graph.InsertNewLayer<InputLayer>(m1->GetInputSlot(0), 0, "m0");

    auto m5 = graph.InsertNewLayer<ActivationLayer>(m6->GetInputSlot(1),
        ActivationDescriptor{},
        "m5");
    auto x2 = graph.InsertNewLayer<ActivationLayer>(m5->GetInputSlot(0), ActivationDescriptor{}, "x2");
    auto m10 = graph.InsertNewLayer<ActivationLayer>(x2->GetInputSlot(0), ActivationDescriptor{}, "m10");
    auto x1 = graph.InsertNewLayer<ActivationLayer>(m10->GetInputSlot(0),
        ActivationDescriptor{},
        "x1");
    auto m4 = graph.InsertNewLayer<ActivationLayer>(x1->GetInputSlot(0),
        ActivationDescriptor{},
        "m4");

    // Connect the other branch to the input layer
    m0->GetOutputSlot(0).Connect(m4->GetInputSlot(0));

    // All selected 'M*' layers will be of Activation type
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the middle layers only
            [](const Layer& l)
    {
        bool toSelect = std::string(l.GetName())[0] == 'm';
        return toSelect;
    });

    // expected results to test against
    auto largerSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({ m0 }),
        CreateOutputsFrom({ m3, m4 }),
        { m0, m1, m2, m3, m4 });

    auto mediumSubgraph = CreateSubgraphViewFrom(std::vector<InputSlot*>{ &m5->GetInputSlot(0), &m6->GetInputSlot(0) },
                                                 std::vector<OutputSlot*>{}, { m5, m6 });

    auto smallerSubgraph =
        CreateSubgraphViewFrom(CreateInputsFrom({ m10 }), CreateOutputsFrom({ m10 }), { m10 });

    CHECK(subgraphs.size() == 3);
    if (subgraphs.size() == 3)
    {
        // we need to have valid subgraph pointers here
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));
        CHECK((subgraphs[2] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr && subgraphs[2].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr& lhs, SubgraphViewSelector::SubgraphViewPtr& rhs)
            {
                return (lhs->GetIConnectableLayers().size() < rhs->GetIConnectableLayers().size());
            }
            );

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], mediumSubgraph);
            CompareSubgraphViews(subgraphs[2], largerSubgraph);
        }
    }
}

TEST_CASE("Random")
{
    // Creates random networks, splits them into subgraphs and checks the resulting subgraphs obey the required
    // dependency rules. We can easily generate very large networks which helps cover corner cases the other
    // small, manually crafted tests have missed. We can also use this to measure performance on large networks.
    constexpr bool debug = false; // Enable this to dump dot files and performance timings.

    std::mt19937 randomGenerator;

    // Helper function to get a random number in [0, maxExclusive)
    auto GetRandom = [&randomGenerator](auto maxExclusive) {
        // Note we could use uniform_int_distribution here, but that gives inconsistent results across platforms
        // which makes it harder to reproduce results.
        // It appears that uniform_real_distribution is consistent across MSVC and gcc so we use that and round it.
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        return static_cast<decltype(maxExclusive)>(uniform(randomGenerator) * static_cast<float>(maxExclusive));
    };
    // Helper function to get a bool that has probability 'trueProb' of being true.
    auto GetRandomFlag = [&randomGenerator](float trueProb) {
        std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
        return uniform(randomGenerator) < trueProb;
    };

    constexpr uint32_t numTests = 100;
    for (uint32_t testIdx = 0; testIdx < numTests; ++testIdx)
    {
        randomGenerator.seed(testIdx); // Set a deterministic seed for reproducibility.

        // Create random graph
        Graph graph;
        {
            // First add the layers, without any connections. The following random constants determine the number of
            // each layer to add, along with the chance that each layer will be 'supported' (i.e. selected for
            // inclusion in the resulting subgraphs).
            uint32_t numInputs = 1 + GetRandom(4u);
            uint32_t numConstants = 1 + GetRandom(4u);
            uint32_t numOutputs = 1 + GetRandom(4u);
            uint32_t numConcats = 0 + GetRandom(500u);
            uint32_t numSplits = 0 + GetRandom(500u);
            float supportedProb = 0.7f;

            for (uint32_t i = 0; i < numInputs; ++i)
            {
                std::string name = "input" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                graph.AddLayer<InputLayer>(static_cast<LayerBindingId>(i), name.c_str());
            }
            for (uint32_t i = 0; i < numConstants; ++i)
            {
                std::string name = "constant" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                graph.AddLayer<ConstantLayer>(name.c_str());
            }
            for (uint32_t i = 0; i < numOutputs; ++i)
            {
                std::string name = "output" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                graph.AddLayer<OutputLayer>(static_cast<LayerBindingId>(i), name.c_str());
            }
            for (uint32_t i = 0; i < numConcats; ++i)
            {
                std::string name = "concat" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                numInputs = 1 + GetRandom(3u);
                OriginsDescriptor concatDesc(numInputs);
                graph.AddLayer<ConcatLayer>(concatDesc, name.c_str());
            }
            for (uint32_t i = 0; i < numSplits; ++i)
            {
                std::string name = "split" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                numOutputs = 1 + GetRandom(3u);
                ViewsDescriptor splitDesc(numOutputs);
                graph.AddLayer<SplitterLayer>(splitDesc, name.c_str());
            }

            // Associate each layer with a "depth" parameter. This is used when creating connections to ensure
            // that we don't have any loops, by only connecting to layers with a lower "depth".
            // This can be thought of as distance from the "top" of the graph (assuming the graph flows top-to-bottom).
            // Unfortunately this approach ends up producing very "wide" graphs,
            // which probably isn't very representative of 'real' networks.
            uint32_t maxLayerDepth = 5 + GetRandom(2000u);
            std::map<Layer*, uint32_t> layerDepths;
            std::map<uint32_t, std::vector<Layer*>> layersAtDepth;
            for (Layer* layer : graph)
            {
                uint32_t depth;
                if (layer->GetType() == LayerType::Input || layer->GetType() == LayerType::Constant)
                {
                    // There needs to be at least one input-like layer above everything else, otherwise would be
                    // nothing for them to connect to!
                    depth = 0;
                }
                else
                {
                    // Other layers are randomly assigned to later depths.
                    depth = 1 + GetRandom(maxLayerDepth);
                }
                layerDepths[layer] = depth;
                layersAtDepth[depth].push_back(layer);
            }

            // Connect layers to each other. Every input slot of every layer must be connected, but it doesn't
            // matter if an output slot goes unused.
            for (Layer* layer : graph)
            {
                for (uint32_t inputSlotIdx = 0; inputSlotIdx < layer->GetNumInputSlots(); ++inputSlotIdx)
                {
                    InputSlot& inputSlot = layer->GetInputSlot(inputSlotIdx);
                    uint32_t maxLayerDepthToConnectTo = layerDepths[layer];
                    // This prevents a connection causing a loop
                    // Finding a layer to connect to may take multiple attempts, so keep trying until it works.
                    while (inputSlot.GetConnectedOutputSlot() == nullptr)
                    {
                        uint32_t layerDepth = GetRandom(maxLayerDepthToConnectTo);
                        const std::vector<Layer*>& layersToChooseFrom = layersAtDepth[layerDepth];
                        if (layersToChooseFrom.size() == 0)
                        {
                            continue;
                        }
                        Layer* layerToConnectWith = layersToChooseFrom[GetRandom(layersToChooseFrom.size())];
                        if (layerToConnectWith->GetNumOutputSlots() == 0)
                        {
                            continue;
                        }
                        uint32_t outputSlotIdx = GetRandom(layerToConnectWith->GetNumOutputSlots());
                        layerToConnectWith->GetOutputSlot(outputSlotIdx).Connect(inputSlot);
                    }
                }
            }
        }

        if (debug)
        {
            std::ofstream f("INPUT_" + std::to_string(testIdx) + ".dot");
            graph.SerializeToDot(f);
        }

        // Run the splitting algorithm, selecting all nodes ending in an 'S' (as randomly assigned above).
        auto startTime = std::chrono::high_resolution_clock::now();

        SubgraphViewSelector::Subgraphs subgraphs =
            SubgraphViewSelector::SelectSubgraphs(graph,
                [](const Layer& l) { return std::string(l.GetName()).back() == 'S'; });

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
        if (debug)
        {
            std::cout << "Test " << testIdx << ": " << duration.count() << " microseconds" << std::endl;
        }

        // Build a map of which subgraph is assigned to each layer.
        // This helps some of the following code.
        std::map<Layer*, SubgraphView*> layerToSubgraph;
        for (Layer* layer : graph)
        {
            size_t i = 0;
            for (std::unique_ptr<SubgraphView>& subgraph : subgraphs)
            {
                std::string name = std::to_string(i++);
                if (std::find(subgraph->cbeginIConnectable(), subgraph->cendIConnectable(), layer)
                    != subgraph->cendIConnectable())
                {
                    layerToSubgraph[layer] = subgraph.get();
                    break;
                }
            }
        }

        if (debug)
        {
            // Before dumping the dot file, set each Layer's BackendId property so that the dot file
            // shows the resulting subgraph assignments.
            for (Layer* layer : graph)
            {
                std::string name = "NotAssigned";
                auto subgraphIt = layerToSubgraph.find(layer);
                if (subgraphIt != layerToSubgraph.end())
                {
                    auto subgraphIdx = std::distance(subgraphs.begin(),
                            std::find_if(subgraphs.begin(), subgraphs.end(),
                                [&](auto& s) { return s.get() == subgraphIt->second; }));
                    name = std::to_string(subgraphIdx);
                }
                layer->SetBackendId(armnn::BackendId(name));
            }

            std::ofstream f("GRAPH_" + std::to_string(testIdx) + ".dot");
            graph.SerializeToDot(f);
        }

        // Check the dependencies between subgraphs to make sure that the algorithm has produced a valid result.
        // Starting from each of the input slots of each subgraph, recurse up the graph and ensure that we never
        // encounter a layer that belongs to the subgraph that we started from.
        for (std::unique_ptr<SubgraphView>& subgraph : subgraphs)
        {
            for (IInputSlot* inSlot : subgraph->GetIInputSlots())
            {
                std::queue<Layer*> toProcess;
                toProcess.push(&PolymorphicDowncast<InputSlot*>(inSlot)->GetConnectedOutputSlot()->GetOwningLayer());
                while (toProcess.size() > 0)
                {
                    Layer* l = toProcess.front();
                    toProcess.pop();

                    CHECK(layerToSubgraph[l] != subgraph.get());

                    for (const InputSlot& is : l->GetInputSlots())
                    {
                        toProcess.push(&is.GetConnectedOutputSlot()->GetOwningLayer());
                    }
                }
            }
        }
    }
}

}

TEST_SUITE("IntegrationTests")
{
TEST_CASE("SingleSubgraph")
{
    // This test case represents the scenario when we have one subgraph
    // in which two layers have GpuAcc backend assigned

    //Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    convLayer1->SetBackendId(Compute::GpuAcc);

    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");
    convLayer2->SetBackendId(Compute::GpuAcc);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(convLayer2->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // GpuAcc sub graph selector
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the GpuAcc layers only
            [](const Layer & l){
                bool toSelect = (l.GetBackendId() == Compute::GpuAcc);
                return toSelect;
            });

    CHECK(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        CHECK((subgraphs[0] != nullptr));

        if (subgraphs[0].get() != nullptr)
        {
            unsigned int numInputSlots = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetIInputSlots().size());
            unsigned int numOutputSlots = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetIOutputSlots().size());

            CHECK((numInputSlots == 1));
            CHECK((numOutputSlots == 1));

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraphInputConn1 = subgraphs[0]->GetIInputSlot(0)->GetConnection();
            IInputSlot* subgraphOutputConn1 = subgraphs[0]->GetIOutputSlot(0)->GetConnection(0);

            // Construct dummy pre-compiled layer
            PreCompiledDescriptor preCompiledDescriptor(numInputSlots, numOutputSlots);
            Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

            // Substitute sub-graph with pre-compiled layer
            graph.SubstituteSubgraph(*subgraphs[0], preCompiledLayer);

            // Check that connections are correct after substitution
            CHECK_EQ(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);

            CHECK_EQ(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
        }
    }
}

TEST_CASE("MultipleSubgraphs")
{
    // This test case represents the scenario when we have two subgraphs
    // in which two layers have CpuAcc backend assigned

    //Construct graph
    Graph graph;

    Layer* const inputLayer = graph.AddLayer<InputLayer>(0, "input");

    ViewsDescriptor splitterDescriptor(2);
    Layer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");
    splitterLayer->SetBackendId(Compute::CpuAcc);

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    OriginsDescriptor concatDescriptor(2);
    Layer* const pConcatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");
    pConcatLayer->SetBackendId(Compute::CpuAcc);

    Layer* const outputLayer = graph.AddLayer<OutputLayer>(0, "output");

    inputLayer->GetOutputSlot(0).Connect(splitterLayer->GetInputSlot(0));
    splitterLayer->GetOutputSlot(0).Connect(convLayer1->GetInputSlot(0));
    splitterLayer->GetOutputSlot(1).Connect(convLayer2->GetInputSlot(0));
    convLayer1->GetOutputSlot(0).Connect(pConcatLayer->GetInputSlot(0));
    convLayer2->GetOutputSlot(0).Connect(pConcatLayer->GetInputSlot(1));
    pConcatLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    // CpuAcc sub graph selector
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the CpuAcc layers only
            [](const Layer & l){
                bool toSelect = (l.GetBackendId() == Compute::CpuAcc);
                return toSelect;
            });

    CHECK(subgraphs.size() == 2);
    if(subgraphs.size() == 2)
    {
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            //Sort subgraphs by their inputSlot size.
            std::sort(subgraphs.begin(), subgraphs.end(),
                      [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                      {
                          return (lhs->GetIInputSlots().size() < rhs->GetIInputSlots().size());
                      }
            );

            unsigned int numInputSlots1  = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetIInputSlots().size());
            unsigned int numOutputSlots1 = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetIOutputSlots().size());

            unsigned int numInputSlots2  = armnn::numeric_cast<unsigned int>(subgraphs[1]->GetIInputSlots().size());
            unsigned int numOutputSlots2 = armnn::numeric_cast<unsigned int>(subgraphs[1]->GetIOutputSlots().size());

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraph1InputConn  = subgraphs[0]->GetIInputSlot(0)->GetConnection();
            IInputSlot* subgraph1OutputConn1 = subgraphs[0]->GetIOutputSlot(0)->GetConnection(0);
            IInputSlot* subgraph1OutputConn2 = subgraphs[0]->GetIOutputSlot(1)->GetConnection(0);

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraph2InputConn1 = subgraphs[1]->GetIInputSlot(0)->GetConnection();
            IOutputSlot* subgraph2InputConn2 = subgraphs[1]->GetIInputSlot(1)->GetConnection();
            IInputSlot* subgraph2OutputConn  = subgraphs[1]->GetIOutputSlot(0)->GetConnection(0);

            PreCompiledDescriptor preCompiledDescriptor1(numInputSlots1, numOutputSlots1);
            Layer* const preCompiledLayer1 = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor1, "pre-compiled1");

            PreCompiledDescriptor preCompiledDescriptor2(numInputSlots2, numOutputSlots2);
            Layer* const preCompiledLayer2 = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor2, "pre-compiled2");

            // Substitute sub-graph with pre-compiled layer
            graph.SubstituteSubgraph(*subgraphs[0], preCompiledLayer1);
            graph.SubstituteSubgraph(*subgraphs[1], preCompiledLayer2);

            // Check that connections are correct after substitution
            CHECK_EQ(preCompiledLayer1->GetInputSlot(0).GetConnection(), subgraph1InputConn);
            CHECK_EQ(preCompiledLayer1->GetOutputSlot(0).GetConnection(0), subgraph1OutputConn1);
            CHECK_EQ(preCompiledLayer1->GetOutputSlot(1).GetConnection(0), subgraph1OutputConn2);

            CHECK_EQ(preCompiledLayer2->GetInputSlot(0).GetConnection(), subgraph2InputConn1);
            CHECK_EQ(preCompiledLayer2->GetInputSlot(1).GetConnection(), subgraph2InputConn2);
            CHECK_EQ(preCompiledLayer2->GetOutputSlot(0).GetConnection(0), subgraph2OutputConn);
        }
    }
}

TEST_CASE("SubgraphCycles")
{
    // This case represent the scenario when a naive split could lead to a cyclic dependency between two subgraphs
    //
    // X0 -> M0 -> X1 -> M2 -> X2
    // X0 -> M0 -> M1 -> M2 -> X2
    //
    /*
          X0
          |
          |
          M0
         / |
        /  |
       X1  M1
        \ /
        M2
        |
        X2
    */
    // The expected result for this is that M0,M1 will be part of one subgraph and M2 in another and the
    // input and output slots in the subgraphs will be set accordingly.
    //
    Graph graph;

    OriginsDescriptor originsDescriptor(2);
    auto x0 = graph.AddLayer<InputLayer>(0, "x0");
    auto m0 = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "m0");
    auto x1 = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "x1");
    auto m1 = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "m1");
    auto m2 = graph.AddLayer<AdditionLayer>("m2");
    auto x2 = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "x2");

    x0->GetOutputSlot(0).Connect(m0->GetInputSlot(0));
    m0->GetOutputSlot(0).Connect(x1->GetInputSlot(0));
    m0->GetOutputSlot(0).Connect(m1->GetInputSlot(0));
    x1->GetOutputSlot(0).Connect(m2->GetInputSlot(0));
    m1->GetOutputSlot(0).Connect(m2->GetInputSlot(1));
    m2->GetOutputSlot(0).Connect(x2->GetInputSlot(0));

    // All selected 'M*' layers will be have 'm' in the name
    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
                {
                    bool toSelect = (l.GetNameStr().find('m') != std::string::npos);
                    return toSelect;
                });

    // expected results to test against
    auto inputSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({m0}),
                                                CreateOutputsFrom({m0, m1}),
                                                {m0, m1});

    auto outputSubgraph = CreateSubgraphViewFrom(CreateInputsFrom({m2}),
                                                 CreateOutputsFrom({m2}),
                                                 {m2});

    CHECK(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        CHECK((subgraphs[0] != nullptr));
        CHECK((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                      [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                          {
                              return (lhs->GetIConnectableLayers().size() < rhs->GetIConnectableLayers().size());
                          }
            );

            // one subgraph needs to be size=1 and the other one is 4
            CHECK(subgraphs[0]->GetIConnectableLayers().size() == 1);
            CHECK(subgraphs[1]->GetIConnectableLayers().size() == 2);

            CompareSubgraphViews(subgraphs[0], outputSubgraph);
            CompareSubgraphViews(subgraphs[1], inputSubgraph);
        }
    }
}

TEST_CASE("SubgraphOrder")
{
    Graph graph;

    auto input      = graph.AddLayer<InputLayer>(0, "Input");
    auto activation = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "Activation");
    auto output     = graph.AddLayer<OutputLayer>(1, "Output");

    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Add in out of order
    auto view = CreateSubgraphViewFrom({},
                                       {},
                                       {output, input, activation});

    // Check the layers are sorted topologically in the view
    int idx=0;
    LayerType expectedSorted[] = {LayerType::Input, LayerType::Activation, LayerType::Output};
    view->ForEachLayer([&idx, &expectedSorted](const Layer* l)
        {
            CHECK((expectedSorted[idx] == l->GetType()));
            idx++;
        }
    );
}

TEST_CASE("SubgraphViewWorkingCopy")
{
    Graph graph;

    auto input      = graph.AddLayer<InputLayer>(0, "Input");
    auto activation = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "Activation");
    auto output     = graph.AddLayer<OutputLayer>(1, "Output");

    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Add in out of order
    auto view = CreateSubgraphViewFrom({output, input, activation},
                                       {},
                                       {});

    SubgraphView workingCopy = view->GetWorkingCopy();

    // Check the layers are sorted topologically in the view
    int idx=0;
    LayerType expectedSorted[] = {LayerType::Input, LayerType::Activation, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSorted](const IConnectableLayer* l)
        {
            CHECK((expectedSorted[idx] == l->GetType()));
            idx++;
        }
    );
}

bool ReplaceConstantMultiplicationWithDepthwise(SubgraphView& subgraph,
                                                IConnectableLayer* layer)
{
    if (layer->GetType() == LayerType::Multiplication)
    {
        IInputSlot* patternSubgraphInput = &layer->GetInputSlot(0);

        const IConnectableLayer* inputLayer    = &patternSubgraphInput->GetConnection()->GetOwningIConnectableLayer();
        const IConnectableLayer* constantLayer = &layer->GetInputSlot(1).GetConnection()->GetOwningIConnectableLayer();

        // Figure out which of the two inputs is the constant
        if (constantLayer->GetType() != LayerType::Constant)
        {
            patternSubgraphInput = &layer->GetInputSlot(1);
            std::swap(inputLayer, constantLayer);
        }

        if (constantLayer->GetType() == LayerType::Constant)
        {
            const TensorInfo& inputInfo = inputLayer->GetOutputSlot(0).GetTensorInfo();
            const TensorInfo& constInfo = constantLayer->GetOutputSlot(0).GetTensorInfo();

            // Add a Depthwise only where the constant input is a scalar that takes the form { 1, 1, 1, C }.
            // The scalar is used as weights for the convolution.
            if (constInfo.GetShape() == TensorShape({ 1, 1, 1, inputInfo.GetShape()[3] }))
            {
                auto replacementGraph = INetwork::Create();

                DepthwiseConvolution2dDescriptor desc;
                desc.m_DataLayout = DataLayout::NHWC;

                TensorInfo weightInfo        = constInfo;
                const TensorInfo& outputInfo = layer->GetOutputSlot(0).GetTensorInfo();
                unsigned int M               = outputInfo.GetShape()[3] / inputInfo.GetShape()[3];
                ARMNN_ASSERT_MSG(M == 1, "Constant multiplication only support 1x1x1xC, so M should always be 1 here");
                weightInfo.SetShape({ 1, 1, 1, constInfo.GetShape()[3] * M });    //1HW(I*M)

                const void* weightData =  PolymorphicPointerDowncast<const ConstantLayer>(constantLayer)
                        ->m_LayerOutput->GetConstTensor<void>();
                TensorInfo            weightsInfo = constInfo;
                ConstTensor           weights(weightsInfo, weightData);

                const auto depthwiseLayer = replacementGraph->AddDepthwiseConvolution2dLayer(
                        desc, weights, armnn::EmptyOptional(), "Replacement for Constant-Multiplication");

                auto& outslot = layer->GetOutputSlot(0);
                SubgraphView::IOutputSlots outputs{ &outslot };
                SubgraphView::IConnectableLayers layers;
                layers.push_back(layer);
                layers.push_back(const_cast<IConnectableLayer*>(constantLayer));

                SubgraphView patternSubgraph(std::move(layers), {patternSubgraphInput}, {&layer->GetOutputSlot(0)});

                subgraph.SubstituteSubgraph(patternSubgraph, depthwiseLayer );

                return true;
            }
        }
    }
    return false;
}

bool ReplaceTestMultiplication(SubgraphView& subgraph,
                           IConnectableLayer* layer)
{
    if (layer->GetType() == LayerType::Multiplication)
    {

        switch (layer->GetType())
        {
            case LayerType::Multiplication:
                return ReplaceConstantMultiplicationWithDepthwise(subgraph, layer);
                break;
            default:
                throw Exception("Found unknown MultiplicationSupportedMode value");
                break;
        }
    }
    return false;
}

void ReplaceUnsupportedLayers(SubgraphView& subgraph)
{
    using ReplacementFunc                    = bool (*)(SubgraphView&, IConnectableLayer*);
    const ReplacementFunc replacementFuncs[] = {
            &ReplaceTestMultiplication,
    };

    subgraph.ForEachLayer([replacementFuncs, &subgraph](IConnectableLayer* layer)
           {
               auto madeChange = false;
               for (const ReplacementFunc f : replacementFuncs)
               {
                   madeChange = f(subgraph, layer);
                   if (madeChange)
                   {
                       goto nextIteration;
                   }
               }
               nextIteration:;
           }
    );
}

TEST_CASE("SubgraphViewWorkingCopyReplacementFunc")
{
    Graph graph;

    const TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

    std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
    std::iota(constData.begin(), constData.end(), 0);
    ConstTensor constTensor(constInfo, constData);

    // Add the original pattern
    IConnectableLayer* input = graph.AddLayer<InputLayer>(0, "input");
    auto constant = graph.AddLayer<ConstantLayer>("const");

    constant->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    IConnectableLayer* mul      = graph.AddLayer<MultiplicationLayer>("mul");
    IConnectableLayer* output   = graph.AddLayer<OutputLayer>(0, "output");

    // Create connections between layers
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    constant->GetOutputSlot(0).SetTensorInfo(constInfo);
    mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

    input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Add in out of order
    auto view = CreateSubgraphViewFrom({output, input, mul, constant},
                                       {},
                                       {});

    SubgraphView workingCopy = view->GetWorkingCopy();

    // Check the WorkingCopy is as expected before replacement
    CHECK(workingCopy.GetIConnectableLayers().size() == 4);
    int idx=0;
    LayerType expectedSorted[] = {LayerType::Input, LayerType::Constant, LayerType::Multiplication, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSorted](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSorted[idx] == l->GetType()));
                                             idx++;
                                         }
    );

    // Replace Multiplication and Constant with Depthwise
    ReplaceUnsupportedLayers(workingCopy);

    // Check the layers are as expected
    CHECK(workingCopy.GetIConnectableLayers().size() == 3);
    idx=0;
    LayerType expectedSortedReplaced[] = {LayerType::Input, LayerType::DepthwiseConvolution2d, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSortedReplaced](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSortedReplaced[idx] == l->GetType()));
                                             idx++;
                                         }
    );
}

TEST_CASE("SubgraphViewWorkingCopySubstituteSubgraph")
{
    Graph graph;

    auto input = graph.AddLayer<InputLayer>(0, "Input");
    auto activation = graph.AddLayer<ActivationLayer>(ActivationDescriptor{}, "Activation");
    auto output = graph.AddLayer<OutputLayer>(1, "Output");

    input->GetOutputSlot(0).Connect(activation->GetInputSlot(0));
    activation->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Add in out of order
    auto view = CreateSubgraphViewFrom({output, input, activation},
                                       {},
                                       {});

    // Check SubstituteSubgraphView throws when called on original SubgraphView
    SubgraphView temp(input);
    CHECK_THROWS_AS(view->SubstituteSubgraph(temp, input), NullPointerException);

    // Check that GetWorkingCopy() being called on a working copy throws an exception
    auto workingCopy = view->GetWorkingCopy();
    CHECK_THROWS_AS(workingCopy.GetWorkingCopy(), Exception);
}

TEST_CASE("SubgraphViewWorkingCopyOptimizationViews")
{
    Graph graph;

    const TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

    std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
    std::iota(constData.begin(), constData.end(), 0);
    ConstTensor constTensor(constInfo, constData);

    // Add the original pattern
    IConnectableLayer* input = graph.AddLayer<InputLayer>(0, "input");
    auto constant = graph.AddLayer<ConstantLayer>("const");

    constant->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    IConnectableLayer* mul      = graph.AddLayer<MultiplicationLayer>("mul");
    IConnectableLayer* output   = graph.AddLayer<OutputLayer>(0, "output");

    // Create connections between layers
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    constant->GetOutputSlot(0).SetTensorInfo(constInfo);
    mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

    input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    //Add in out of order
    auto view = CreateSubgraphViewFrom({output, input, mul, constant},
                                       {},
                                       {});

    SubgraphView workingCopy = view->GetWorkingCopy();

    // Check the WorkingCopy is as expected before replacement
    int idx=0;
    LayerType expectedSorted[] = {LayerType::Input, LayerType::Constant, LayerType::Multiplication, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSorted](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSorted[idx] == l->GetType()));
                                             idx++;
                                         }
    );

    // Replace Multiplication and Constant with Depthwise
    ReplaceUnsupportedLayers(workingCopy);

    // Check the layers are as expected
    idx=0;
    LayerType expectedSortedReplaced[] = {LayerType::Input, LayerType::DepthwiseConvolution2d, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSortedReplaced](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSortedReplaced[idx] == l->GetType()));
                                             idx++;
                                         }
    );


    // At this stage NPU would take the working copy and create CompiledBlocPtr with it.

    // We will just check that the procompiledLayer can still be added to the optimizationViews via a SubgraphView.
    OptimizationViews optimizationViews;

    CompiledBlobPtr ptr;
    IConnectableLayer* preCompiledLayer = optimizationViews.GetINetwork()->AddPrecompiledLayer(
            PreCompiledDescriptor(view->GetNumInputSlots(), view->GetNumOutputSlots()),
            std::move(ptr),
            EmptyOptional(),
            "pre-compiled");


    optimizationViews.AddSubstitution({ *view, SubgraphView(preCompiledLayer) });
    CHECK(optimizationViews.Validate(*view));
}

TEST_CASE("SubgraphViewWorkingCopyReplaceSlots")
{
    Graph graph;
    const TensorInfo inputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);
    const TensorInfo constInfo({ 1, 1, 1, 16 }, DataType::QAsymmU8, 0.9f, 0, true);
    const TensorInfo outputInfo({ 1, 8, 8, 16 }, DataType::QAsymmU8, 1.0f, 0);

    std::vector<uint8_t> constData(constInfo.GetNumElements(), 0);
    std::iota(constData.begin(), constData.end(), 0);
    ConstTensor constTensor(constInfo, constData);

    // Add the original pattern
    IConnectableLayer* input = graph.AddLayer<InputLayer>(0, "input");
    auto constant = graph.AddLayer<ConstantLayer>("const");

    constant->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    IConnectableLayer* mul      = graph.AddLayer<MultiplicationLayer>("mul");
    IConnectableLayer* output   = graph.AddLayer<OutputLayer>(0, "output");

    // Create connections between layers
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    constant->GetOutputSlot(0).SetTensorInfo(constInfo);
    mul->GetOutputSlot(0).SetTensorInfo(outputInfo);

    input->GetOutputSlot(0).Connect(mul->GetInputSlot(0));
    constant->GetOutputSlot(0).Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    auto view = CreateSubgraphViewFrom({output, input, mul, constant},
                                       CreateIInputsFrom({mul}),
                                        CreateIOutputsFrom({mul}));

    SubgraphView workingCopy = view->GetWorkingCopy();

    // Check the WorkingCopy is as expected before replacement
    CHECK(workingCopy.GetIConnectableLayers().size() == 4);
    int idx=0;
    LayerType expectedSorted[] = {LayerType::Input, LayerType::Constant, LayerType::Multiplication, LayerType::Output};
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSorted](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSorted[idx] == l->GetType()));
                                             idx++;
                                         }
    );

    // Replace Multiplication and Constant with Depthwise
    ReplaceUnsupportedLayers(workingCopy);

    // Check the layers are as expected
    idx=0;
    LayerType expectedSortedReplaced[] = {LayerType::Input, LayerType::DepthwiseConvolution2d, LayerType::Output};
    CHECK(workingCopy.GetIConnectableLayers().size() == 3);
    workingCopy.ForEachIConnectableLayer([&idx, &expectedSortedReplaced](const IConnectableLayer* l)
                                         {
                                             CHECK((expectedSortedReplaced[idx] == l->GetType()));
                                             idx++;
                                         }
    );
}

}

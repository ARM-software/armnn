//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <Graph.hpp>
#include <SubgraphView.hpp>
#include <SubgraphViewSelector.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <fstream>
#include <map>
#include <queue>
#include <random>
#include <chrono>
using namespace armnn;

namespace
{

bool AreAnySubgraphLayersPresentInGraph(const SubgraphView::Layers &subgraphLayers, const Graph &graph)
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
    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());
}

void CompareSubgraphViews(SubgraphViewSelector::SubgraphViewPtr& result,
                          SubgraphViewSelector::SubgraphViewPtr& expected)
{
    // expect both to be valid subgraphs
    BOOST_TEST((result.get() != nullptr));
    BOOST_TEST((expected.get() != nullptr));

    if (result.get() != nullptr && expected.get() != nullptr)
    {
        // try to detect all other obvious errors too, mainly because here
        // we can get a nicer error message from boost, the collection test
        // also report error for these
        BOOST_TEST(result->GetInputSlots().size() == expected->GetInputSlots().size());
        BOOST_TEST(result->GetOutputSlots().size() == expected->GetOutputSlots().size());
        BOOST_TEST(result->GetLayers().size() == expected->GetLayers().size());

        auto resultLayers = ToSortedArray<Layer *>(result->GetLayers().begin(),
                                                   result->GetLayers().end());
        auto expectedLayers = ToSortedArray<Layer *>(expected->GetLayers().begin(),
                                                     expected->GetLayers().end());
        CompareVectors(resultLayers, expectedLayers);

        auto resultInputs = ToSortedArray<InputSlot *>(result->GetInputSlots().begin(),
                                                       result->GetInputSlots().end());
        auto expectedInputs = ToSortedArray<InputSlot *>(expected->GetInputSlots().begin(),
                                                         expected->GetInputSlots().end());
        CompareVectors(resultInputs, expectedInputs);

        auto resultOutputs = ToSortedArray<OutputSlot *>(result->GetOutputSlots().begin(),
                                                         result->GetOutputSlots().end());
        auto expectedOutputs = ToSortedArray<OutputSlot *>(expected->GetOutputSlots().begin(),
                                                           expected->GetOutputSlots().end());
        CompareVectors(resultOutputs, expectedOutputs);
    }
}

} // namespace <anonymous>

BOOST_AUTO_TEST_SUITE(SubgraphSubstitution)

BOOST_AUTO_TEST_CASE(SingleInputSingleOutput)
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
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                            CreateOutputsFrom({convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn = subgraph->GetInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetOutputSlot(0)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(1, 1);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

BOOST_AUTO_TEST_CASE(SingleInputSingleOutputSubstituteGraph)
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
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1}),
                                                                            CreateOutputsFrom({convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn = subgraph->GetInputSlot(0)->GetConnection();
    IInputSlot* subgraphOutputConn = subgraph->GetOutputSlot(0)->GetConnection(0);

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
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

BOOST_AUTO_TEST_CASE(MultiInputSingleOutput)
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
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1, convLayer2}),
                                                                            CreateOutputsFrom({concatLayer}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetInputSlot(0)->GetConnection();
    IOutputSlot* subgraphInputConn2 = subgraph->GetInputSlot(1)->GetConnection();

    IInputSlot* subgraphOutputConn = subgraph->GetOutputSlot(0)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(2, 1);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(1).GetConnection(), subgraphInputConn2);

    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn);
}

BOOST_AUTO_TEST_CASE(SingleInputMultiOutput)
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
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({splitterLayer}),
                                                                            CreateOutputsFrom({convLayer1, convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetInputSlot(0)->GetConnection();

    IInputSlot* subgraphOutputConn1 = subgraph->GetOutputSlot(0)->GetConnection(0);
    IInputSlot* subgraphOutputConn2 = subgraph->GetOutputSlot(1)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(1, 2);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);

    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(1).GetConnection(0), subgraphOutputConn2);
}

BOOST_AUTO_TEST_CASE(MultiInputMultiOutput)
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
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom(CreateInputsFrom({convLayer1, convLayer2}),
                                                                            CreateOutputsFrom({convLayer1, convLayer2}),
                                                                            {});

    // Save sub-graph connections for comparison after substitution
    IOutputSlot* subgraphInputConn1 = subgraph->GetInputSlot(0)->GetConnection();
    IOutputSlot* subgraphInputConn2 = subgraph->GetInputSlot(1)->GetConnection();

    IInputSlot* subgraphOutputConn1 = subgraph->GetOutputSlot(0)->GetConnection(0);
    IInputSlot* subgraphOutputConn2 = subgraph->GetOutputSlot(1)->GetConnection(0);

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(2, 2);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that connections are correct after substitution
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(1).GetConnection(), subgraphInputConn2);

    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
    BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(1).GetConnection(0), subgraphOutputConn2);
}

BOOST_AUTO_TEST_CASE(EraseReplacedLayers)
{
    // Construct graph
    Graph graph;

    graph.AddLayer<InputLayer>(0, "input");

    ViewsDescriptor splitterDescriptor(2);
    Layer* const splitterLayer = graph.AddLayer<SplitterLayer>(splitterDescriptor, "splitter");

    Convolution2dDescriptor convDescriptor;
    Layer* const convLayer1 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv1");
    Layer* const convLayer2 = graph.AddLayer<Convolution2dLayer>(convDescriptor, "conv2");

    OriginsDescriptor concatDescriptor(2);
    Layer* const concatLayer = graph.AddLayer<ConcatLayer>(concatDescriptor, "concat");

    graph.AddLayer<OutputLayer>(0, "output");

    // Construct sub-graph
    SubgraphViewSelector::SubgraphViewPtr subgraph = CreateSubgraphViewFrom({},
                                                                            {},
                                                                            {splitterLayer,
                                                                             convLayer1,
                                                                             convLayer2,
                                                                             concatLayer});

    // Construct dummy pre-compiled layer
    PreCompiledDescriptor preCompiledDescriptor(0, 0);
    Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

    // Save sub-graph layers for later verification
    const SubgraphView::Layers subgraphLayers = subgraph->GetLayers();

    // Substitute sub-graph with pre-compiled layer
    graph.SubstituteSubgraph(*subgraph, preCompiledLayer);

    // Check that the layers belonging to the sub-graph have been erased from the graph after substitution
    BOOST_CHECK(!AreAnySubgraphLayersPresentInGraph(subgraphLayers, graph));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(SubgraphSelection)

BOOST_AUTO_TEST_CASE(SubgraphForEmptyGraph)
{
    Graph graph;
    SubgraphView subgraph(graph);

    BOOST_TEST(subgraph.GetInputSlots().empty());
    BOOST_TEST(subgraph.GetOutputSlots().empty());
    BOOST_TEST(subgraph.GetLayers().empty());
}

BOOST_AUTO_TEST_CASE(SubgraphForEntireGraph)
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

    BOOST_TEST(subgraph.GetInputSlots().empty());
    BOOST_TEST(subgraph.GetOutputSlots().empty());
    BOOST_TEST(subgraph.GetLayers().size() == graph.GetNumLayers());
}

BOOST_AUTO_TEST_CASE(NoSubgraphsForNoMatch)
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    graph.InsertNewLayer<InputLayer>(output->GetInputSlot(0), 0, "input");

    SubgraphViewSelector::Subgraphs subgraphs =
        SubgraphViewSelector::SelectSubgraphs(graph, [](const Layer &) { return false; });

    BOOST_TEST(subgraphs.empty());
}

BOOST_AUTO_TEST_CASE(OneSubgraphsSelectedASingleMatch)
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

    BOOST_TEST(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({output}),
                                               // outputs of 'output' will be empty
                                               CreateOutputsFrom({output}),
                                               {output});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(MultipleLayersSelectedInTheMiddle)
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

    BOOST_TEST(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({mid1}),
                                               CreateOutputsFrom({mid0}),
                                               {mid1, mid0});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(DisjointGraphs)
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
    BOOST_TEST(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));
        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            if (std::find(subgraphs[0]->GetLayers().begin(), subgraphs[0]->GetLayers().end(), i0) !=
                    subgraphs[0]->GetLayers().end())
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

BOOST_AUTO_TEST_CASE(IslandInTheMiddle)
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

    BOOST_TEST(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr& lhs, SubgraphViewSelector::SubgraphViewPtr& rhs)
            {
                return (lhs->GetLayers().size() < rhs->GetLayers().size());
            }
            );

            BOOST_TEST(subgraphs[0]->GetLayers().size() == 2);
            BOOST_TEST(subgraphs[1]->GetLayers().size() == 5);

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], largerSubgraph);
        }
    }
}

BOOST_AUTO_TEST_CASE(MultipleSimpleSubgraphs)
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

    BOOST_TEST(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                {
                    return (lhs->GetLayers().size() < rhs->GetLayers().size());
                }
            );

            BOOST_TEST(subgraphs[0]->GetLayers().size() == 1);
            BOOST_TEST(subgraphs[1]->GetLayers().size() == 2);

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], largerSubgraph);
        }
    }
}

BOOST_AUTO_TEST_CASE(SimpleLinearTest)
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

    BOOST_CHECK(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1}),
                                               CreateOutputsFrom({layerM2}),
                                               {layerM1, layerM2});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(MultiInputSingleOutput)
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

    BOOST_CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1, layerM2}),
                                               CreateOutputsFrom({layerM3}),
                                               {layerM1, layerM2, layerM3});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(SingleInputMultiOutput)
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

    BOOST_CHECK(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({layerM1}),
                                               CreateOutputsFrom({layerM2, layerM3}),
                                               {layerM1, layerM2, layerM3});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(MultiInputMultiOutput)
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


    BOOST_CHECK(subgraphs.size() == 1);
    if (subgraphs.size() == 1)
    {
        auto expected = CreateSubgraphViewFrom(CreateInputsFrom({m1, m2}),
                                               CreateOutputsFrom({m4, m5}),
                                               {m1, m2, m3, m4, m5});

        CompareSubgraphViews(subgraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(ValidMerge)
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

    BOOST_TEST(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            if (subgraphs[0]->GetInputSlots().size() == 1)
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

BOOST_AUTO_TEST_CASE(PropagatedDependencies)
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

    BOOST_TEST(subgraphs.size() == 3);
    if (subgraphs.size() == 3)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));
        BOOST_TEST((subgraphs[2] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr && subgraphs[2].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                [](SubgraphViewSelector::SubgraphViewPtr& lhs, SubgraphViewSelector::SubgraphViewPtr& rhs)
            {
                return (lhs->GetLayers().size() < rhs->GetLayers().size());
            }
            );

            CompareSubgraphViews(subgraphs[0], smallerSubgraph);
            CompareSubgraphViews(subgraphs[1], mediumSubgraph);
            CompareSubgraphViews(subgraphs[2], largerSubgraph);
        }
    }
}

BOOST_AUTO_TEST_CASE(Random)
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
                uint32_t numInputs = 1 + GetRandom(3u);
                OriginsDescriptor concatDesc(numInputs);
                graph.AddLayer<ConcatLayer>(concatDesc, name.c_str());
            }
            for (uint32_t i = 0; i < numSplits; ++i)
            {
                std::string name = "split" + std::to_string(i) + (GetRandomFlag(supportedProb) ? "S" : "N");
                uint32_t numOutputs = 1 + GetRandom(3u);
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
                    uint32_t maxLayerDepthToConnectTo = layerDepths[layer]; // This prevents a connection causing a loop
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
                if (std::find(subgraph->begin(), subgraph->end(), layer) != subgraph->end())
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
            for (InputSlot* inputSlot : subgraph->GetInputSlots())
            {
                std::queue<Layer*> toProcess;
                toProcess.push(&inputSlot->GetConnectedOutputSlot()->GetOwningLayer());
                while (toProcess.size() > 0)
                {
                    Layer* l = toProcess.front();
                    toProcess.pop();

                    BOOST_CHECK(layerToSubgraph[l] != subgraph.get());

                    for (const InputSlot& is : l->GetInputSlots())
                    {
                        toProcess.push(&is.GetConnectedOutputSlot()->GetOwningLayer());
                    }
                }
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(IntegrationTests)

BOOST_AUTO_TEST_CASE(SingleSubgraph)
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

    BOOST_TEST(subgraphs.size() == 1);
    if(subgraphs.size() == 1)
    {
        BOOST_TEST((subgraphs[0] != nullptr));

        if (subgraphs[0].get() != nullptr)
        {
            unsigned int numInputSlots = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetInputSlots().size());
            unsigned int numOutputSlots = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetOutputSlots().size());

            BOOST_TEST((numInputSlots == 1));
            BOOST_TEST((numOutputSlots == 1));

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraphInputConn1 = subgraphs[0]->GetInputSlot(0)->GetConnection();
            IInputSlot* subgraphOutputConn1 = subgraphs[0]->GetOutputSlot(0)->GetConnection(0);

            // Construct dummy pre-compiled layer
            PreCompiledDescriptor preCompiledDescriptor(numInputSlots, numOutputSlots);
            Layer* const preCompiledLayer = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor, "pre-compiled");

            // Substitute sub-graph with pre-compiled layer
            graph.SubstituteSubgraph(*subgraphs[0], preCompiledLayer);

            // Check that connections are correct after substitution
            BOOST_CHECK_EQUAL(preCompiledLayer->GetInputSlot(0).GetConnection(), subgraphInputConn1);

            BOOST_CHECK_EQUAL(preCompiledLayer->GetOutputSlot(0).GetConnection(0), subgraphOutputConn1);
        }
    }
}

BOOST_AUTO_TEST_CASE(MultipleSubgraphs)
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

    BOOST_TEST(subgraphs.size() == 2);
    if(subgraphs.size() == 2)
    {
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            //Sort subgraphs by their inputSlot size.
            std::sort(subgraphs.begin(), subgraphs.end(),
                      [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                      {
                          return (lhs->GetInputSlots().size() < rhs->GetInputSlots().size());
                      }
            );

            unsigned int numInputSlots1  = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetInputSlots().size());
            unsigned int numOutputSlots1 = armnn::numeric_cast<unsigned int>(subgraphs[0]->GetOutputSlots().size());

            unsigned int numInputSlots2  = armnn::numeric_cast<unsigned int>(subgraphs[1]->GetInputSlots().size());
            unsigned int numOutputSlots2 = armnn::numeric_cast<unsigned int>(subgraphs[1]->GetOutputSlots().size());

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraph1InputConn  = subgraphs[0]->GetInputSlot(0)->GetConnection();
            IInputSlot* subgraph1OutputConn1 = subgraphs[0]->GetOutputSlot(0)->GetConnection(0);
            IInputSlot* subgraph1OutputConn2 = subgraphs[0]->GetOutputSlot(1)->GetConnection(0);

            // Save sub-graph connections for comparison after substitution
            IOutputSlot* subgraph2InputConn1 = subgraphs[1]->GetInputSlot(0)->GetConnection();
            IOutputSlot* subgraph2InputConn2 = subgraphs[1]->GetInputSlot(1)->GetConnection();
            IInputSlot* subgraph2OutputConn  = subgraphs[1]->GetOutputSlot(0)->GetConnection(0);

            PreCompiledDescriptor preCompiledDescriptor1(numInputSlots1, numOutputSlots1);
            Layer* const preCompiledLayer1 = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor1, "pre-compiled1");

            PreCompiledDescriptor preCompiledDescriptor2(numInputSlots2, numOutputSlots2);
            Layer* const preCompiledLayer2 = graph.AddLayer<PreCompiledLayer>(preCompiledDescriptor2, "pre-compiled2");

            // Substitute sub-graph with pre-compiled layer
            graph.SubstituteSubgraph(*subgraphs[0], preCompiledLayer1);
            graph.SubstituteSubgraph(*subgraphs[1], preCompiledLayer2);

            // Check that connections are correct after substitution
            BOOST_CHECK_EQUAL(preCompiledLayer1->GetInputSlot(0).GetConnection(), subgraph1InputConn);
            BOOST_CHECK_EQUAL(preCompiledLayer1->GetOutputSlot(0).GetConnection(0), subgraph1OutputConn1);
            BOOST_CHECK_EQUAL(preCompiledLayer1->GetOutputSlot(1).GetConnection(0), subgraph1OutputConn2);

            BOOST_CHECK_EQUAL(preCompiledLayer2->GetInputSlot(0).GetConnection(), subgraph2InputConn1);
            BOOST_CHECK_EQUAL(preCompiledLayer2->GetInputSlot(1).GetConnection(), subgraph2InputConn2);
            BOOST_CHECK_EQUAL(preCompiledLayer2->GetOutputSlot(0).GetConnection(0), subgraph2OutputConn);
        }
    }
}

BOOST_AUTO_TEST_CASE(SubgraphCycles)
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

    BOOST_TEST(subgraphs.size() == 2);
    if (subgraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subgraphs[0] != nullptr));
        BOOST_TEST((subgraphs[1] != nullptr));

        if (subgraphs[0].get() != nullptr && subgraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subgraphs.begin(), subgraphs.end(),
                      [](SubgraphViewSelector::SubgraphViewPtr & lhs, SubgraphViewSelector::SubgraphViewPtr & rhs)
                          {
                              return (lhs->GetLayers().size() < rhs->GetLayers().size());
                          }
            );

            // one subgraph needs to be size=1 and the other one is 4
            BOOST_TEST(subgraphs[0]->GetLayers().size() == 1);
            BOOST_TEST(subgraphs[1]->GetLayers().size() == 2);

            CompareSubgraphViews(subgraphs[0], outputSubgraph);
            CompareSubgraphViews(subgraphs[1], inputSubgraph);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

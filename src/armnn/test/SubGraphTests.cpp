//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <armnn/ArmNN.hpp>

#include <Graph.hpp>
#include <SubGraph.hpp>
#include <SubGraphSelector.hpp>

#include <backends/CpuTensorHandle.hpp>

using namespace armnn;

namespace
{

//
// this helper only works if all layers where the inputs connect to are not selected
//
SubGraph::InputSlots CreateInputsFrom(const std::vector<Layer *> & layers)
{
    SubGraph::InputSlots result;
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
SubGraph::OutputSlots CreateOutputsFrom(const std::vector<Layer *> & layers)
{
    SubGraph::OutputSlots result;
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
// resulting subgraph, so the pass bay value is intentional
//
SubGraphSelector::SubGraphPtr CreateSubGraphFrom(SubGraph::InputSlots inputs,
                                                 SubGraph::OutputSlots outputs,
                                                 SubGraph::Layers layers)
{
    return std::make_unique<SubGraph>(std::move(inputs), std::move(outputs), std::move(layers));
}

template <typename T, typename Iterator>
std::vector<T> ToSortedArray(Iterator begin, Iterator end)
{
    std::vector<T> result(begin, end);
    std::sort(result.begin(), result.end());
    return result;
}

template <typename T>
void CompareVectors(const std::vector<T> & result, const std::vector<T> & expected)
{
    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), expected.begin(), expected.end());
}

void CompareSubGraphs(SubGraphSelector::SubGraphPtr & result,
                      SubGraphSelector::SubGraphPtr & expected)
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

BOOST_AUTO_TEST_SUITE(SubGraphSelection)

BOOST_AUTO_TEST_CASE(NoSubGraphsForNoMatch)
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    graph.InsertNewLayer<InputLayer>(output->GetInputSlot(0), 0, "input");

    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(graph, [](const Layer &) { return false; });

    BOOST_TEST(subGraphs.empty());
}

BOOST_AUTO_TEST_CASE(OneSubGraphsSelectedASingleMatch)
{
    Graph graph;

    auto output = graph.AddLayer<OutputLayer>(0, "output");
    graph.InsertNewLayer<InputLayer>(output->GetInputSlot(0), 0, "input");

    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(
            graph,
            // select the output layer only
            [](const Layer & l)
            {
                bool isOutput = l.GetNameStr().compare("output") == 0;
                return isOutput;
            });

    BOOST_TEST(subGraphs.size() == 1);
    if (subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({output}),
                                           // outputs of 'output' will be empty
                                           CreateOutputsFrom({output}),
                                           {output});

        CompareSubGraphs(subGraphs[0], expected);
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

    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation);
                return toSelect;
            });

    BOOST_TEST(subGraphs.size() == 1);
    if (subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({mid1}),
                                           CreateOutputsFrom({mid0}),
                                           {mid1, mid0});

        CompareSubGraphs(subGraphs[0], expected);
    }
}

BOOST_AUTO_TEST_CASE(IslandInTheMiddle)
{
    // This case represent the scenario when a non-selected X1 node placed in the middle
    // of the selected M* nodes:
    //
    // X0 -> M1 -> M2 -> M3 -> X2
    // X0 -> M4 -> X1 -> M5 -> X2
    //
    /*
          X0
          / \
        M1   M4
         |   |
        M2   X1 < the island in the middle !
         |   |
        M3   M5
          \ /
          X2
    */
    // The expected result for this is that M1,M2,M3,M4 will be part of one subgraph and
    // M5 will be part of another subgraph and the input and output slots in the subgraphs
    // will be set accordingly.
    //
    Graph graph;

    OriginsDescriptor mergerDescriptor(2);
    auto x2 = graph.AddLayer<MergerLayer>(mergerDescriptor, "x2");
    auto m3 = graph.InsertNewLayer<ActivationLayer>(x2->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m3");
    auto m2 = graph.InsertNewLayer<ActivationLayer>(m3->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m2");
    auto m1 = graph.InsertNewLayer<ActivationLayer>(m2->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m1");
    auto x0 = graph.InsertNewLayer<InputLayer>(m1->GetInputSlot(0), 0, "x0");

    auto m5 = graph.InsertNewLayer<ActivationLayer>(x2->GetInputSlot(1),
                                                    ActivationDescriptor{},
                                                    "m5");
    auto x1 = graph.InsertNewLayer<Convolution2dLayer>(m5->GetInputSlot(0),
                                                       Convolution2dDescriptor{},
                                                       "x1");
    auto m4 = graph.InsertNewLayer<ActivationLayer>(x1->GetInputSlot(0),
                                                    ActivationDescriptor{},
                                                    "m4");

    // Connect the other branch to the input layer
    x0->GetOutputSlot(0).Connect(m4->GetInputSlot(0));

    // All selected 'M*' layers will be of Activation type
    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation);
                return toSelect;
            });

    // expected results to test against
    auto largerSubGraph = CreateSubGraphFrom(CreateInputsFrom({m1, m4}),
                                             CreateOutputsFrom({m3, m4}),
                                             {m1, m4, m2, m3});

    auto smallerSubGraph = CreateSubGraphFrom(CreateInputsFrom({m5}),
                                              CreateOutputsFrom({m5}),
                                              {m5});

    BOOST_TEST(subGraphs.size() == 2);
    if (subGraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subGraphs[0] != nullptr));
        BOOST_TEST((subGraphs[1] != nullptr));

        if (subGraphs[0].get() != nullptr && subGraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subGraphs.begin(), subGraphs.end(),
                [](SubGraphSelector::SubGraphPtr & lhs, SubGraphSelector::SubGraphPtr & rhs)
                {
                    return (lhs->GetLayers().size() < rhs->GetLayers().size());
                }
            );

            // one subgraph needs to be size=1 and the other one is 4
            BOOST_TEST(subGraphs[0]->GetLayers().size() == 1);
            BOOST_TEST(subGraphs[1]->GetLayers().size() == 4);

            CompareSubGraphs(subGraphs[0], smallerSubGraph);
            CompareSubGraphs(subGraphs[1], largerSubGraph);
        }
    }
}

BOOST_AUTO_TEST_CASE(MultipleSimpleSubGraphs)
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
    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(
            graph,
            // select the middle layers only
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation);
                return toSelect;
            });

    // expected results to test against
    auto largerSubGraph = CreateSubGraphFrom(CreateInputsFrom({m1}),
                                             CreateOutputsFrom({m2}),
                                             {m1, m2});

    auto smallerSubGraph = CreateSubGraphFrom(CreateInputsFrom({m3}),
                                              CreateOutputsFrom({m3}),
                                              {m3});

    BOOST_TEST(subGraphs.size() == 2);
    if (subGraphs.size() == 2)
    {
        // we need to have valid subgraph pointers here
        BOOST_TEST((subGraphs[0] != nullptr));
        BOOST_TEST((subGraphs[1] != nullptr));

        if (subGraphs[0].get() != nullptr && subGraphs[1].get() != nullptr)
        {
            // sort the subgraphs by layer size, so it is simpler to test
            std::sort(subGraphs.begin(), subGraphs.end(),
                [](SubGraphSelector::SubGraphPtr & lhs, SubGraphSelector::SubGraphPtr & rhs)
                {
                    return (lhs->GetLayers().size() < rhs->GetLayers().size());
                }
            );

            BOOST_TEST(subGraphs[0]->GetLayers().size() == 1);
            BOOST_TEST(subGraphs[1]->GetLayers().size() == 2);

            CompareSubGraphs(subGraphs[0], smallerSubGraph);
            CompareSubGraphs(subGraphs[1], largerSubGraph);
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

    SubGraphSelector::SubGraphs subGraphs =
            SubGraphSelector::SelectSubGraphs(
                    graph,
                    // select the activation layers M1 and M2
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation);
                        return toSelect;
                    });

    BOOST_CHECK(subGraphs.size() == 1);
    if(subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({layerM1}),
                                           CreateOutputsFrom({layerM2}),
                                           {layerM1, layerM2});

        CompareSubGraphs(subGraphs[0], expected);
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

    SubGraphSelector::SubGraphs subGraphs =
            SubGraphSelector::SelectSubGraphs(
                    graph,
                    // select Activation and Addition Layers M1, M2 and M3
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation
                                         || l.GetType() == LayerType::Addition);
                        return toSelect;
                    });

    BOOST_CHECK(subGraphs.size() == 1);
    if (subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({layerM1, layerM2}),
                                           CreateOutputsFrom({layerM3}),
                                           {layerM1, layerM2, layerM3});

        CompareSubGraphs(subGraphs[0], expected);
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

    //      X2
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

    SubGraphSelector::SubGraphs subGraphs =
            SubGraphSelector::SelectSubGraphs(
                    graph,
                    // select Activation and Splitter Layers M1, M2 and M3
                    [](const Layer & l)
                    {
                        bool toSelect = (l.GetType() == LayerType::Activation
                                         || l.GetType() == LayerType::Splitter);
                        return toSelect;
                    });

    BOOST_CHECK(subGraphs.size() == 1);
    if(subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({layerM1}),
                                           CreateOutputsFrom({layerM2, layerM3}),
                                           {layerM1, layerM2, layerM3});

        CompareSubGraphs(subGraphs[0], expected);
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
    OriginsDescriptor mergerDescriptor(2);

    auto x1 = graph.AddLayer<InputLayer>(0, "x1");
    auto x2 = graph.AddLayer<InputLayer>(1, "x2");

    auto m1 = graph.AddLayer<ActivationLayer>(activationDefaults, "m1");
    auto m2 = graph.AddLayer<ActivationLayer>(activationDefaults, "m2");
    auto m3 = graph.AddLayer<MergerLayer>(mergerDescriptor, "m3");

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


    SubGraphSelector::SubGraphs subGraphs =
        SubGraphSelector::SelectSubGraphs(
            graph,
            // select Activation and Merger Layers M1, M2, M3, M4, M5
            [](const Layer & l)
            {
                bool toSelect = (l.GetType() == LayerType::Activation
                                 || l.GetType() == LayerType::Merger);
                return toSelect;
            });


    BOOST_CHECK(subGraphs.size() == 1);
    if (subGraphs.size() == 1)
    {
        auto expected = CreateSubGraphFrom(CreateInputsFrom({m1, m2}),
                                           CreateOutputsFrom({m4, m5}),
                                           {m1, m2, m3, m4, m5});

        CompareSubGraphs(subGraphs[0], expected);
    }
}

BOOST_AUTO_TEST_SUITE_END()

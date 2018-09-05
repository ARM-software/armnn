//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include "Graph.hpp"
#include "Observable.hpp"

BOOST_AUTO_TEST_SUITE(Observable)

BOOST_AUTO_TEST_CASE(AddedLayerObservableTest)
{
    armnn::Graph graph;

    // Create a graph observable
    armnn::AddedLayerObservable layerObservable(graph);

    // Add a few layers
    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    auto input = graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Check the observable has observed the changes
    std::list<armnn::Layer*> testLayers({ output, input });

    BOOST_CHECK_EQUAL_COLLECTIONS(layerObservable.begin(), layerObservable.end(),
                                  testLayers.begin(), testLayers.end());
}

BOOST_AUTO_TEST_CASE(ClearAddedLayerObservableTest)
{
    armnn::Graph graph;

    // Create a graph observable
    armnn::AddedLayerObservable addedLayerObservable(graph);

    // Add a few layers
    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    addedLayerObservable.Clear();

    // Check the observable has observed the changes
    std::list<armnn::Layer*> emptyList({});

    BOOST_CHECK_EQUAL_COLLECTIONS(addedLayerObservable.begin(), addedLayerObservable.end(),
                                  emptyList.begin(), emptyList.end());
}

BOOST_AUTO_TEST_CASE(ErasedLayerNamesObservableTest)
{
    armnn::Graph graph;

    // Create a graph observable
    armnn::ErasedLayerNamesObservable erasedLayerNamesObservable(graph);

    // Add a few layers
    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    graph.EraseLayer(output);

    // Check the observable has observed the changes
    std::list<std::string> testList({"output"});

    BOOST_CHECK_EQUAL_COLLECTIONS(erasedLayerNamesObservable.begin(), erasedLayerNamesObservable.end(),
                                  testList.begin(), testList.end());
}

BOOST_AUTO_TEST_CASE(ClearErasedLayerNamesObservableTest)
{
    armnn::Graph graph;

    // Create a graph observable
    armnn::ErasedLayerNamesObservable erasedLayerNamesObservable(graph);

    // Add a few layers
    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    graph.EraseLayer(output);

    erasedLayerNamesObservable.Clear();

    // Check the observable has observed the changes
    std::list<std::string> emptyList({});

    BOOST_CHECK_EQUAL_COLLECTIONS(erasedLayerNamesObservable.begin(), erasedLayerNamesObservable.end(),
                                  emptyList.begin(), emptyList.end());
}

BOOST_AUTO_TEST_SUITE_END()


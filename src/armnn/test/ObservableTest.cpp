//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <doctest/doctest.h>

#include "Graph.hpp"
#include "Observable.hpp"

TEST_SUITE("Observable")
{
TEST_CASE("AddedLayerObservableTest")
{
    armnn::Graph graph;

    // Create a graph observable
    armnn::AddedLayerObservable layerObservable(graph);

    // Add a few layers
    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    auto input = graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Check the observable has observed the changes
    std::list<armnn::Layer*> testLayers({ output, input });

    CHECK(std::equal(layerObservable.begin(), layerObservable.end(),
                                  testLayers.begin(), testLayers.end()));
}

TEST_CASE("ClearAddedLayerObservableTest")
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

    CHECK(std::equal(addedLayerObservable.begin(), addedLayerObservable.end(),
                                  emptyList.begin(), emptyList.end()));
}

TEST_CASE("ErasedLayerNamesObservableTest")
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

    CHECK(std::equal(erasedLayerNamesObservable.begin(), erasedLayerNamesObservable.end(),
                                  testList.begin(), testList.end()));
}

TEST_CASE("ClearErasedLayerNamesObservableTest")
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

    CHECK(std::equal(erasedLayerNamesObservable.begin(), erasedLayerNamesObservable.end(),
                                  emptyList.begin(), emptyList.end()));
}

}


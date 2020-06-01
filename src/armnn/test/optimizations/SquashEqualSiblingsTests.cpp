//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TestUtils.hpp"

#include <Optimizer.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn::optimizations;

BOOST_AUTO_TEST_CASE(SquashEqualSiblingsTest)
{
    armnn::Graph graph;

    armnn::LayerBindingId outputId = 0;

    const armnn::TensorInfo info({ 1, 2, 3, 5 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 5, 2, 3 }, armnn::DataType::Float32);

    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    // Inserts equal permutes, equal reshapes and something else.
    const armnn::PermuteDescriptor permDesc({ 0, 2, 3, 1 });
    const armnn::ReshapeDescriptor reshapeDesc{ { 1, 3, 1, 5 } };

    armnn::Layer* layer;

    layer = graph.AddLayer<armnn::PermuteLayer>(permDesc, "");
    layer->GetOutputSlot().SetTensorInfo(permuted);
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::ReshapeLayer>(reshapeDesc, "");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::FloorLayer>("");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::ReshapeLayer>(reshapeDesc, "");
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    layer = graph.AddLayer<armnn::PermuteLayer>(permDesc, "");
    layer->GetOutputSlot().SetTensorInfo(permuted);
    layer->GetOutputSlot().Connect(graph.AddLayer<armnn::OutputLayer>(outputId++, "")->GetInputSlot(0));
    input->GetOutputSlot().Connect(layer->GetInputSlot(0));

    BOOST_TEST(CheckSequence(
        graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>, &IsLayerOfType<armnn::PermuteLayer>,
        &IsLayerOfType<armnn::ReshapeLayer>, &IsLayerOfType<armnn::FloorLayer>, &IsLayerOfType<armnn::ReshapeLayer>,
        &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::OutputLayer>, &IsLayerOfType<armnn::OutputLayer>,
        &IsLayerOfType<armnn::OutputLayer>, &IsLayerOfType<armnn::OutputLayer>, &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(SquashEqualPermuteSiblings(), SquashEqualReshapeSiblings()));

    // The permutes and reshapes are squashed.

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::ReshapeLayer>,
                             &IsLayerOfType<armnn::FloorLayer>, &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>, &IsLayerOfType<armnn::OutputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>, &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_SUITE_END()
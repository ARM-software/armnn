//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TestUtils.hpp"

#include <Optimizer.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn::optimizations;

BOOST_AUTO_TEST_CASE(InsertDebugOptimizationTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 2, 2, 1, 3 }, armnn::DataType::Float32);

    // Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto floor = graph.AddLayer<armnn::FloorLayer>("floor");
    floor->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot().Connect(output->GetInputSlot(0));

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::FloorLayer>, &IsLayerOfType<armnn::OutputLayer>));

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(InsertDebugLayer()));

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::DebugLayer>, &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::DebugLayer>, &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_SUITE_END()
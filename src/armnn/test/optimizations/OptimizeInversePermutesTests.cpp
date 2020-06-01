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

BOOST_AUTO_TEST_CASE(OptimizeInversePermutesTest)
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Inserts two permutes, one the inverse of the other.
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0), armnn::PermuteDescriptor({ 0, 2, 3, 1 }),
                                              "perm0231");
    graph.InsertNewLayer<armnn::PermuteLayer>(output->GetInputSlot(0), armnn::PermuteDescriptor({ 0, 3, 1, 2 }),
                                              "perm0312");

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeInversePermutes()));

    // The permutes are removed.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_CASE(OptimizeInverseTransposesTest)
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Inserts two permutes, one the inverse of the other.
    graph.InsertNewLayer<armnn::TransposeLayer>(output->GetInputSlot(0),
                                                armnn::TransposeDescriptor({ 0, 3, 1, 2 }),
                                                "transpose0312");
    graph.InsertNewLayer<armnn::TransposeLayer>(output->GetInputSlot(0),
                                                armnn::TransposeDescriptor({ 0, 2, 3, 1 }),
                                                "transpose0231");

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::TransposeLayer>, &IsLayerOfType<armnn::TransposeLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeInverseTransposes()));

    // The permutes are removed.
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));
}

BOOST_AUTO_TEST_SUITE_END()
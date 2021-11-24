//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("OptimizeConsecutiveReshapesTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo info0({ 1, 2, 3, 5 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");
    auto input  = graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    input->GetOutputHandler().SetTensorInfo(info0);

    {
        // Inserts two reshapes.
        const armnn::TensorInfo info1({ 1, 30, 1, 1 }, armnn::DataType::Float32);
        const armnn::TensorInfo info2({ 1, 2, 1, 15 }, armnn::DataType::Float32);

        std::string reshape1Name = "reshape1";
        std::string reshape2Name = "reshape2";

        auto reshape1 = graph.InsertNewLayer<armnn::ReshapeLayer>(
            output->GetInputSlot(0), armnn::ReshapeDescriptor{ info1.GetShape() }, reshape1Name.c_str());
        auto reshape2 = graph.InsertNewLayer<armnn::ReshapeLayer>(
            output->GetInputSlot(0), armnn::ReshapeDescriptor{ info2.GetShape() }, reshape2Name.c_str());

        reshape1->GetOutputHandler().SetTensorInfo(info1);
        reshape2->GetOutputHandler().SetTensorInfo(info2);

        CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::ReshapeLayer>, &IsLayerOfType<armnn::ReshapeLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));

        armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeConsecutiveReshapes()));

        auto checkReshape = [&info2](const armnn::Layer* const layer) -> bool {
            const auto reshapeLayer = static_cast<const armnn::ReshapeLayer*>(layer);
            return IsLayerOfType<armnn::ReshapeLayer>(layer) &&
                   (reshapeLayer->GetParameters().m_TargetShape == info2.GetShape()) &&
                   (reshapeLayer->GetOutputHandler().GetTensorInfo().GetShape() == info2.GetShape());
        };

        // The two reshapes are replaced by a single equivalent reshape.
        CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>, checkReshape,
                                 &IsLayerOfType<armnn::OutputLayer>));

        // Check the new reshape layer has the other two reshapes as related layers
        std::list<std::string> testRelatedLayers = { reshape2Name, reshape1Name };

        CHECK(CheckRelatedLayers<armnn::ReshapeLayer>(graph, testRelatedLayers));
    }

    {
        // Inserts a reshape to the input shape.
        auto reshapeToIn = graph.InsertNewLayer<armnn::ReshapeLayer>(
            output->GetInputSlot(0), armnn::ReshapeDescriptor{ info0.GetShape() }, "reshapeToIn");

        reshapeToIn->GetOutputHandler().SetTensorInfo(info0);

        armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(OptimizeConsecutiveReshapes()));

        // The two reshapes are removed.
        CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                 &IsLayerOfType<armnn::OutputLayer>));
    }
}

}
//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("TransposeAsReshapeTest")
{
    armnn::Graph graph;

    std::string transposeLayerName = "transpose";

    const armnn::TensorInfo infoIn({ 1, 2, 3, 1 }, armnn::DataType::Float32);
    const armnn::TensorInfo infoOut({ 1, 1, 2, 3 }, armnn::DataType::Float32);

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input")
            ->GetOutputHandler()
            .SetTensorInfo(infoIn);

    // Inserts transpose.
    graph
            .InsertNewLayer<armnn::TransposeLayer>(output->GetInputSlot(0), armnn::TransposeDescriptor({ 0, 3, 1, 2 }),
                                                   transposeLayerName.c_str())
            ->GetOutputHandler()
            .SetTensorInfo(infoOut);

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::TransposeLayer>, &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(TransposeAsReshape()));

    // The transpose is replaced by an equivalent reshape.

    auto checkReshape = [&infoOut](const armnn::Layer* const layer) -> bool {
        const auto reshapeLayer = static_cast<const armnn::ReshapeLayer*>(layer);
        return IsLayerOfType<armnn::ReshapeLayer>(layer) &&
               (reshapeLayer->GetParameters().m_TargetShape == infoOut.GetShape()) &&
               (reshapeLayer->GetOutputHandler().GetTensorInfo().GetShape() == infoOut.GetShape());
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>, checkReshape,
                             &IsLayerOfType<armnn::OutputLayer>));

    std::list<std::string> testRelatedLayers = { transposeLayerName };
    CHECK(CheckRelatedLayers<armnn::ReshapeLayer>(graph, testRelatedLayers));
}

}
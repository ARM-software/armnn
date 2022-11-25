//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("Fp32NetworkToFp16OptimizationTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo infoFP32({ 2, 2, 1, 3 }, armnn::DataType::Float32);

    // Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(infoFP32);

    auto floor = graph.AddLayer<armnn::FloorLayer>("floor");
    floor->GetOutputSlot().SetTensorInfo(infoFP32);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(floor->GetInputSlot(0));
    floor->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::FloorLayer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(Fp32NetworkToFp16Converter()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                                                      &IsLayerOfType<armnn::FloorLayer>,
                                                      &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                                                      &IsLayerOfType<armnn::OutputLayer>));

    CHECK(floor->GetDataType() == armnn::DataType::Float16);
    CHECK(floor->GetInputSlot(0).GetConnectedOutputSlot()->GetTensorInfo().GetDataType() == armnn::DataType::Float16);
    CHECK(floor->GetOutputSlot(0).GetTensorInfo().GetDataType() == armnn::DataType::Float16);
}

}
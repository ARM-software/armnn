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

TEST_CASE("MovePermuteUpTest")
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo permuted({ 1, 3, 5, 2 }, armnn::DataType::Float32);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    std::string permuteLayerName = "original_permute";

    // Insert permute
    head = graph.InsertNewLayer<armnn::PermuteLayer>(head->GetInputSlot(0), armnn::PermuteDescriptor({ 0, 2, 3, 1 }),
                                                     permuteLayerName.c_str());

    head->GetOutputHandler().SetTensorInfo(permuted);

    // Inserts layers that don't care about data format.
    head = graph.InsertNewLayer<armnn::ActivationLayer>(head->GetInputSlot(0), armnn::ActivationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::AdditionLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    // Inserts input for 2nd input of Addition.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler()
        .SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FakeQuantizationLayer>(head->GetInputSlot(0),
                                                              armnn::FakeQuantizationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::FloorLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MemCopyLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::MultiplicationLayer>(head->GetInputSlot(0), "");
    head->GetOutputHandler().SetTensorInfo(info);

    // Inserts input for 2nd input of Multiplication.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(1), inputId++, "")
        ->GetOutputHandler()
        .SetTensorInfo(info);

    // Inserts input.
    graph.InsertNewLayer<armnn::InputLayer>(head->GetInputSlot(0), inputId++, "")
        ->GetOutputHandler()
        .SetTensorInfo(info);

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>, &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::MultiplicationLayer>, &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>, &IsLayerOfType<armnn::FakeQuantizationLayer>,
                             &IsLayerOfType<armnn::AdditionLayer>, &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(MovePermuteUp()));

    // The permute is moved to the top. New permutes for layers with multiple inputs.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>, &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::PermuteLayer>,
                             &IsLayerOfType<armnn::PermuteLayer>, &IsLayerOfType<armnn::MultiplicationLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>, &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::FakeQuantizationLayer>, &IsLayerOfType<armnn::AdditionLayer>,
                             &IsLayerOfType<armnn::ActivationLayer>, &IsLayerOfType<armnn::OutputLayer>));

    std::list<std::string> testRelatedLayers = { permuteLayerName };

    CHECK(CheckRelatedLayers<armnn::PermuteLayer>(graph, testRelatedLayers));
}

}
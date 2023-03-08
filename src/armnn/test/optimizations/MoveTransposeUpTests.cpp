//
// Copyright © 2020-2021,2023 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("MoveTransposeUpTest")
{
    const armnn::TensorInfo info({ 1, 5, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo transposed({ 1, 3, 5, 2 }, armnn::DataType::Float32);

    armnn::Graph graph;

    armnn::LayerBindingId inputId = 0;

    armnn::Layer* head = graph.AddLayer<armnn::OutputLayer>(0, "output");

    std::string transposeLayerName = "original_transpose";

    // Insert transpose
    head = graph.InsertNewLayer<armnn::TransposeLayer>(head->GetInputSlot(0),
                                                       armnn::TransposeDescriptor({ 0, 3, 1, 2 }),
                                                       transposeLayerName.c_str());

    head->GetOutputHandler().SetTensorInfo(transposed);

    // Inserts layers that don't care about data format.
    head = graph.InsertNewLayer<armnn::ActivationLayer>(head->GetInputSlot(0), armnn::ActivationDescriptor{}, "");
    head->GetOutputHandler().SetTensorInfo(info);

    head = graph.InsertNewLayer<armnn::ElementwiseBinaryLayer>(head->GetInputSlot(0), armnn::BinaryOperation::Add, "");
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

    head = graph.InsertNewLayer<armnn::ElementwiseBinaryLayer>(head->GetInputSlot(0), armnn::BinaryOperation::Mul, "");
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
                             &IsLayerOfType<armnn::ElementwiseBinaryLayer>, &IsLayerOfType<armnn::MemCopyLayer>,
                             &IsLayerOfType<armnn::FloorLayer>, &IsLayerOfType<armnn::FakeQuantizationLayer>,
                             &IsLayerOfType<armnn::ElementwiseBinaryLayer>, &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::TransposeLayer>, &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(MoveTransposeUp()));

    // The transpose is moved to the top. New transposes for layers with multiple inputs.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::InputLayer>, &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::TransposeLayer>, &IsLayerOfType<armnn::TransposeLayer>,
                             &IsLayerOfType<armnn::TransposeLayer>, &IsLayerOfType<armnn::ElementwiseBinaryLayer>,
                             &IsLayerOfType<armnn::MemCopyLayer>, &IsLayerOfType<armnn::FloorLayer>,
                             &IsLayerOfType<armnn::FakeQuantizationLayer>,
                             &IsLayerOfType<armnn::ElementwiseBinaryLayer>, &IsLayerOfType<armnn::ActivationLayer>,
                             &IsLayerOfType<armnn::OutputLayer>));

    std::list<std::string> testRelatedLayers = { transposeLayerName };

    CHECK(CheckRelatedLayers<armnn::TransposeLayer>(graph, testRelatedLayers));
}

}
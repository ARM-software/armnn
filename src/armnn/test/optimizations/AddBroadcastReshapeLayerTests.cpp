//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace optimizations;

void AddBroadcastReshapeLayerOptimizerTest(const TensorInfo& info0,
                                           const TensorInfo& info1,
                                           const TensorInfo& outputInfo,
                                           const std::string& reshapeLayerName,
                                           const TensorShape& expectedReshapeShape,
                                           const DataType expectedDataType)
{
    Graph graph;

    auto input0 = graph.AddLayer<InputLayer>(0, "input0");
    auto input1 = graph.AddLayer<InputLayer>(1, "input1");
    auto add = graph.AddLayer<AdditionLayer>("add");
    auto output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().SetTensorInfo(info0);
    input1->GetOutputSlot().SetTensorInfo(info1);
    add->GetOutputSlot().SetTensorInfo(outputInfo);

    input0->GetOutputSlot().Connect(add->GetInputSlot(0));
    input1->GetOutputSlot().Connect(add->GetInputSlot(1));
    add->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, reshapeLayerName);
    CHECK(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    CHECK((addedReshapeTensorInfo.GetShape() == expectedReshapeShape));
    CHECK((addedReshapeTensorInfo.GetDataType() == expectedDataType));
}

TEST_CASE("AddBroadcastReshapeLayerSimpleTest")
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 1 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, info0, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 1, 1 }),
                                          DataType::Float32);
}

TEST_CASE("AddBroadcastReshapeLayer1DTest")
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 1, 1, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 1, 5 }),
                                          DataType::Float32);
}

TEST_CASE("AddBroadcastReshapeLayer2DTest")
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 3, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 3, 5 }),
                                          DataType::Float32);
}

TEST_CASE("AddBroadcastReshapeLayer3DTest")
{
    const TensorInfo info0({ 2, 1, 1, 1 }, DataType::Float32);
    const TensorInfo info1({ 3, 4, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 2, 3, 4, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 3, 4, 5 }),
                                          DataType::Float32);
}

TEST_CASE("AddBroadcastReshapeLayer3DMergedTest")
{
    const TensorInfo info0({ 2, 3, 1, 1 }, DataType::Float32);
    const TensorInfo info1({ 3, 4, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 2, 3, 4, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 3, 4, 5 }),
                                          DataType::Float32);
}

TEST_CASE("AddBroadcastReshapeLayerSubtractionTest")
{
    Graph graph;
    const TensorInfo info0({ 5 }, DataType::Float32);
    const TensorInfo info1({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::Float32);

    auto input0 = graph.AddLayer<InputLayer>(0, "input0");
    auto input1 = graph.AddLayer<InputLayer>(1, "input1");
    auto sub = graph.AddLayer<SubtractionLayer>("sub");
    auto output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().SetTensorInfo(info0);
    input1->GetOutputSlot().SetTensorInfo(info1);
    sub->GetOutputSlot().SetTensorInfo(outputInfo);

    input0->GetOutputSlot().Connect(sub->GetInputSlot(0));
    input1->GetOutputSlot().Connect(sub->GetInputSlot(1));
    sub->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<SubtractionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<SubtractionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:sub-0");
    CHECK(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    CHECK((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 1, 5 })));
    CHECK((addedReshapeTensorInfo.GetDataType() == DataType::Float32));
}

TEST_CASE("AddBroadcastReshapeLayerDivisionTest")
{
    Graph graph;
    const TensorInfo info0({ 1, 4, 5 }, DataType::QAsymmS8);
    const TensorInfo info1({ 1, 2, 4, 5 }, DataType::QAsymmS8);
    const TensorInfo outputInfo({ 1, 2, 4, 5 }, DataType::QAsymmS8);

    auto input0 = graph.AddLayer<InputLayer>(0, "input0");
    auto input1 = graph.AddLayer<InputLayer>(1, "input1");
    auto div = graph.AddLayer<DivisionLayer>("div");
    auto output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().SetTensorInfo(info0);
    input1->GetOutputSlot().SetTensorInfo(info1);
    div->GetOutputSlot().SetTensorInfo(outputInfo);

    input0->GetOutputSlot().Connect(div->GetInputSlot(0));
    input1->GetOutputSlot().Connect(div->GetInputSlot(1));
    div->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<DivisionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<DivisionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:div-0");
    CHECK(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    CHECK((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 4, 5 })));
    CHECK((addedReshapeTensorInfo.GetDataType() == DataType::QAsymmS8));
}

TEST_CASE("AddBroadcastReshapeLayerMultiplicationTest")
{
    Graph graph;
    const TensorInfo info0({ 3, 5 }, DataType::QAsymmU8);
    const TensorInfo info1({ 1, 2, 3, 5 }, DataType::QAsymmU8);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::QAsymmU8);

    auto input0 = graph.AddLayer<InputLayer>(0, "input0");
    auto input1 = graph.AddLayer<InputLayer>(1, "input1");
    auto mul = graph.AddLayer<MultiplicationLayer>("mul");
    auto output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().SetTensorInfo(info0);
    input1->GetOutputSlot().SetTensorInfo(info1);
    mul->GetOutputSlot().SetTensorInfo(outputInfo);

    input0->GetOutputSlot().Connect(mul->GetInputSlot(0));
    input1->GetOutputSlot().Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:mul-0");
    CHECK(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    CHECK((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 3, 5 })));
    CHECK((addedReshapeTensorInfo.GetDataType() == DataType::QAsymmU8));
}

TEST_CASE("AddNoBroadcastReshapeLayerTest")
{
    Graph graph;
    const TensorInfo info0({ 1, 1, 1, 1 }, DataType::QAsymmU8);
    const TensorInfo info1({ 1, 2, 3, 5 }, DataType::QAsymmU8);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::QAsymmU8);

    auto input0 = graph.AddLayer<InputLayer>(0, "input0");
    auto input1 = graph.AddLayer<InputLayer>(1, "input1");
    auto mul = graph.AddLayer<MultiplicationLayer>("mul");
    auto output = graph.AddLayer<OutputLayer>(0, "output");
    input0->GetOutputSlot().SetTensorInfo(info0);
    input1->GetOutputSlot().SetTensorInfo(info1);
    mul->GetOutputSlot().SetTensorInfo(outputInfo);

    input0->GetOutputSlot().Connect(mul->GetInputSlot(0));
    input1->GetOutputSlot().Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has not been added to the graph
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:mul-0");
    CHECK(!reshapeLayer);
}

TEST_CASE("ReshapeParentConstLayerTest")
{
    Graph graph;
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::QAsymmU8);
    const TensorInfo info1({ 5 }, DataType::QAsymmU8, 0.0f, 0, true);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::QAsymmU8);

    auto input = graph.AddLayer<InputLayer>(0, "input");
    auto constant = graph.AddLayer<ConstantLayer>("constant");
    auto mul = graph.AddLayer<MultiplicationLayer>("mul");
    auto output = graph.AddLayer<OutputLayer>(0, "output");

    uint8_t tensor[] = { 1, 1, 1, 1, 1 };

    constant->m_LayerOutput = std::make_unique<ScopedTensorHandle>(ConstTensor(info1, &tensor));

    input->GetOutputSlot().SetTensorInfo(info0);
    constant->GetOutputSlot().SetTensorInfo(info1);
    mul->GetOutputSlot().SetTensorInfo(outputInfo);

    input->GetOutputSlot().Connect(mul->GetInputSlot(0));
    constant->GetOutputSlot().Connect(mul->GetInputSlot(1));
    mul->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ConstantLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has not been added to the graph
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ConstantLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    TensorShape expectedShape = TensorShape{ 1, 1, 1, 5 };
    CHECK(constant->m_LayerOutput.get()->GetTensorInfo().GetShape() == expectedShape);

    CHECK(constant->m_LayerOutput.get()->GetTensorInfo().GetNumDimensions() == info0.GetNumDimensions());

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:mul-0");
    CHECK(!reshapeLayer);
}

TEST_CASE("ReshapeParentConstAddLayerMultipleConnectionsTest")
{
    // In this test case we recreate the situation where an Addition layer has
    // a constant second term, e.g. [1,512] + [1]. The AddBroadcastReshapeLayer
    // should modify the constant tensor info to match the number of dimensions.
    // However, if this constant term is being reused elsewhere then we shouldn't
    // modify it. Instead we insert a resize layer.

    // What we'll do is have two sequential add layers both using the same const tensor.
    Graph graph;
    const TensorInfo inputInfo({ 1, 512 }, DataType::Float32);
    const TensorInfo constantTermInfo({ 1 }, DataType::Float32, 0.0f, 0, true);
    const TensorInfo outputInfo({ 1, 512 }, DataType::Float32);

    auto input = graph.AddLayer<InputLayer>(0, "input");
    auto constant = graph.AddLayer<ConstantLayer>("constant");
    auto add1 = graph.AddLayer<AdditionLayer>("add1");
    auto add2 = graph.AddLayer<AdditionLayer>("add2");
    auto output = graph.AddLayer<OutputLayer>(0, "output");

    input->GetOutputSlot().SetTensorInfo(inputInfo);
    constant->GetOutputSlot().SetTensorInfo(constantTermInfo);
    float tensor[] = { 2.0f };
    constant->m_LayerOutput = std::make_unique<ScopedTensorHandle>(ConstTensor(constantTermInfo, &tensor));
    add1->GetOutputSlot().SetTensorInfo(outputInfo);

    input->GetOutputSlot().Connect(add1->GetInputSlot(0));
    constant->GetOutputSlot().Connect(add1->GetInputSlot(1));
    add1->GetOutputSlot().Connect(add2->GetInputSlot(0));
    add2->GetOutputSlot().Connect(output->GetInputSlot(0));
    // This second connection should prevent the modification of the const output tensor.
    constant->GetOutputSlot().Connect(add2->GetInputSlot(1));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ConstantLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape should have been added before each addition layer.
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ConstantLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Ensure the output shape of the constant hasn't changed.
    CHECK(constant->m_LayerOutput.get()->GetTensorInfo().GetShape() == constantTermInfo.GetShape());
    // There should be two extra reshape layers with appropriate names.
    Layer* const reshapeLayer1 = GetFirstLayerWithName(graph, "Reshape_for:add1-1");
    Layer* const reshapeLayer2 = GetFirstLayerWithName(graph, "Reshape_for:add2-1");
    CHECK(reshapeLayer1);
    CHECK(reshapeLayer2);
}



}
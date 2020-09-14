//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../GraphUtils.hpp"
#include "../TestUtils.hpp"

#include <Optimizer.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Optimizer)
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

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<AdditionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, reshapeLayerName);
    BOOST_TEST(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    BOOST_TEST((addedReshapeTensorInfo.GetShape() == expectedReshapeShape));
    BOOST_TEST((addedReshapeTensorInfo.GetDataType() == expectedDataType));
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayerSimpleTest)
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 1 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, info0, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 1, 1 }),
                                          DataType::Float32);
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayer1DTest)
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 1, 1, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 1, 5 }),
                                          DataType::Float32);
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayer2DTest)
{
    const TensorInfo info0({ 1, 2, 3, 5 }, DataType::Float32);
    const TensorInfo info1({ 3, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 2, 3, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 1, 3, 5 }),
                                          DataType::Float32);
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayer3DTest)
{
    const TensorInfo info0({ 2, 1, 1, 1 }, DataType::Float32);
    const TensorInfo info1({ 3, 4, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 2, 3, 4, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 3, 4, 5 }),
                                          DataType::Float32);
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayer3DMergedTest)
{
    const TensorInfo info0({ 2, 3, 1, 1 }, DataType::Float32);
    const TensorInfo info1({ 3, 4, 5 }, DataType::Float32);
    const TensorInfo outputInfo({ 2, 3, 4, 5 }, DataType::Float32);
    AddBroadcastReshapeLayerOptimizerTest(info0, info1, outputInfo, "Reshape_for:add-1",
                                          TensorShape({ 1, 3, 4, 5 }),
                                          DataType::Float32);
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayerSubtractionTest)
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

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<SubtractionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<SubtractionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:sub-0");
    BOOST_TEST(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    BOOST_TEST((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 1, 5 })));
    BOOST_TEST((addedReshapeTensorInfo.GetDataType() == DataType::Float32));
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayerDivisionTest)
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

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<DivisionLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<DivisionLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:div-0");
    BOOST_TEST(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    BOOST_TEST((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 4, 5 })));
    BOOST_TEST((addedReshapeTensorInfo.GetDataType() == DataType::QAsymmS8));
}

BOOST_AUTO_TEST_CASE(AddBroadcastReshapeLayerMultiplicationTest)
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

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has been added to the graph correctly
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<ReshapeLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:mul-0");
    BOOST_TEST(reshapeLayer);
    auto addedReshapeTensorInfo = reshapeLayer->GetOutputSlot().GetTensorInfo();

    // Tensorshape and the data type are correct
    BOOST_TEST((addedReshapeTensorInfo.GetShape() == TensorShape({ 1, 1, 3, 5 })));
    BOOST_TEST((addedReshapeTensorInfo.GetDataType() == DataType::QAsymmU8));
}

BOOST_AUTO_TEST_CASE(AddNoBroadcastReshapeLayerTest)
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

    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    // Run optimizer
    armnn::Optimizer::Pass(graph, MakeOptimizations(AddBroadcastReshapeLayer()));

    // Broadcast reshape layer has not been added to the graph
    BOOST_TEST(CheckSequence(graph.cbegin(), graph.cend(),
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<InputLayer>,
                             &IsLayerOfType<MultiplicationLayer>,
                             &IsLayerOfType<OutputLayer>));

    Layer* const reshapeLayer = GetFirstLayerWithName(graph, "Reshape_for:mul-0");
    BOOST_TEST(!reshapeLayer);
}

BOOST_AUTO_TEST_SUITE_END()
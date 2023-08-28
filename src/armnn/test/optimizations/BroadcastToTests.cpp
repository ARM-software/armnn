//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"

#include <Network.hpp>
#include <ResolveType.hpp>
#include <armnn/INetwork.hpp>
#include <TestUtils.hpp>
#include <Optimizer.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
    using namespace armnn;
    using namespace armnn::optimizations;

    TEST_CASE("DeleteBroadcastToAfterMulLayer")
    {
        Graph              graph;
        const unsigned int inputShape[]   = {1, 3};
        const unsigned int outputShape[]  = {4, 3};

        //rank of input is 1 and of output is 2
        TensorInfo inputInfo(1, inputShape, DataType::Float32);
        TensorInfo floorInfo(1, inputShape, DataType::Float32);
        TensorInfo outputInfo(2, outputShape, DataType::Float32);

        Layer* input = graph.AddLayer<InputLayer>(0, "input");
        input->GetOutputSlot().SetTensorInfo(inputInfo);

        FloorLayer* floorLayer = graph.AddLayer<FloorLayer>("floor");
        floorLayer->GetOutputSlot().SetTensorInfo(inputInfo);

        BroadcastToDescriptor broadcastToDescriptor({4, 3});
        BroadcastToLayer* broadcastToLayer = graph.AddLayer<BroadcastToLayer>(broadcastToDescriptor, "broadcast_to");
        broadcastToLayer->GetOutputSlot().SetTensorInfo(floorInfo);

        ElementwiseBinaryDescriptor elementwiseBinaryDescriptor(BinaryOperation::Mul);
        ElementwiseBinaryLayer* elementwiseBinaryLayer =
                graph.AddLayer<ElementwiseBinaryLayer>(elementwiseBinaryDescriptor, "multiplication");
        elementwiseBinaryLayer->GetOutputSlot().SetTensorInfo(outputInfo);

        Layer* output = graph.AddLayer<OutputLayer>(0, "output");

        // Connect up layers - input -> floor -> broadcast_to -> multiplication -> output
        input->GetOutputSlot().Connect(floorLayer->GetInputSlot(0));
        floorLayer->GetOutputSlot().Connect(broadcastToLayer->GetInputSlot(0));
        broadcastToLayer->GetOutputSlot().Connect(elementwiseBinaryLayer->GetInputSlot(0));
        elementwiseBinaryLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<FloorLayer>,
                            &IsLayerOfType<BroadcastToLayer>,
                            &IsLayerOfType<ElementwiseBinaryLayer>,
                            &IsLayerOfType<OutputLayer>));

        Optimizer::Pass(graph, MakeOptimizations(BroadcastToOptimizationLayer()));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<FloorLayer>,
                            &IsLayerOfType<ElementwiseBinaryLayer>,
                            &IsLayerOfType<OutputLayer>));
    }

    TEST_CASE("DeleteBroadcastToNullptr")
    {
        Graph              graph;
        const unsigned int inputShape[]   = {1, 3};
        const unsigned int outputShape[]  = {4, 3};

        //rank of input is 1 and of output is 2
        TensorInfo inputInfo(1, inputShape, DataType::Float32);
        TensorInfo outputInfo(2, outputShape, DataType::Float32);

        Layer* input = graph.AddLayer<InputLayer>(0, "input");
        input->GetOutputSlot().SetTensorInfo(inputInfo);

        ElementwiseBinaryDescriptor elementwiseBinaryDescriptor(BinaryOperation::Mul);
        ElementwiseBinaryLayer* elementwiseBinaryLayer =
                graph.AddLayer<ElementwiseBinaryLayer>(elementwiseBinaryDescriptor, "multiplication");
        elementwiseBinaryLayer->GetOutputSlot().SetTensorInfo(outputInfo);

        Layer* output = graph.AddLayer<OutputLayer>(0, "output");

        // Connect up layers - input -> broadcast_to -> multiplication -> output
        input->GetOutputSlot().Connect(elementwiseBinaryLayer->GetInputSlot(0));
        elementwiseBinaryLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ElementwiseBinaryLayer>,
                            &IsLayerOfType<OutputLayer>));

        Optimizer::Pass(graph, MakeOptimizations(BroadcastToOptimizationLayer()));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<ElementwiseBinaryLayer>,
                            &IsLayerOfType<OutputLayer>));
    }

    TEST_CASE("DeleteBroadcastToNotElementWise")
    {
        Graph              graph;
        const unsigned int inputShape[]   = {1, 3};
        const unsigned int outputShape[]  = {4, 3};

        //rank of input is 1 and of output is 2
        TensorInfo inputInfo(1, inputShape, DataType::Float32);
        TensorInfo broadcastToInfo(2, outputShape, DataType::Float32);
        TensorInfo outputInfo(2, outputShape, DataType::Float32);

        Layer* input = graph.AddLayer<InputLayer>(0, "input");
        input->GetOutputSlot().SetTensorInfo(inputInfo);

        BroadcastToDescriptor broadcastToDescriptor({4, 3});
        BroadcastToLayer* broadcastToLayer = graph.AddLayer<BroadcastToLayer>(broadcastToDescriptor, "broadcast_to");
        broadcastToLayer->GetOutputSlot().SetTensorInfo(broadcastToInfo);

        TileDescriptor tileDescriptor({2, 3});
        TileLayer* tileLayer = graph.AddLayer<TileLayer>(tileDescriptor, "tile");
        tileLayer->GetOutputSlot().SetTensorInfo(outputInfo);

        Layer* output = graph.AddLayer<OutputLayer>(0, "output");

        // Connect up layers - input -> broadcast_to -> tile -> output
        input->GetOutputSlot().Connect(broadcastToLayer->GetInputSlot(0));
        broadcastToLayer->GetOutputSlot().Connect(tileLayer->GetInputSlot(0));
        tileLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<BroadcastToLayer>,
                            &IsLayerOfType<TileLayer>,
                            &IsLayerOfType<OutputLayer>));

        Optimizer::Pass(graph, MakeOptimizations(BroadcastToOptimizationLayer()));

        CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                            &IsLayerOfType<InputLayer>,
                            &IsLayerOfType<BroadcastToLayer>,
                            &IsLayerOfType<TileLayer>,
                            &IsLayerOfType<OutputLayer>));
    }
}

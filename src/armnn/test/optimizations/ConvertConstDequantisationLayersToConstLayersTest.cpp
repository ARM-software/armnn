//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"
#include <Network.hpp>
#include <TestUtils.hpp>
#include <doctest/doctest.h>
#include <armnn/backends/TensorHandle.hpp>
#include <Optimizer.hpp>

TEST_SUITE("Optimizer")
{
using namespace armnn;
using namespace armnn::optimizations;

TEST_CASE("ConvertConstFloat16DequantizeToConstFloat32")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float16, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    ConstantLayer *constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    DequantizeLayer *dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    OutputLayer *output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> dequantize -> output
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkConstantFloat16 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::Float16);
    };
    auto checkConstantFloat32 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::Float32);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat16,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(ConvertConstDequantisationLayersToConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat32,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("ConvertConstInt8DequantizeToConstFloat32")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::QAsymmS8, 1.0, 0, true);
    const TensorInfo outputDequantizeInfo(4, shape, DataType::Float32, 1.0, 0, true);

    ConstantLayer *constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<int8_t> constantValues(constTensorInfo.GetNumElements(), 5);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    DequantizeLayer *dequantizeLayer = graph.AddLayer<DequantizeLayer>("dequantize");
    dequantizeLayer->GetOutputSlot().SetTensorInfo(outputDequantizeInfo);

    OutputLayer *output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> dequantize -> output
    constantLayer->GetOutputSlot().Connect(dequantizeLayer->GetInputSlot(0));
    dequantizeLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkConstantQAsymmS8 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::QAsymmS8);
    };
    auto checkConstantFloat32 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::Float32);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantQAsymmS8,
                        &IsLayerOfType<DequantizeLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(ConvertConstDequantisationLayersToConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat32,
                        &IsLayerOfType<OutputLayer>));
}
}

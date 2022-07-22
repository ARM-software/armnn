//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <LayersFwd.hpp>
#include <Network.hpp>
#include <NetworkUtils.hpp>
#include <Optimizer.hpp>
#include <TestUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn;
using namespace armnn::optimizations;

TEST_CASE("FuseConvertFp32Fp16intoConst")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float32, 1.0, 0, true);
    const TensorInfo outputConvertInfo(4, shape, DataType::BFloat16, 1.0, 0, true);

    ConstantLayer* constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 3.1416f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    ConvertFp32ToBf16Layer* convertLayer = graph.AddLayer<ConvertFp32ToBf16Layer>("convert");
    convertLayer->GetOutputSlot().SetTensorInfo(outputConvertInfo);

    OutputLayer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> convert -> output
    constantLayer->GetOutputSlot().Connect(convertLayer->GetInputSlot(0));
    convertLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkConstantFloat32 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::Float32);
    };
    auto checkConstantBFloat16 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::BFloat16);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantFloat32,
                        &IsLayerOfType<ConvertFp32ToBf16Layer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FuseConversionLayersIntoConstLayers()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        checkConstantBFloat16,
                        &IsLayerOfType<OutputLayer>));
}

TEST_CASE("RevertConstantWeightsToFP32")
{
    Graph graph;
    const unsigned int shape[] = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float32, 1.0, 0, true);
    const TensorInfo outputConvertInfo(4, shape, DataType::BFloat16, 1.0, 0, true);

    TensorInfo inputInfo(4, shape, DataType::Float32);
    auto* input = graph.AddLayer<InputLayer>(0, "input0");
    input->GetOutputSlot().SetTensorInfo(inputInfo);

    auto* constantLayer = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 3.1416f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constantLayer->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constantLayer->GetOutputSlot().SetTensorInfo(constTensorInfo);

    ConvertFp32ToBf16Layer* convertLayerInputs = graph.AddLayer<ConvertFp32ToBf16Layer>("convert");
    convertLayerInputs->GetOutputSlot().SetTensorInfo(outputConvertInfo);
    ConvertFp32ToBf16Layer* convertLayerWeights = graph.AddLayer<ConvertFp32ToBf16Layer>("convert2");
    convertLayerWeights->GetOutputSlot().SetTensorInfo(outputConvertInfo);
    ConvertFp32ToBf16Layer* convertLayerBiases = graph.AddLayer<ConvertFp32ToBf16Layer>("convert3");
    convertLayerBiases->GetOutputSlot().SetTensorInfo(outputConvertInfo);

    auto* biases  = graph.AddLayer<armnn::ConstantLayer>("Biases");
    biases->m_LayerOutput  = std::make_unique<armnn::ScopedTensorHandle>(constTensor);
    biases->GetOutputSlot().SetTensorInfo(constTensorInfo);

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = true;
    auto* conv = graph.AddLayer<armnn::Convolution2dLayer>(descriptor, "conv2d");
    const armnn::TensorInfo infoFP32({ 2, 3, 8, 1 }, armnn::DataType::Float32);
    conv->GetOutputSlot().SetTensorInfo(infoFP32);

    auto* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up Input    -> Convert ->
    //            Constant -> Convert -> Conv2d -> Output
    //            Constant -> Convert ->
    input->GetOutputSlot().Connect(convertLayerInputs->GetInputSlot(0));
    constantLayer->GetOutputSlot().Connect(convertLayerWeights->GetInputSlot(0));
    biases->GetOutputSlot().Connect(convertLayerBiases->GetInputSlot(0));

    convertLayerInputs->GetOutputSlot().Connect(conv->GetInputSlot(0));
    convertLayerWeights->GetOutputSlot().Connect(conv->GetInputSlot(1));
    convertLayerBiases->GetOutputSlot().Connect(conv->GetInputSlot(2));

    conv->GetOutputSlot().Connect(output->GetInputSlot(0));

    auto checkConstantFloat32 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::Float32);
    };
    auto checkConstantBFloat16 = [](const armnn::Layer *const layer) -> bool {
        return IsLayerOfType<ConstantLayer>(layer) &&
               (layer->GetDataType() == DataType::BFloat16);
    };

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantFloat32,
                        checkConstantFloat32,
                        &IsLayerOfType<ConvertFp32ToBf16Layer>,
                        &IsLayerOfType<ConvertFp32ToBf16Layer>,
                        &IsLayerOfType<ConvertFp32ToBf16Layer>,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FuseConversionLayersIntoConstLayers()));

    bool revert = RevertConstantWeightsToFP32(conv);

    // Erase unconnected layer as occurs during Topological Sort.
    graph.EraseLayer(convertLayerInputs);

    CHECK(revert);
    CHECK(constantLayer->GetDataType() == DataType::Float32);

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkConstantBFloat16,
                        checkConstantFloat32,
                        &IsLayerOfType<Convolution2dLayer>,
                        &IsLayerOfType<OutputLayer>));
}
}

//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Half.hpp>
#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("ConvertConstantsFloatToHalfTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::Float16);

    // Create const tensor from fp32 data
    unsigned int dims[] = { 4, 1, 1, 1 };
    std::vector<float> floatWeights{ 1.0f, 2.0f, 3.0f, 4.0f };
    armnn::TensorInfo weightsInfo = armnn::TensorInfo(4, dims, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::ConstTensor weights(weightsInfo, floatWeights);

    // Create simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc      = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->GetOutputSlot().SetTensorInfo(info);

    auto weightsLayer = graph.AddLayer<armnn::ConstantLayer>("weights");
    weightsLayer->m_LayerOutput = std::make_unique<armnn::ScopedTensorHandle>(weights);
    weightsLayer->GetOutputSlot().SetTensorInfo(weightsInfo);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    weightsLayer->GetOutputSlot().Connect(fc->GetInputSlot(1));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Check tensor data type before conversion
    CHECK(weightsLayer->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToHalf()));

    // Check tensor data type after conversion
    CHECK(weightsLayer->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Check whether data matches expected fp16 data
    const Half* data = weightsLayer->m_LayerOutput->GetConstTensor<Half>();
    CHECK(data[0] == Half(1.0f));
    CHECK(data[1] == Half(2.0f));
    CHECK(data[2] == Half(3.0f));
    CHECK(data[3] == Half(4.0f));
}


TEST_CASE("ConvertConstantsFloatToHalfTest_constant")
{
    armnn::Graph graph;

    // Create the simple test network with Weights and Biases as inputs to a FullyConnected layer.
    auto input   = graph.AddLayer<armnn::InputLayer>(0, "Input");
    auto weights = graph.AddLayer<armnn::ConstantLayer>("Weights");
    auto biases  = graph.AddLayer<armnn::ConstantLayer>("Biases");

    armnn::FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = true;
    desc.m_ConstantWeights = true;
    auto fcLayer = graph.AddLayer<armnn::FullyConnectedLayer>(desc, "FullyConnected");
    auto output  = graph.AddLayer<armnn::OutputLayer>(1, "Output");

    float expectedWeightsData[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float expectedBiasesData[]  = { 2.0f, 2.0f };

    const armnn::TensorInfo inputInfo  ({ 1, 2, 2, 3 }, armnn::DataType::Float16);
    const armnn::TensorInfo outputInfo ({ 1, 2, 2, 3 }, armnn::DataType::Float16);
    const armnn::TensorInfo weightsInfo({ 4 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 2 }, armnn::DataType::Float32, 0.0f, 0, true);

    // Set the m_LayerOutput for the optimizer to point to.
    armnn::ConstTensor weightsTensor(weightsInfo, &expectedWeightsData);
    armnn::ConstTensor biasesTensor(biasesInfo, &expectedBiasesData);
    weights->m_LayerOutput = std::make_unique<armnn::ScopedTensorHandle>(weightsTensor);
    biases->m_LayerOutput  = std::make_unique<armnn::ScopedTensorHandle>(biasesTensor);

    input->GetOutputSlot().SetTensorInfo(inputInfo);
    weights->GetOutputSlot().SetTensorInfo(weightsInfo);
    biases->GetOutputSlot().SetTensorInfo(biasesInfo);
    fcLayer->GetOutputSlot().SetTensorInfo(outputInfo);

    // Connect up the layers
    input->GetOutputSlot(0).Connect(fcLayer->GetInputSlot(0));
    weights->GetOutputSlot(0).Connect(fcLayer->GetInputSlot(1));
    biases->GetOutputSlot(0).Connect(fcLayer->GetInputSlot(2));
    fcLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Check tensor data type before conversion
    CHECK(5 == graph.GetNumLayers());
    CHECK(weights->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToHalf()));

    // Check tensor data type after conversion
    CHECK(5 == graph.GetNumLayers());
    CHECK(weights->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Check whether weights data matches expected fp16 data
    const Half* data = weights->m_LayerOutput->GetConstTensor<Half>();
    CHECK(data[0] == Half(1.0f));
    CHECK(data[1] == Half(2.0f));
    CHECK(data[2] == Half(3.0f));
    CHECK(data[3] == Half(4.0f));

    // Check whether bias data matches expected fp16 data
    const Half* biasData = biases->m_LayerOutput->GetConstTensor<Half>();
    CHECK(biasData[0] == Half(2.0f));
    CHECK(biasData[1] == Half(2.0f));
}


}
//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("RedirectMembersToConstantInputsFullyConnectedTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo inputInfo  ({ 1, 2, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo ({ 1, 2, 2, 3 }, armnn::DataType::Float32);
    const armnn::TensorInfo weightsInfo({ 4 }, armnn::DataType::Float32, 0.0f, 0, true);
    const armnn::TensorInfo biasesInfo ({ 2 }, armnn::DataType::Float32, 0.0f, 0, true);

    // Check if isConstant is enabled for weights and biases tensor info.
    CHECK(weightsInfo.IsConstant());
    CHECK(biasesInfo.IsConstant());

    armnn::FullyConnectedDescriptor desc;
    desc.m_BiasEnabled = true;
    desc.m_ConstantWeights = false;

    // Create the simple test network with Weights and Biases as inputs to a FullyConnected layer.
    auto input   = graph.AddLayer<armnn::InputLayer>(0, "Input");
    auto weights = graph.AddLayer<armnn::ConstantLayer>("Weights");
    auto biases  = graph.AddLayer<armnn::ConstantLayer>("Biases");
    auto fcLayer = graph.AddLayer<armnn::FullyConnectedLayer>(desc, "FullyConnected");
    auto output  = graph.AddLayer<armnn::OutputLayer>(1, "Output");

    float expectedWeightsData[] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float expectedBiasesData[]  = { 2.0f, 2.0f };

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

    // Member variables should be null before optimization.
    CHECK(fcLayer->m_Weight == nullptr);
    CHECK(fcLayer->m_Bias == nullptr);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(RedirectMembersToConstantInputs()));

    // Check if member variables are not null and shape is set correctly.
    CHECK(fcLayer->m_Weight != nullptr);
    CHECK(fcLayer->m_Bias != nullptr);
    CHECK(fcLayer->m_Weight->GetTensorInfo().GetShape() == weightsInfo.GetShape());
    CHECK(fcLayer->m_Bias->GetTensorInfo().GetShape() == biasesInfo.GetShape());

    // Check whether data matches expected float data
    const float* weightsData = fcLayer->m_Weight->GetConstTensor<float>();
    CHECK(weightsData[0] == expectedWeightsData[0]);
    CHECK(weightsData[1] == expectedWeightsData[1]);
    CHECK(weightsData[2] == expectedWeightsData[2]);
    CHECK(weightsData[3] == expectedWeightsData[3]);

    const float* biasesData = fcLayer->m_Bias->GetConstTensor<float>();
    CHECK(biasesData[0] == expectedBiasesData[0]);
    CHECK(biasesData[1] == expectedBiasesData[1]);
}

}
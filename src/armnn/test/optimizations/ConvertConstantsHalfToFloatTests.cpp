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

TEST_CASE("ConvertConstantsHalfToFloatTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::Float32);

    // Create the half precision input data
    unsigned int dims[] = { 4, 1, 1, 1 };
    std::vector<float> convWeightsData{ 1.f, 2.f, 3.f, 4.f };
    std::vector<uint16_t> halfWeights(4);
    armnnUtils::FloatingPointConverter::ConvertFloat32To16(convWeightsData.data(), convWeightsData.size(),
                                                           halfWeights.data());
    armnn::TensorInfo weightInfo = armnn::TensorInfo(4, dims, armnn::DataType::Float16, 0.0f, 0, true);
    armnn::ConstTensor weights(weightInfo, halfWeights);

    //Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc      = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->GetOutputSlot().SetTensorInfo(info);

    auto weightsLayer = graph.AddLayer<armnn::ConstantLayer>("weights");
    weightsLayer->m_LayerOutput = std::make_unique<armnn::ScopedTensorHandle>(weights);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightInfo);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    //Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    weightsLayer->GetOutputSlot().Connect(fc->GetInputSlot(1));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    //Test the tensor info is correct.
    CHECK(weightsLayer->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsHalfToFloat()));

    //Test the tensor info is correct.
    CHECK(weightsLayer->m_LayerOutput->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Now test the data matches float32 data
    const float* data = weightsLayer->m_LayerOutput->GetConstTensor<float>();
    CHECK(1.0f == data[0]);
    CHECK(2.0f == data[1]);
    CHECK(3.0f == data[2]);
    CHECK(4.0f == data[3]);
}

}
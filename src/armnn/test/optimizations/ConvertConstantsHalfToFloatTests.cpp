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
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float16, 0.0f, 0, true), halfWeights);

    //Create the simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc      = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->m_Weight = std::make_unique<armnn::ScopedTensorHandle>(weights);
    fc->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    //Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    //Test the tensor info is correct.
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsHalfToFloat()));

    //Test the tensor info is correct.
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Now test the data matches float32 data
    const float* data = fc->m_Weight->GetConstTensor<float>();
    CHECK(1.0f == data[0]);
    CHECK(2.0f == data[1]);
    CHECK(3.0f == data[2]);
    CHECK(4.0f == data[3]);
}

}
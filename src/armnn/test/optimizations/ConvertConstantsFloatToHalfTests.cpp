//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>
#include <Half.hpp>

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
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32, 0.0f, 0, true), floatWeights);

    // Create simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc      = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->m_Weight = std::make_unique<armnn::ScopedTensorHandle>(weights);
    fc->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Check tensor data type before conversion
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToHalf()));

    // Check tensor data type after conversion
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Check whether data matches expected fp16 data
    const Half* data = fc->m_Weight->GetConstTensor<Half>();
    CHECK(data[0] == Half(1.0f));
    CHECK(data[1] == Half(2.0f));
    CHECK(data[2] == Half(3.0f));
    CHECK(data[3] == Half(4.0f));
}

}
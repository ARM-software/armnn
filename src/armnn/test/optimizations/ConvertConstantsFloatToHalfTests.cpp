//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../TestUtils.hpp"

#include <Optimizer.hpp>
#include <Half.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(Optimizer)
using namespace armnn::optimizations;

BOOST_AUTO_TEST_CASE(ConvertConstantsFloatToHalfTest)
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::Float16);

    // Create const tensor from fp32 data
    unsigned int dims[] = { 4, 1, 1, 1 };
    std::vector<float> floatWeights{ 1.0f, 2.0f, 3.0f, 4.0f };
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::Float32), floatWeights);

    // Create simple test network
    auto input = graph.AddLayer<armnn::InputLayer>(0, "input");
    input->GetOutputSlot().SetTensorInfo(info);

    auto fc      = graph.AddLayer<armnn::FullyConnectedLayer>(armnn::FullyConnectedDescriptor(), "fc");
    fc->m_Weight = std::make_unique<armnn::ScopedCpuTensorHandle>(weights);
    fc->GetOutputSlot().SetTensorInfo(info);

    auto output = graph.AddLayer<armnn::OutputLayer>(1, "output");

    // Connect up the layers
    input->GetOutputSlot().Connect(fc->GetInputSlot(0));
    fc->GetOutputSlot().Connect(output->GetInputSlot(0));

    // Check tensor data type before conversion
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToHalf()));

    // Check tensor data type after conversion
    BOOST_CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float16);

    // Check whether data matches expected fp16 data
    Half* data = fc->m_Weight->GetTensor<Half>();
    BOOST_CHECK(data[0] == Half(1.0f));
    BOOST_CHECK(data[1] == Half(2.0f));
    BOOST_CHECK(data[2] == Half(3.0f));
    BOOST_CHECK(data[3] == Half(4.0f));
}

BOOST_AUTO_TEST_SUITE_END()
//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <BFloat16.hpp>
#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("ConvertConstantsFloatToBFloatTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::BFloat16);

    // Create const tensor from fp32 data
    unsigned int dims[] = { 4, 2, 1, 1 };
    std::vector<float> floatWeights{ 0.0f, -1.0f,
                                     3.8f, // 0x40733333 Round down
                                     3.1055E+29f, // 0x707ADC3C Round up
                                     9.149516E-10f, // 0x307B7FFF Round down
                                    -3.8f, // 0xC0733333 Round down
                                    -3.1055E+29f, // 0xF07ADC3C Round up
                                    -9.149516E-10f // 0xB07B7FFF Round down
                                   };
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
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsFloatToBFloat()));

    // Check tensor data type after conversion
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::BFloat16);

    // Check whether data matches expected Bf16 data
    const BFloat16* data = fc->m_Weight->GetConstTensor<BFloat16>();
    CHECK(data[0] == BFloat16(0.0f));
    CHECK(data[1] == BFloat16(-1.0f));
    CHECK(data[2] == BFloat16(3.796875f)); // 0x4073
    CHECK(data[3] == BFloat16(3.1072295E29f)); // 0x707B
    CHECK(data[4] == BFloat16(9.131327E-10f)); // 0x307B
    CHECK(data[5] == BFloat16(-3.796875f)); // 0xC073
    CHECK(data[6] == BFloat16(-3.1072295E29f)); // 0xF07B
    CHECK(data[7] == BFloat16(-9.131327E-10f)); // 0xB07B
}

TEST_CASE("ConvertConstantsBFloatToFloatTest")
{
    armnn::Graph graph;

    const armnn::TensorInfo info({ 1, 1, 1, 2 }, armnn::DataType::Float32);

    // Create the BFloat16 precision input data
    unsigned int dims[] = { 4, 2, 1, 1 };
    std::vector<float> convWeightsData{ 0.f, -1.f,
                                        3.796875f, // 0x4073
                                        3.1072295E29f, // 0x707B
                                        9.131327E-10f, // 0x307B
                                       -3.796875f, // 0xC073
                                       -3.1072295E29f, // 0xF07B
                                       -9.131327E-10f // 0xB07B
                                       };
    std::vector<uint16_t> bfWeights(8);
    armnnUtils::FloatingPointConverter::ConvertFloat32ToBFloat16(convWeightsData.data(), convWeightsData.size(),
                                                                 bfWeights.data());
    armnn::ConstTensor weights(armnn::TensorInfo(4, dims, armnn::DataType::BFloat16, 0.0f, 0, true), bfWeights);

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
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::BFloat16);

    // Run the optimizer
    armnn::Optimizer::Pass(graph, armnn::MakeOptimizations(ConvertConstantsBFloatToFloat()));

    //Test the tensor info is correct.
    CHECK(fc->m_Weight->GetTensorInfo().GetDataType() == armnn::DataType::Float32);

    // Now test the data matches float32 data
    const float* data = fc->m_Weight->GetConstTensor<float>();
    CHECK(data[0] == 0.0f);
    CHECK(data[1] == -1.0f);
    CHECK(data[2] == 3.796875f);
    CHECK(data[3] == 3.1072295E29f);
    CHECK(data[4] == 9.131327E-10f);
    CHECK(data[5] == -3.796875f);
    CHECK(data[6] == -3.1072295E29f);
    CHECK(data[7] == -9.131327E-10f);
}

}
//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LayersFwd.hpp"
#include <Network.hpp>
#include <doctest/doctest.h>
#include <Optimizer.hpp>
#include <TestUtils.hpp>

TEST_SUITE("Optimizer")
{
using namespace armnn;
using namespace armnn::optimizations;

TEST_CASE("ConvertConstPermuteToConst")
{
    Graph graph;
    const unsigned int shape[]  = {1, 2, 2, 3};

    const TensorInfo constTensorInfo(4, shape, DataType::Float32, 1.0, 0, true);

    ConstantLayer* constant = graph.AddLayer<ConstantLayer>("constant");
    std::vector<float> constantValues(constTensorInfo.GetNumElements(), 4.5f);
    ConstTensor constTensor(constTensorInfo, constantValues.data());
    constant->m_LayerOutput = std::make_shared<ScopedTensorHandle>(constTensor);
    constant->GetOutputSlot().SetTensorInfo(constTensorInfo);

    PermuteDescriptor desc({ 0, 2, 3, 1 });
    PermuteLayer* permuteLayer = graph.AddLayer<PermuteLayer>(desc, "permute");
    TensorInfo infoPermuted = armnnUtils::Permuted(constTensorInfo, { 0, 2, 3, 1 });
    permuteLayer->GetOutputSlot().SetTensorInfo(infoPermuted);

    OutputLayer* output = graph.AddLayer<OutputLayer>(0, "output");

    // Connect up constant -> permute -> output
    constant->GetOutputSlot().Connect(permuteLayer->GetInputSlot(0));
    permuteLayer->GetOutputSlot().Connect(output->GetInputSlot(0));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<PermuteLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Optimizer::Pass(graph, MakeOptimizations(FusePermuteIntoConstLayer()));

    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<ConstantLayer>,
                        &IsLayerOfType<OutputLayer>));

    TensorShape tensorShape = constant->GetOutputSlot(0).GetTensorInfo().GetShape();
    CHECK(tensorShape[0] == shape[0]);
    CHECK(tensorShape[1] == shape[3]);
    CHECK(tensorShape[2] == shape[1]);
    CHECK(tensorShape[3] == shape[2]);

}

}

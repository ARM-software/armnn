//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <TestUtils.hpp>

#include <Optimizer.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("Optimizer")
{
using namespace armnn::optimizations;

TEST_CASE("OptimizeInverseConversionsTest")
{
    armnn::Graph graph;

    auto output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    graph.InsertNewLayer<armnn::InputLayer>(output->GetInputSlot(0), 0, "input");

    // Fp32ToFp16 conversion followed by an inverse Fp16ToFp32 conversion
    graph.InsertNewLayer<armnn::ConvertFp32ToFp16Layer>(output->GetInputSlot(0), "convert1");
    graph.InsertNewLayer<armnn::ConvertFp16ToFp32Layer>(output->GetInputSlot(0), "convert2");

    graph.InsertNewLayer<armnn::Convolution2dLayer>(output->GetInputSlot(0), Convolution2dDescriptor(), "conv");

    // Fp16ToFp32 conversion followed by an inverse Fp32ToFp16 conversion
    graph.InsertNewLayer<armnn::ConvertFp16ToFp32Layer>(output->GetInputSlot(0), "convert3");
    graph.InsertNewLayer<armnn::ConvertFp32ToFp16Layer>(output->GetInputSlot(0), "convert4");

    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>, &IsLayerOfType<armnn::Convolution2dLayer>,
                             &IsLayerOfType<armnn::ConvertFp16ToFp32Layer>,
                             &IsLayerOfType<armnn::ConvertFp32ToFp16Layer>, &IsLayerOfType<armnn::OutputLayer>));

    armnn::Optimizer::Pass(
        graph, armnn::MakeOptimizations(OptimizeInverseConversionsFp16(), OptimizeInverseConversionsFp32()));

    // Check that all consecutive inverse conversions are removed
    CHECK(CheckSequence(graph.cbegin(), graph.cend(), &IsLayerOfType<armnn::InputLayer>,
                             &IsLayerOfType<armnn::Convolution2dLayer>, &IsLayerOfType<armnn::OutputLayer>));
}

}
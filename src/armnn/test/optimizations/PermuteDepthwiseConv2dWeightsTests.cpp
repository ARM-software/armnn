//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../armnnTestUtils/GraphUtils.hpp"
#include "../armnnTestUtils/TestUtils.hpp"

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{
#if defined(ARMCOMPUTENEON_ENABLED) || defined(ARMCOMPUTECL_ENABLED)
armnn::INetworkPtr CreateSimpleDepthwiseConv2dNetwork(const armnn::TensorInfo& inputTensorInfo,
                                                      const armnn::TensorInfo& outputTensorInfo,
                                                      const armnn::TensorInfo& weightsTensorInfo,
                                                      armnn::DepthwiseConvolution2dDescriptor descriptor)
{
    armnn::INetworkPtr network(armnn::INetwork::Create());

    armnn::IConnectableLayer* inputLayer  = network->AddInputLayer(0, "input");
    armnn::IConnectableLayer* weightsInputLayer   = network->AddInputLayer(1, "weights_input");
    armnn::IConnectableLayer* depthwiseLayer = network->AddDepthwiseConvolution2dLayer(descriptor, "depthwise_conv2d");
    armnn::IConnectableLayer* outputLayer = network->AddOutputLayer(0, "output");

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    weightsInputLayer->GetOutputSlot(0).SetTensorInfo(weightsTensorInfo);
    depthwiseLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    inputLayer->GetOutputSlot(0).Connect(depthwiseLayer->GetInputSlot(0));
    weightsInputLayer->GetOutputSlot(0).Connect(depthwiseLayer->GetInputSlot(1));
    depthwiseLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    return network;
}

void PermuteDepthwiseConv2dWeightsTestRunner(INetworkPtr& network,
                                             const TensorShape& outputShape,
                                             Compute backendId)
{
    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {backendId}, run->GetDeviceSpec());

    Graph& graph = GetGraphForTesting(optNet.get());

    CHECK(graph.GetNumLayers() == 5);
    CHECK(CheckSequence(graph.cbegin(),
                        graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<InputLayer>,
                        &IsLayerOfType<PermuteLayer>,
                        &IsLayerOfType<DepthwiseConvolution2dLayer>,
                        &IsLayerOfType<OutputLayer>));

    armnn::Layer* const permuteLayer = GetFirstLayerWithName(graph, "permute_layer");
    CHECK(permuteLayer);

    // Swap original shape to compare with new shape.
    unsigned int weightsShape[] = {outputShape[0], outputShape[1], outputShape[2], outputShape[3]};

    // Tensorshape and the data type are correct
    // [ 1, H, W, I*M] --> [ 1, I * M, H, W ]
    TensorShape newShape = permuteLayer->GetOutputSlot().GetTensorInfo().GetShape();
    CHECK((newShape[0] == weightsShape[0]));
    CHECK((newShape[1] == weightsShape[3]));
    CHECK((newShape[2] == weightsShape[1]));
    CHECK((newShape[3] == weightsShape[2]));
}

void PermuteDepthwiseConv2dWeightsTest(Compute backendId)
{
    armnn::TensorInfo inputTensorInfo({ 1, 1, 2, 3 }, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 2 }, armnn::DataType::Float32);
    armnn::TensorInfo weightsTensorInfo({ 2, 6 }, armnn::DataType::Float32);

    DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = false;

    armnn::INetworkPtr network = CreateSimpleDepthwiseConv2dNetwork(inputTensorInfo,
                                                                    outputTensorInfo,
                                                                    weightsTensorInfo,
                                                                    descriptor);

    PermuteDepthwiseConv2dWeightsTestRunner(network,
                                            weightsTensorInfo.GetShape(),
                                            backendId);
}
#endif
}

#if defined(ARMCOMPUTECL_ENABLED)
TEST_SUITE("Optimizer_PermuteDepthwiseConv2dWeightsGpuAcc")
{
TEST_CASE("PermuteDepthwiseConv2dWeightsGpuAccTest")
{
    PermuteDepthwiseConv2dWeightsTest(Compute::GpuAcc);
}
}
#endif

#if defined(ARMCOMPUTENEON_ENABLED)
TEST_SUITE("Optimizer_PermuteDepthwiseConv2dWeightsCpuAcc")
{
TEST_CASE("PermuteDepthwiseConv2dWeightsCpuAccTest")
{
    PermuteDepthwiseConv2dWeightsTest(Compute::CpuAcc);
}
}
#endif

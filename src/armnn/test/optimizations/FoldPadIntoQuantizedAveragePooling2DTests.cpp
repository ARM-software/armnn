//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <GraphUtils.hpp>
#include <TestUtils.hpp>

#include <armnn/INetwork.hpp>

#include <doctest/doctest.h>

using namespace armnn;

namespace
{
#if defined(ARMNNREF_ENABLED)||defined(ARMCOMPUTECL_ENABLED)
void FoldPadIntoQuantizedAvgPoolTest(Compute backendId)
{
    // Create a network
    INetworkPtr network = INetwork::Create();

    const unsigned int inputShape[]  = {1, 2, 2, 3};
    const unsigned int paddedShape[] = {1, 4, 4, 3};
    const unsigned int outputShape[] = {1, 2, 2, 3};

    TensorInfo inputInfo(4, inputShape, DataType::QAsymmU8, 1.0f, 0.0f);
    TensorInfo paddedInfo(4, paddedShape, DataType::QAsymmU8, 1.0f, 0.0f);
    TensorInfo outputInfo(4, outputShape, DataType::QAsymmU8, 1.0f, 0.0f);

    IConnectableLayer* input = network->AddInputLayer(0, "input");
    input->GetOutputSlot(0).SetTensorInfo(inputInfo);

    PadDescriptor padDescriptor({{0, 0},
                                 {1, 1},
                                 {1, 1},
                                 {0, 0}});

    IConnectableLayer* padLayer = network->AddPadLayer(padDescriptor, "pad");
    padLayer->GetOutputSlot(0).SetTensorInfo(paddedInfo);

    Pooling2dDescriptor pooling2dDescriptor;
    pooling2dDescriptor.m_PoolType   = PoolingAlgorithm::Average;
    pooling2dDescriptor.m_PoolWidth  = 3;
    pooling2dDescriptor.m_PoolHeight = 3;
    pooling2dDescriptor.m_StrideX    = 1;
    pooling2dDescriptor.m_StrideY    = 1;
    pooling2dDescriptor.m_DataLayout = DataLayout::NHWC;

    IConnectableLayer* pool2dLayer = network->AddPooling2dLayer(pooling2dDescriptor, "pool2d");
    pool2dLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    IConnectableLayer* output = network->AddOutputLayer(0, "output");

    // Connect up layers - input -> pad -> pool2d -> output
    input->GetOutputSlot(0).Connect(padLayer->GetInputSlot(0));
    padLayer->GetOutputSlot(0).Connect(pool2dLayer->GetInputSlot(0));
    pool2dLayer->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    // Create ArmNN runtime
    IRuntimePtr run = IRuntime::Create(IRuntime::CreationOptions());

    // Optimise ArmNN network
    IOptimizedNetworkPtr optNet = Optimize(*network, {backendId}, run->GetDeviceSpec());

    auto checkPadFoldedIntoPool2d = [&](const Layer* const layer) {
        if (!IsLayerOfType<Pooling2dLayer>(layer) || (layer->GetNameStr() != "folded-pad-into-pool2d"))
        {
            return false;
        }

        const auto                pool2dLayer       = static_cast<const Pooling2dLayer*>(layer);
        const Pooling2dDescriptor pool2dLayerParams = pool2dLayer->GetParameters();

        Pooling2dDescriptor pool2dLayerParamsNoPad = pool2dLayerParams;
        pool2dLayerParamsNoPad.m_PadLeft       = 0;
        pool2dLayerParamsNoPad.m_PadRight      = 0;
        pool2dLayerParamsNoPad.m_PadTop        = 0;
        pool2dLayerParamsNoPad.m_PadBottom     = 0;
        // If we fold then PaddingMethod will be set to Ignore. The original will be Exclude.
        pool2dLayerParamsNoPad.m_PaddingMethod = PaddingMethod::Exclude;

        return (pool2dLayerParamsNoPad == pooling2dDescriptor) && (pool2dLayerParams.m_PadLeft == 1) &&
            (pool2dLayerParams.m_PadRight == 1) && (pool2dLayerParams.m_PadTop == 1) &&
            (pool2dLayerParams.m_PadBottom == 1) && (pool2dLayerParams.m_PaddingMethod == PaddingMethod::IgnoreValue);
    };

    Graph& graph = GetGraphForTesting(optNet.get());
    CHECK(CheckSequence(graph.cbegin(), graph.cend(),
                        &IsLayerOfType<InputLayer>,
                        checkPadFoldedIntoPool2d,
                        &IsLayerOfType<OutputLayer>));
}
#endif
}


#if defined(ARMNNREF_ENABLED)
TEST_SUITE("Optimizer_FoldPadIntoQuantizedAvgPoolCpuRef")
{
TEST_CASE("FoldPadIntoQuantizedAvgPoolCpuRefTest")
{
    FoldPadIntoQuantizedAvgPoolTest(Compute::CpuRef);
}
}
#endif

#if defined(ARMCOMPUTECL_ENABLED)
TEST_SUITE("Optimizer_FoldPadIntoQuantizedAvgPoolGpuAcc")
{
TEST_CASE("FoldPadIntoQuantizedAvgPoolGpuAccTest")
{
    FoldPadIntoQuantizedAvgPoolTest(Compute::GpuAcc);
}
}
#endif

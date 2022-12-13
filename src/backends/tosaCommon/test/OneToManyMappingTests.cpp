//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AvgPool2DIgnoreValueChecker.hpp"
#include <armnn/IRuntime.hpp>

using namespace armnn;
using namespace tosa;

TEST_SUITE("TosaOperatorMappingOneToManyTests")
{
TEST_CASE("GetTosaMapping_AvgPool2DIgnoreValueLayer")
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape         = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> intermediateShape  = {{ 1, 1, 6, 6 }};
    std::vector<std::vector<int32_t>> outputShape        = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor);
    VerifyAvgPool2DIgnoreValue(basicBlock,
                               inputShape,
                               outputShape,
                               intermediateShape,
                               descriptor);
}

TEST_CASE("GetTosaMappingFromLayer_AvgPool2DIgnoreValueLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;

    IConnectableLayer* input0  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool    = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape         = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> intermediateShape  = {{ 1, 1, 6, 6 }};
    std::vector<std::vector<int32_t>> outputShape        = {{ 1, 1, 3, 3 }};

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool));
    VerifyAvgPool2DIgnoreValue(basicBlock,
                              inputShape,
                              outputShape,
                              intermediateShape,
                              descriptor);
}
}
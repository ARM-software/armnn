//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

TEST_SUITE("TosaOperatorMappingOneToOneTests")
{
TEST_CASE("GetTosaMapping_AdditionLayer")
{
    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 2, 4, 2 }, { 1, 2, 4, 2 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 2, 4, 2 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::Addition, {&info, &info}, {&info}, BaseDescriptor(), false);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_ADD, Attribute_NONE, BaseDescriptor(), LayerType::Addition);
}

TEST_CASE("GetTosaMappingFromLayer_AdditionLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0, "input0");
    IConnectableLayer* input1 = net->AddInputLayer(1, "input1");
    IConnectableLayer* add    = net->AddAdditionLayer("add");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);

    input0->GetOutputSlot(0).SetTensorInfo(info);
    input1->GetOutputSlot(0).SetTensorInfo(info);
    add->GetOutputSlot(0).SetTensorInfo(info);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 2, 4, 2 }, { 1, 2, 4, 2 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 2, 4, 2 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(add), false);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_ADD, Attribute_NONE, BaseDescriptor(), LayerType::Addition);
}

TEST_CASE("GetTosaMapping_MaxPool2DLayer")
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor, false);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_MAX_POOL2D, Attribute_PoolAttribute, descriptor, LayerType::Pooling2d);
}

TEST_CASE("GetTosaMappingFromLayer_MaxPool2DLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    IConnectableLayer* input0  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool    = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool), false);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_MAX_POOL2D, Attribute_PoolAttribute, descriptor, LayerType::Pooling2d);
}

TEST_CASE("GetTosaMapping_AvgPool2DLayer")
{
    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor, false);
    AssertTosaOneToOneMappingBasicBlock(basicBlock,
                                        inputShape,
                                        outputShape,
                                        Op_AVG_POOL2D,
                                        Attribute_PoolAttribute,
                                        descriptor,
                                        LayerType::Pooling2d);
}

TEST_CASE("GetTosaMappingFromLayer_AvgPool2DLayer")
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
    descriptor.m_PaddingMethod = armnn::PaddingMethod::Exclude;

    IConnectableLayer* input0  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool    = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    armnn::TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    armnn::TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool), false);
    AssertTosaOneToOneMappingBasicBlock(basicBlock,
                                        inputShape,
                                        outputShape,
                                        Op_AVG_POOL2D,
                                        Attribute_PoolAttribute,
                                        descriptor,
                                        LayerType::Pooling2d);
}

TEST_CASE("GetTosaMapping_Unimplemented")
{
    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::UnidirectionalSequenceLstm, {}, {}, BaseDescriptor(), false);

    CHECK(basicBlock->GetName() == "");
    CHECK(basicBlock->GetTensors().size() == 0);
    CHECK(basicBlock->GetOperators().size() == 1);
    CHECK(basicBlock->GetInputs().size() == 0);
    CHECK(basicBlock->GetOutputs().size() == 0);

    TosaSerializationOperator* op = basicBlock->GetOperators()[0];
    CHECK(op->GetAttributeType() == Attribute_NONE);
    CHECK(op->GetOp() == tosa::Op_UNKNOWN);
    CHECK(op->GetInputTensorNames().size() == 0);
    CHECK(op->GetOutputTensorNames().size() == 0);
}
}

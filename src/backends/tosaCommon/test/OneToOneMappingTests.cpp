//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"
#include "CommonTestUtils.hpp"

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
        GetTosaMapping(nullptr, LayerType::Addition, {&info, &info}, {&info}, BaseDescriptor());
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
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(add));
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_ADD, Attribute_NONE, BaseDescriptor(), LayerType::Addition);
}

TEST_CASE("GetTosaMapping_ConstantLayer")
{
    TensorInfo outputInfo = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 2, 4, 2 }};

    TosaSerializationBasicBlock* basicBlock =
            GetTosaMapping(nullptr, LayerType::Constant, {}, {&outputInfo}, BaseDescriptor());
    AssertTosaOneToOneMappingBasicBlock(
            basicBlock, {}, outputShape, Op_CONST, Attribute_NONE, BaseDescriptor(), LayerType::Constant);
}

TEST_CASE("GetTosaMappingFromLayer_ConstantLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 2, 4, 2 }};

    std::vector<float> data = GenerateRandomData<float>(info.GetNumElements());
    ConstTensor constTensor(info, data);

    IConnectableLayer* constant = net->AddConstantLayer(constTensor, "constant");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    constant->GetOutputSlot(0).Connect(output->GetInputSlot(0));
    constant->GetOutputSlot(0).SetTensorInfo(info);

    TosaSerializationBasicBlock* basicBlock =
            GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(constant));
    AssertTosaOneToOneMappingBasicBlock(
            basicBlock, {}, outputShape, Op_CONST, Attribute_NONE, BaseDescriptor(), LayerType::Constant);
}

TEST_CASE("GetTosaMapping_Conv2dLayer")
{
    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_DilationX   = 2;
    descriptor.m_DilationY   = 2;
    descriptor.m_BiasEnabled = true;

    const TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    const TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    const TensorInfo biasesInfo ({ 1 }, DataType::Float32, 0.0f, 0, true);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 5, 5, 1 }, { 1, 3, 3, 1 }, { 1 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 3, 3, 1 }};

    TosaSerializationBasicBlock* basicBlock = GetTosaMapping(nullptr,
                                                             LayerType::Convolution2d,
                                                             {&inputInfo, &weightsInfo, &biasesInfo},
                                                             {&outputInfo},
                                                             descriptor);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_CONV2D, Attribute_ConvAttribute, descriptor, LayerType::Convolution2d);
}

TEST_CASE("GetTosaMappingFromLayer_Conv2dLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    Convolution2dDescriptor descriptor;
    descriptor.m_PadLeft     = 1;
    descriptor.m_PadRight    = 1;
    descriptor.m_PadTop      = 1;
    descriptor.m_PadBottom   = 1;
    descriptor.m_StrideX     = 2;
    descriptor.m_StrideY     = 2;
    descriptor.m_DilationX   = 2;
    descriptor.m_DilationY   = 2;
    descriptor.m_BiasEnabled = true;

    const TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    const TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    const TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    const TensorInfo biasesInfo ({ 1 }, DataType::Float32, 0.0f, 0, true);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 5, 5, 1 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 3, 3, 1 }};

    std::vector<float> weightsData = GenerateRandomData<float>(weightsInfo.GetNumElements());
    ConstTensor weights(weightsInfo, weightsData);

    std::vector<float> biasesData = GenerateRandomData<float>(biasesInfo.GetNumElements());
    ConstTensor biases(biasesInfo, biasesData);

    IConnectableLayer* const inputLayer  = net->AddInputLayer(0, "input0");
    IConnectableLayer* const weightsLayer = net->AddConstantLayer(weights, "weights");
    IConnectableLayer* const biasesLayer = net->AddConstantLayer(biases, "biases");
    IConnectableLayer* const convLayer   = net->AddConvolution2dLayer(descriptor, "conv2d");
    IConnectableLayer* const outputLayer = net->AddOutputLayer(0);

    inputLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(0));
    weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
    biasesLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));
    convLayer->GetOutputSlot(0).Connect(outputLayer->GetInputSlot(0));

    inputLayer->GetOutputSlot(0).SetTensorInfo(inputInfo);
    weightsLayer->GetOutputSlot(0).SetTensorInfo(weightsInfo);
    biasesLayer->GetOutputSlot(0).SetTensorInfo(biasesInfo);
    convLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    TosaSerializationBasicBlock* basicBlock = GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(convLayer));
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_CONV2D, Attribute_ConvAttribute, descriptor, LayerType::Convolution2d);
}

TEST_CASE("GetTosaMapping_MaxPool2DLayer")
{
    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = PaddingMethod::Exclude;

    TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor);
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_MAX_POOL2D, Attribute_PoolAttribute, descriptor, LayerType::Pooling2d);
}

TEST_CASE("GetTosaMappingFromLayer_MaxPool2DLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = PoolingAlgorithm::Max;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = PaddingMethod::Exclude;

    IConnectableLayer* input  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool   = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    input->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool));
    AssertTosaOneToOneMappingBasicBlock(
        basicBlock, inputShape, outputShape, Op_MAX_POOL2D, Attribute_PoolAttribute, descriptor, LayerType::Pooling2d);
}

TEST_CASE("GetTosaMapping_AvgPool2DLayer")
{
    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = PaddingMethod::Exclude;

    TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::Pooling2d, {&inputTensorInfo}, {&outputTensorInfo}, descriptor);
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

    Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = PoolingAlgorithm::Average;
    descriptor.m_PoolWidth = descriptor.m_PoolHeight = 2;
    descriptor.m_StrideX = descriptor.m_StrideY = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PaddingMethod = PaddingMethod::Exclude;

    IConnectableLayer* input0  = net->AddInputLayer(0, "input0");
    IConnectableLayer* pool    = net->AddPooling2dLayer(descriptor, "pool");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input0->GetOutputSlot(0).Connect(pool->GetInputSlot(0));
    pool->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo inputTensorInfo({ 1, 1, 4, 4 }, DataType::Float32);
    TensorInfo outputTensorInfo({ 1, 1, 3, 3 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 1, 1, 4, 4 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 1, 1, 3, 3 }};

    input0->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    pool->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(pool));
    AssertTosaOneToOneMappingBasicBlock(basicBlock,
                                        inputShape,
                                        outputShape,
                                        Op_AVG_POOL2D,
                                        Attribute_PoolAttribute,
                                        descriptor,
                                        LayerType::Pooling2d);
}

TEST_CASE("GetTosaMapping_ReshapeLayer")
{
    TensorInfo inputInfo = TensorInfo({ 2, 3 }, DataType::Float32);
    TensorInfo outputInfo = TensorInfo({ 6 }, DataType::Float32);

    std::vector<std::vector<int32_t>> inputShape  = {{ 2, 3 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 6 }};

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = { 6 };

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::Reshape, {&inputInfo}, {&outputInfo}, descriptor);
    AssertTosaOneToOneMappingBasicBlock(basicBlock,
                                        inputShape,
                                        outputShape,
                                        Op_RESHAPE,
                                        Attribute_ReshapeAttribute,
                                        descriptor,
                                        LayerType::Reshape);
}

TEST_CASE("GetTosaMappingFromLayer_ReshapeLayer")
{
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    ReshapeDescriptor descriptor;
    descriptor.m_TargetShape = { 6 };

    IConnectableLayer* input   = net->AddInputLayer(0, "input");
    IConnectableLayer* reshape = net->AddReshapeLayer(descriptor, "reshape");
    IConnectableLayer* output  = net->AddOutputLayer(0, "output");

    input->GetOutputSlot(0).Connect(reshape->GetInputSlot(0));
    reshape->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo inputInfo = TensorInfo({ 2, 3 }, DataType::Float32);
    TensorInfo outputInfo = TensorInfo({ 6 }, DataType::Float32);

    input->GetOutputSlot(0).SetTensorInfo(inputInfo);
    reshape->GetOutputSlot(0).SetTensorInfo(outputInfo);

    std::vector<std::vector<int32_t>> inputShape  = {{ 2, 3 }};
    std::vector<std::vector<int32_t>> outputShape = {{ 6 }};

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(reshape));
    AssertTosaOneToOneMappingBasicBlock(basicBlock,
                                        inputShape,
                                        outputShape,
                                        Op_RESHAPE,
                                        Attribute_ReshapeAttribute,
                                        descriptor,
                                        LayerType::Reshape);
}

TEST_CASE("GetTosaMapping_Unimplemented")
{
    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(nullptr, LayerType::UnidirectionalSequenceLstm, {}, {}, BaseDescriptor());

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

//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <Layer.hpp>

#include <tosaCommon/TosaMappings.hpp>

#include <doctest/doctest.h>

using namespace armnn;
using namespace tosa;

void AssertTosaOneToOneMappingBasicBlock(TosaSerializationBasicBlock* basicBlock,
                                         std::vector<int32_t> shape,
                                         uint32_t numInputs,
                                         uint32_t numOutputs,
                                         Op tosaOp,
                                         std::string operatorString,
                                         DType dataType = DType_FLOAT)
{
    std::string blockStr = operatorString + "_block_";
    ARMNN_ASSERT(basicBlock->GetName().find(blockStr)  != std::string::npos);
    ARMNN_ASSERT(basicBlock->GetInputs().size() == numInputs);
    ARMNN_ASSERT(basicBlock->GetOutputs().size() == numOutputs);
    ARMNN_ASSERT(basicBlock->GetOperators().size() == 1);
    ARMNN_ASSERT(basicBlock->GetTensors().size() == (numInputs + numOutputs));

    TosaSerializationOperator* op = basicBlock->GetOperators().at(0);
    ARMNN_ASSERT(op->GetInputTensorNames().size() == numInputs);
    ARMNN_ASSERT(op->GetOutputTensorNames().size() == numOutputs);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        std::basic_string<char> blockInputName = basicBlock->GetInputs()[i];
        std::basic_string<char> operatorInputName  = op->GetInputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[i]->GetName();

        std::string opStr = operatorString + "_input" + std::to_string(i) + "_";

        ARMNN_ASSERT(blockInputName == operatorInputName);
        ARMNN_ASSERT(tensorName == operatorInputName);
        ARMNN_ASSERT(blockInputName.find(opStr)  != std::string::npos);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        std::basic_string<char> blockOutputName = basicBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = op->GetOutputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[numInputs + i]->GetName();

        std::string opStr = operatorString + "_output" + std::to_string(i) + "_";

        ARMNN_ASSERT(blockOutputName == operatorOutputName);
        ARMNN_ASSERT(tensorName == operatorOutputName);
        ARMNN_ASSERT(blockOutputName.find(opStr)  != std::string::npos);
    }

    ARMNN_ASSERT(op->GetAttributeType() == Attribute_NONE);
    ARMNN_ASSERT(op->GetOp() == tosaOp);

    TosaSerializationTensor* tensor0 = basicBlock->GetTensors()[0];
    ARMNN_ASSERT(tensor0->GetDtype() == dataType);
    ARMNN_ASSERT(tensor0->GetData().size() == 0);
    ARMNN_ASSERT(tensor0->GetShape() == shape);
}

TEST_SUITE("TosaOperatorMappingOneToOneTests")
{
TEST_CASE("GetTosaMapping_AdditionLayer")
{
    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);
    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::Addition, {&info, &info}, {&info}, BaseDescriptor());
    AssertTosaOneToOneMappingBasicBlock(basicBlock, { 1, 2, 4, 2 }, 2, 1, Op::Op_ADD, "Op_ADD");
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

    TosaSerializationBasicBlock* basicBlock =
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(add));
    AssertTosaOneToOneMappingBasicBlock(basicBlock, { 1, 2, 4, 2 }, 2, 1, Op::Op_ADD, "Op_ADD");
}

TEST_CASE("GetTosaMapping_Unimplemented")
{
    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::UnidirectionalSequenceLstm, {}, {}, BaseDescriptor());

    ARMNN_ASSERT(basicBlock->GetName() == "");
    ARMNN_ASSERT(basicBlock->GetTensors().size() == 0);
    ARMNN_ASSERT(basicBlock->GetOperators().size() == 1);
    ARMNN_ASSERT(basicBlock->GetInputs().size() == 0);
    ARMNN_ASSERT(basicBlock->GetOutputs().size() == 0);

    TosaSerializationOperator* op = basicBlock->GetOperators()[0];
    ARMNN_ASSERT(op->GetAttributeType() == Attribute_NONE);
    ARMNN_ASSERT(op->GetOp() == tosa::Op_UNKNOWN);
    ARMNN_ASSERT(op->GetInputTensorNames().size() == 0);
    ARMNN_ASSERT(op->GetOutputTensorNames().size() == 0);
}
}

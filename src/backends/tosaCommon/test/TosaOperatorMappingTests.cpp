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
                                         DType dataType = DType_FP32)
{
    std::string blockStr = operatorString + "_block_";
    CHECK(basicBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(basicBlock->GetInputs().size() == numInputs);
    CHECK(basicBlock->GetOutputs().size() == numOutputs);
    CHECK(basicBlock->GetOperators().size() == 1);
    CHECK(basicBlock->GetTensors().size() == (numInputs + numOutputs));

    TosaSerializationOperator* op = basicBlock->GetOperators().at(0);
    CHECK(op->GetInputTensorNames().size() == numInputs);
    CHECK(op->GetOutputTensorNames().size() == numOutputs);

    for (uint32_t i = 0; i < numInputs; i++)
    {
        std::basic_string<char> blockInputName = basicBlock->GetInputs()[i];
        std::basic_string<char> operatorInputName  = op->GetInputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[i]->GetName();

        std::string opStr = operatorString + "_input" + std::to_string(i) + "_";

        CHECK(blockInputName == operatorInputName);
        CHECK(tensorName == operatorInputName);
        CHECK(blockInputName.find(opStr)  != std::string::npos);
    }

    for (uint32_t i = 0; i < numOutputs; i++)
    {
        std::basic_string<char> blockOutputName = basicBlock->GetOutputs()[i];
        std::basic_string<char> operatorOutputName  = op->GetOutputTensorNames()[i];
        std::basic_string<char> tensorName = basicBlock->GetTensors()[numInputs + i]->GetName();

        std::string opStr = operatorString + "_output" + std::to_string(i) + "_";

        CHECK(blockOutputName == operatorOutputName);
        CHECK(tensorName == operatorOutputName);
        CHECK(blockOutputName.find(opStr)  != std::string::npos);
    }

    CHECK(op->GetAttributeType() == Attribute_NONE);
    CHECK(op->GetOp() == tosaOp);

    TosaSerializationTensor* tensor0 = basicBlock->GetTensors()[0];
    CHECK(tensor0->GetDtype() == dataType);
    CHECK(tensor0->GetData().size() == 0);
    CHECK(tensor0->GetShape() == shape);
}

TEST_SUITE("TosaOperatorMappingOneToOneTests")
{
TEST_CASE("GetTosaMapping_AdditionLayer")
{
    TensorInfo info = TensorInfo({ 1, 2, 4, 2 }, DataType::Float32, 0.0f, 0, true);
    TosaSerializationBasicBlock* basicBlock =
        GetTosaMapping(LayerType::Addition, {&info, &info}, {&info}, BaseDescriptor(), false);
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
        GetTosaMappingFromLayer(PolymorphicDowncast<Layer*>(add), false);
    AssertTosaOneToOneMappingBasicBlock(basicBlock, { 1, 2, 4, 2 }, 2, 1, Op::Op_ADD, "Op_ADD");
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

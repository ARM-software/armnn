//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AdditionOperator.hpp"

TosaSerializationBasicBlock* ConvertAdditionToTosaOperator(const Layer* layer,
                                                           const std::vector<const TensorInfo*>& inputs,
                                                           const std::vector<const TensorInfo*>& outputs)
{
    std::string input0Name = std::string("input0_");
    std::string input1Name = std::string("input1_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_ADD_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique layer names.
        Layer& connectedLayer0 = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        input0Name = GenerateUniqueName(connectedLayer0, 0);

        Layer& connectedLayer1 = layer->GetInputSlot(1).GetConnectedOutputSlot()->GetOwningLayer();
        input1Name = GenerateUniqueName(connectedLayer1, 1);

        // Get the layer connected to the output slot and determine unique layer name.
        Layer& connectedOutputLayer = layer->GetOutputSlot().GetConnection(0)->GetOwningLayer();
        outputName = GenerateUniqueName(connectedOutputLayer, 0);
    }

    auto* op = new TosaSerializationOperator(Op_ADD,
                                             Attribute_NONE,
                                             nullptr,
                                             {input0Name, input1Name},
                                             {outputName});

    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
    DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    auto* inputTensor0  = new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {});
    auto* inputTensor1  = new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {});
    auto* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           {inputTensor0, inputTensor1, outputTensor0}, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}
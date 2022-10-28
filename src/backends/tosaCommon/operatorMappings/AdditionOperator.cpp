//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AdditionOperator.hpp"

TosaSerializationBasicBlock* ConvertAdditionToTosaOperator(const std::vector<const TensorInfo*>& inputs,
                                                           const std::vector<const TensorInfo*>& outputs)
{
    // A helper function with static global variables ensures uniqueness
    // for dynamically generating input, output and block names
    std::string input0Name = std::string("Op_ADD_input0_")  + GetUniqueTosaMappingID();
    std::string input1Name = std::string("Op_ADD_input1_")  + GetUniqueTosaMappingID();
    std::string outputName = std::string("Op_ADD_output0_") + GetUniqueTosaMappingID();
    std::string blockName  = std::string("Op_ADD_block_")   + GetUniqueTosaMappingID();

    TosaSerializationOperator* op = new TosaSerializationOperator(Op_ADD,
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

    TosaSerializationTensor* inputTensor0  = new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {});
    TosaSerializationTensor* inputTensor1  = new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {});
    TosaSerializationTensor* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           {inputTensor0, inputTensor1, outputTensor0}, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}
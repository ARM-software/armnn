//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2DOperator.hpp"

TosaSerializationBasicBlock* ConvertPooling2DToTosaOperator(const std::vector<const TensorInfo*>& inputs,
                                                            const std::vector<const TensorInfo*>& outputs,
                                                            bool isMain,
                                                            const Pooling2dDescriptor* poolDescriptor)
{
    std::string poolType = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? "Op_MAX" : "Op_AVG";
    Op opcode = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? Op_MAX_POOL2D : Op_AVG_POOL2D;

    // A helper function with static global variables ensures uniqueness
    // for dynamically generating input, output and block names
    std::string input0Name = poolType + std::string("_POOL2D_input0_")  + GetUniqueTosaMappingID();
    std::string outputName = poolType + std::string("_POOL2D_output0_") + GetUniqueTosaMappingID();
    std::string blockName  = poolType + std::string("_POOL2D_block_")   + GetUniqueTosaMappingID();

    // If it's the first block, overwrite block name with main.
    if (isMain)
    {
        blockName = std::string("main");
    }

    std::vector<int> pad = {static_cast<int>(poolDescriptor->m_PadTop),
                            static_cast<int>(poolDescriptor->m_PadBottom),
                            static_cast<int>(poolDescriptor->m_PadLeft),
                            static_cast<int>(poolDescriptor->m_PadRight)};
    std::vector<int> kernel = {static_cast<int>(poolDescriptor->m_PoolHeight),
                               static_cast<int>(poolDescriptor->m_PoolWidth)};
    std::vector<int> stride = {static_cast<int>(poolDescriptor->m_StrideY),
                               static_cast<int>(poolDescriptor->m_StrideX)};
    TosaPoolAttribute attribute(pad, kernel, stride, 0, 0, ArmNNToDType(inputs[0]->GetDataType()));

    TosaSerializationOperator* op = new TosaSerializationOperator(opcode,
                                                                  Attribute_PoolAttribute,
                                                                  &attribute,
                                                                  {input0Name},
                                                                  {outputName});

    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    TosaSerializationTensor* inputTensor0  = new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {});
    TosaSerializationTensor* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           {inputTensor0, outputTensor0}, // tensors
                                           {input0Name}, // inputs
                                           {outputName}); // outputs
}
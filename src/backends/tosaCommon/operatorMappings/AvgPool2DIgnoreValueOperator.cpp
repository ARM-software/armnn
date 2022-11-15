//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2DOperator.hpp"

TosaSerializationBasicBlock* ConvertAvgPool2DIgnoreValueToTosaOperator(const std::vector<const TensorInfo*>& inputs,
                                                                       const std::vector<const TensorInfo*>& outputs,
                                                                       bool isMain,
                                                                       const Pooling2dDescriptor* poolDescriptor)
{

    // A helper function with static global variables ensures uniqueness
    // for dynamically generating input, output and block names
    std::string padInputName   = std::string("Op_PAD_input0_")  + GetUniqueTosaMappingID();
    std::string padOutputName  = std::string("Op_PAD_intermediate0_")  + GetUniqueTosaMappingID();
    std::string poolOutputName = std::string("Op_AVG_POOL2D_output0_") + GetUniqueTosaMappingID();
    std::string blockName      = std::string("Op_AVG_POOL2D_block_")   + GetUniqueTosaMappingID();

    // If it's the first block, overwrite block name with main.
    if (isMain)
    {
        blockName = std::string("main");
    }

    std::vector<int> paddings;
    if (poolDescriptor->m_DataLayout == DataLayout::NHWC)
    {
        paddings = {0,
                    0,
                    static_cast<int>(poolDescriptor->m_PadTop),
                    static_cast<int>(poolDescriptor->m_PadBottom),
                    static_cast<int>(poolDescriptor->m_PadLeft),
                    static_cast<int>(poolDescriptor->m_PadRight),
                    0,
                    0
        };
    }
    else
    {
        paddings = {0,
                    0,
                    0,
                    0,
                    static_cast<int>(poolDescriptor->m_PadTop),
                    static_cast<int>(poolDescriptor->m_PadBottom),
                    static_cast<int>(poolDescriptor->m_PadLeft),
                    static_cast<int>(poolDescriptor->m_PadRight)
        };
    }

    TosaPadAttribute padAttribute(paddings, 0, 0.0f);
    TosaSerializationOperator* opPad = new TosaSerializationOperator(Op_PAD,
                                                                     Attribute_PadAttribute,
                                                                     &padAttribute,
                                                                     {padInputName},
                                                                     {padOutputName});

    std::vector<int> pad    = {0, 0, 0, 0};
    std::vector<int> kernel = {static_cast<int>(poolDescriptor->m_PoolHeight),
                               static_cast<int>(poolDescriptor->m_PoolWidth)};
    std::vector<int> stride = {static_cast<int>(poolDescriptor->m_StrideY),
                               static_cast<int>(poolDescriptor->m_StrideX)};
    TosaPoolAttribute poolAttribute(pad, kernel, stride, 0, 0, ArmNNToDType(inputs[0]->GetDataType()));

    TosaSerializationOperator* opPool = new TosaSerializationOperator(Op_AVG_POOL2D,
                                                                      Attribute_PoolAttribute,
                                                                      &poolAttribute,
                                                                      {padOutputName},
                                                                      {poolOutputName});

    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());

    std::vector<int32_t> intermediateShape;
    if (poolDescriptor->m_DataLayout == DataLayout::NHWC)
    {
        intermediateShape = {inputShape[0],
                             inputShape[1] + paddings[2] + paddings[3],
                             inputShape[2] + paddings[4] + paddings[5],
                             inputShape[3]};
    }
    else
    {
        intermediateShape = {inputShape[0],
                             inputShape[1],
                             inputShape[2] + paddings[4] + paddings[5],
                             inputShape[3] + paddings[6] + paddings[7]};
    }

    TosaSerializationTensor* inputTensor  = new TosaSerializationTensor(padInputName, inputShape, inputDType, {});
    TosaSerializationTensor* intermediateTensor  = new TosaSerializationTensor(
        padOutputName, intermediateShape, inputDType, {});
    TosaSerializationTensor* outputTensor = new TosaSerializationTensor(poolOutputName, outputShape, outputDType, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {opPad, opPool}, // operators
                                           {inputTensor, intermediateTensor, outputTensor}, // tensors
                                           {padInputName}, // inputs
                                           {poolOutputName}); // outputs
}
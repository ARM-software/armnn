//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConstantOperator.hpp"

#include <layers/ConstantLayer.hpp>

TosaSerializationBasicBlock* ConvertConstantToTosaOperator(const Layer* layer,
                                                           const std::vector<const TensorInfo*>& outputs)
{
    std::string outputName = std::string("constant_");
    std::string blockName  = std::string("Op_CONST_block_") + GetUniqueTosaMappingID();

    std::vector<uint8_t> uint8Data;

    // If a layer is present then the block will be used for execution, so names need to be unique.
    // Also, set constant tensor data.
    if(layer != nullptr)
    {
        outputName.append(std::to_string(layer->GetGuid()));
        blockName.append(std::to_string(layer->GetGuid()));

        auto constantLayer = PolymorphicDowncast<const armnn::ConstantLayer*>(layer);
        auto tensorInfo = constantLayer->GetOutputSlot().GetTensorInfo();

        uint8Data = ConvertConstantTensorDataToBuffer(constantLayer->m_LayerOutput);
    }

    auto* op = new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {outputName});

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Setup output tensor with constant tensor data if available.
    auto* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, uint8Data);

    return new TosaSerializationBasicBlock(blockName,       // name
                                           {op},            // operators
                                           {outputTensor0}, // tensors
                                           {},              // inputs
                                           {outputName});   // outputs
}
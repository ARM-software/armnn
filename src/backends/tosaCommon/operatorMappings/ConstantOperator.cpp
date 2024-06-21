//
// Copyright © 2022, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConstantOperator.hpp"

#include <layers/ConstantLayer.hpp>

TosaSerializationBasicBlock* ConvertConstantToTosaOperator(const Layer* layer,
                                                           const std::vector<const TensorInfo*>& outputs,
                                                           bool isDepthwiseConv2dWeights = false)
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

    std::vector<int32_t> outputShape0;

    if(isDepthwiseConv2dWeights)
    {
        // Constant weights are connected to a depthwise conv2d layer. From this get the depthwise conv2d input shape.
        TensorShape inputShape = 
            layer->GetOutputSlot().GetConnection(0)->GetOwningLayer().GetInputSlot(0).GetTensorInfo().GetShape();

        unsigned int multiplier = outputs[0]->GetShape()[3]/inputShape[3];

        // TOSA requires depthwise conv2d kernel to be converted from [1, H, W, C * M] to layout [H, W, C, M]
        outputShape0 = {
            static_cast<int32_t>(outputs[0]->GetShape()[1]),
            static_cast<int32_t>(outputs[0]->GetShape()[2]),
            static_cast<int32_t>(inputShape[3]),
            static_cast<int32_t>(multiplier)
        };
    }
    else
    {
        outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    }

    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Setup output tensor with constant tensor data if available.
    auto* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, uint8Data);

    return new TosaSerializationBasicBlock(blockName,       // name
                                           mainName,        // region name
                                           {op},            // operators
                                           {outputTensor0}, // tensors
                                           {},              // inputs
                                           {outputName});   // outputs
}
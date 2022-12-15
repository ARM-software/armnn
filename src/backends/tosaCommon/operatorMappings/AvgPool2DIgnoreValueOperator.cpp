//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2DOperator.hpp"

TosaSerializationBasicBlock* ConvertAvgPool2DIgnoreValueToTosaOperator(const Layer* layer,
                                                                       const std::vector<const TensorInfo*>& inputs,
                                                                       const std::vector<const TensorInfo*>& outputs,
                                                                       const Pooling2dDescriptor* poolDescriptor)
{
    std::string padInputName   = std::string("input0_");
    std::string padOutputName  = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string poolOutputName = std::string("output0_");
    std::string blockName      = std::string("Op_AVG_POOL2D_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensors names.
        Layer& connectedInputLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        padInputName = GenerateUniqueName(connectedInputLayer, 0);

        // Determine unique output tensor name.
        poolOutputName = GenerateUniqueOutputName(*layer, 0);
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
    auto* opPad = new TosaSerializationOperator(Op_PAD,
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

    auto* opPool = new TosaSerializationOperator(Op_AVG_POOL2D,
                                                 Attribute_PoolAttribute,
                                                 &poolAttribute,
                                                 {padOutputName},
                                                 {poolOutputName});

    std::vector<TosaSerializationTensor*> tensors;

    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(padInputName.find("input0_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(padInputName, inputShape, inputDType, {}));
    }

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

    tensors.push_back(new TosaSerializationTensor(padOutputName, intermediateShape, inputDType, {}));
    tensors.push_back(new TosaSerializationTensor(poolOutputName, outputShape, outputDType, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {opPad, opPool}, // operators
                                           tensors, // tensors
                                           {padInputName}, // inputs
                                           {poolOutputName}); // outputs
}
//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Pooling2DOperator.hpp"

TosaSerializationBasicBlock* ConvertPooling2DToTosaOperator(const Layer* layer,
                                                            const std::vector<const TensorInfo*>& inputs,
                                                            const std::vector<const TensorInfo*>& outputs,
                                                            const Pooling2dDescriptor* poolDescriptor)
{
    std::string poolType = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? "Op_MAX" : "Op_AVG";
    Op opcode = (poolDescriptor->m_PoolType == PoolingAlgorithm::Max) ? Op_MAX_POOL2D : Op_AVG_POOL2D;

    std::string input0Name = std::string("input0_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_") + poolType + std::string("_POOL2D_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique layer names.
        Layer& connectedInputLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        input0Name = GenerateUniqueName(connectedInputLayer, 0);

        // Get the layer connected to the output slot and determine unique layer name.
        Layer& connectedOutputLayer = layer->GetOutputSlot().GetConnection(0)->GetOwningLayer();
        outputName = GenerateUniqueName(connectedOutputLayer, 0);
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

    auto* op = new TosaSerializationOperator(opcode,
                                             Attribute_PoolAttribute,
                                             &attribute,
                                             {input0Name},
                                             {outputName});

    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    auto* inputTensor0  = new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {});
    auto* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           {inputTensor0, outputTensor0}, // tensors
                                           {input0Name}, // inputs
                                           {outputName}); // outputs
}
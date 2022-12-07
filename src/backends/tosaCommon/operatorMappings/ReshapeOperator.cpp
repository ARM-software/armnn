//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReshapeOperator.hpp"

TosaSerializationBasicBlock* ConvertReshapeToTosaOperator(const Layer* layer,
                                                          const std::vector<const TensorInfo*>& inputs,
                                                          const std::vector<const TensorInfo*>& outputs,
                                                          const ReshapeDescriptor* reshapeDescriptor)
{
    std::string inputName = std::string("input0_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_RESHAPE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique layer names.
        Layer& connectedLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        inputName = GenerateUniqueName(connectedLayer, 0);

        // Get the layer connected to the output slot and determine unique layer name.
        Layer& connectedOutputLayer = layer->GetOutputSlot().GetConnection(0)->GetOwningLayer();
        outputName = GenerateUniqueName(connectedOutputLayer, 0);
    }

    TosaReshapeAttribute attribute(GetTosaTensorShape(reshapeDescriptor->m_TargetShape));

    auto* op = new TosaSerializationOperator(Op_RESHAPE,
                                             Attribute_ReshapeAttribute,
                                             &attribute,
                                             {inputName},
                                             {outputName});

    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());

    auto* inputTensor  = new TosaSerializationTensor(inputName, inputShape, inputDType, {});
    auto* outputTensor = new TosaSerializationTensor(outputName, outputShape, outputDType, {});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           {inputTensor, outputTensor}, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
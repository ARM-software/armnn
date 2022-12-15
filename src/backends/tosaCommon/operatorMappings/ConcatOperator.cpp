//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConcatOperator.hpp"

TosaSerializationBasicBlock* ConvertConcatToTosaOperator(const Layer* layer,
                                                         const std::vector<const TensorInfo*>& inputs,
                                                         const std::vector<const TensorInfo*>& outputs,
                                                         const OriginsDescriptor* concatDescriptor)
{
    auto numInputs = inputs.size();
    std::vector<std::string> inputNames;
    inputNames.reserve(numInputs);
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_CONCAT_block_") + GetUniqueTosaMappingID();

    // Set input names for validation purposes only.
    if (layer == nullptr)
    {
        for (uint32_t i = 0; i < numInputs; ++i)
        {
            inputNames.push_back("input"+ std::to_string(i) +"_");
        }
    }
    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    else
    {
        // Get the layers connected to the input slots and determine unique tensor names.
        for (uint32_t i = 0; i < numInputs; ++i)
        {
            Layer& connectedLayer = layer->GetInputSlot(i).GetConnectedOutputSlot()->GetOwningLayer();

            std::string inputName = GenerateUniqueName(connectedLayer, i);
            inputNames.push_back(inputName);
        }

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer, 0);
    }

    auto axis = static_cast<int32_t>(concatDescriptor->GetConcatAxis());
    TosaAxisAttribute attribute(axis);

    TosaSerializationOperator* op = new TosaSerializationOperator(Op_CONCAT,
                                                                  Attribute_AxisAttribute,
                                                                  &attribute,
                                                                  inputNames,
                                                                  {outputName});

    std::vector<TosaSerializationTensor*> tensors;
    tensors.reserve(numInputs);

    for (uint32_t i = 0; i < numInputs; ++i)
    {
        // Only add input tensors for validation or when the connected layer is an input layer.
        // As there can't be duplicate tensors and intermediate or constant tensors are created separately.
        if(inputNames[i].find("input") != std::string::npos)
        {
            std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[i]->GetShape());
            DType inputDType = ArmNNToDType(inputs[i]->GetDataType());
            tensors.push_back(new TosaSerializationTensor(inputNames[i], inputShape, inputDType, {}));
        }
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    TosaSerializationTensor* outputTensor0 = new TosaSerializationTensor(outputName, outputShape0, outputDType0, {});
    tensors.push_back(outputTensor0);

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName,     // name
                                           {op},          // operators
                                           tensors,       // tensors
                                           inputNames,    // inputs
                                           {outputName}); // outputs
}
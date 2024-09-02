//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReshapeOperator.hpp"

TosaSerializationBasicBlock* ConvertReshapeToTosaOperator(const Layer* layer,
                                                          const std::vector<const TensorInfo*>& inputs,
                                                          const std::vector<const TensorInfo*>& outputs,
                                                          const ReshapeDescriptor* reshapeDescriptor)
{
    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_RESHAPE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    TosaReshapeAttribute attribute(GetTosaTensorShape(reshapeDescriptor->m_TargetShape));

    auto* op = new TosaSerializationOperator(Op_RESHAPE,
                                             Attribute_ReshapeAttribute,
                                             &attribute,
                                             {inputName},
                                             {outputName});

    std::vector<TosaSerializationTensor*> tensors;

    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        // TOSA spec requires input and output shape to be the same size.
        // Pad the inputShape with 1 values to meet this condition.
        std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
        if (inputShape.size() < outputShape.size())
        {
            for (size_t i = inputShape.size(); i < outputShape.size(); ++i)
            {
                inputShape.emplace_back(1);
            }
        }

        DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape, outputDType, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {op}, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
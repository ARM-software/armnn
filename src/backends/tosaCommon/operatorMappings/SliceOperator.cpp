//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceOperator.hpp"

TosaSerializationBasicBlock* ConvertSliceToTosaOperator(const Layer* layer,
                                                        const std::vector<const TensorInfo*>& inputs,
                                                        const std::vector<const TensorInfo*>& outputs,
                                                        const SliceDescriptor* sliceDescriptor)
{
    std::string inputName = std::string("input0_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_SLICE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensor names.
        Layer& connectedLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        inputName = GenerateUniqueName(connectedLayer, 0);

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer, 0);
    }

    std::vector<int32_t> begin(sliceDescriptor->m_Begin.begin(), sliceDescriptor->m_Begin.end());
    std::vector<int32_t> size(sliceDescriptor->m_Size.begin(), sliceDescriptor->m_Size.end());

    TosaSliceAttribute attribute(begin, size);

    auto* op = new TosaSerializationOperator(Op_SLICE,
                                             Attribute_SliceAttribute,
                                             &attribute,
                                             {inputName},
                                             {outputName});

    std::vector<TosaSerializationTensor*> tensors;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input0_") != std::string::npos)
    {
        std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape, outputDType, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
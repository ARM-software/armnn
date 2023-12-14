//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SplitOperator.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from function convertSplitOp
TosaSerializationBasicBlock* ConvertSplitToTosaOperator(const Layer* layer,
                                                        const std::vector<const TensorInfo*>& inputs,
                                                        const std::vector<const TensorInfo*>& outputs,
                                                        const SplitterDescriptor* splitDescriptor)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE( inputs.size() == 1,
                                         "ConvertSplitToTosaOperator: Split must have only one input" );

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE( outputs.size() >= 1,
                                         "ConvertSplitToTosaOperator: Split must have at least one output" );

    if (!inputs[0]->GetShape().AreAllDimensionsSpecified())
    {
        throw armnn::Exception("ConvertSplitToTosaOperator: Dynamic input dimensions are unsupported.");
    }

    std::string inputName = std::string("input0_");
    std::vector<std::string> outputNames;
    std::string blockName  = std::string("Op_SPLIT_block_") + GetUniqueTosaMappingID();

    unsigned int numSplit = splitDescriptor->GetNumViews();
    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensor names.
        Layer& connectedLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        inputName = GenerateUniqueName(connectedLayer, 0);

        for (unsigned int i=0; i < numSplit; ++i)
        {
            // Determine unique output(s) tensor name.
            std::string outputName = GenerateUniqueOutputName(*layer, i);
            outputNames.push_back(outputName);
        }
    }
    else
    {
        for (unsigned int i=0; i < numSplit; ++i)
        {
            // Determine unique output(s) tensor name.
            std::string outputName = "output" + std::to_string(i) + "_";
            outputNames.push_back(outputName);
        }
    }

    // Each slice op has a different beginning point.
    // The size is the same for each slice op.
    std::vector<int32_t> beginVals;
    beginVals.reserve(inputs[0]->GetNumDimensions());
    std::vector<int32_t> sizeVals;
    sizeVals.reserve(inputs[0]->GetNumDimensions());
    for (unsigned int j = 0; j < inputs[0]->GetNumDimensions(); ++j)
    {
        beginVals.emplace_back(0);
        uint32_t dim = inputs[0]->GetShape()[j];
        sizeVals.emplace_back(dim);
    }

    uint32_t axis = static_cast<uint32_t>(splitDescriptor->GetAxis());
    sizeVals[axis] = sizeVals[axis] / static_cast<int32_t>(numSplit);

    std::vector<TosaSerializationOperator*> ops;
    for (unsigned int i=0; i < numSplit; ++i)
    {
        beginVals[axis] = static_cast<int>(i) * sizeVals[axis];
        TosaSliceAttribute attribute(beginVals, sizeVals);
        auto* op = new TosaSerializationOperator(Op_SLICE,
                                                 Attribute_SliceAttribute,
                                                 &attribute,
                                                 {inputName},
                                                 {outputNames[i]});

        ops.push_back(op);
    }

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

    for (unsigned int i=0; i < numSplit; ++i)
    {
        tensors.push_back(new TosaSerializationTensor(outputNames[i], outputShape, outputDType, {}));
    }
    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           ops, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           outputNames); // outputs
}
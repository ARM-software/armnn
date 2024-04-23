//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SplitOperator.hpp"
#include <backendsCommon/WorkloadUtils.hpp>

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

    std::string inputName = std::string("input_");
    std::vector<std::string> outputNames;
    std::string blockName  = std::string("Op_SPLIT_block_") + GetUniqueTosaMappingID();

    unsigned int numSplit = splitDescriptor->GetNumViews();
    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));

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

    // Configure input and output tensors
    std::set<unsigned int> splitAxis = ComputeSplitAxis(*splitDescriptor, inputs[0]->GetShape());
    if (splitAxis.size() != 1)
    {
        throw InvalidArgumentException("Cannot derive split axis from SplitterDescriptor");
    }
    uint32_t axis = *splitAxis.begin();

    std::vector<TosaSerializationOperator*> ops;
    std::vector<int32_t> beginVals(inputs[0]->GetNumDimensions(), 0);
    for (unsigned int i = 0; i < numSplit; ++i)
    {
        std::vector<int32_t> sizeVals = GetTosaTensorShape(outputs[i]->GetShape());
        TosaSliceAttribute attribute(beginVals, sizeVals);
        auto* op = new TosaSerializationOperator(Op_SLICE,
                                                 Attribute_SliceAttribute,
                                                 &attribute,
                                                 {inputName},
                                                 {outputNames[i]});

        ops.push_back(op);

        // Update the axis begin value for the next split operation, to be the correct size axis value.
        beginVals[axis] += sizeVals[axis];
    }

    std::vector<TosaSerializationTensor*> tensors;
    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());
    for (unsigned int i = 0; i < numSplit; ++i)
    {
        std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[i]->GetShape());
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
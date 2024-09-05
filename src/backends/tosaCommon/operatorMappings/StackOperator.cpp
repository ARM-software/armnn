//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "StackOperator.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from function convertPackOp
TosaSerializationBasicBlock* ConvertStackToTosaOperator(const Layer* layer,
                                                        const std::vector<const TensorInfo*>& inputs,
                                                        const std::vector<const TensorInfo*>& outputs,
                                                        const StackDescriptor* stackDescriptor)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() >= 1,
                                        "ConvertStackToTosaOperator: Stack must have at least one input");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() == 1,
                                        "ConvertStackToTosaOperator: Stack must have only one output");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs[0]->GetShape() != TensorShape(Dimensionality::Scalar),
                                        "ConvertStackToTosaOperator: Scalar / Rank 0 input not supported");

    const auto inputTensorRank = inputs[0]->GetNumDimensions();

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputTensorRank != 0,
                                        "ConvertStackToTosaOperator: Scalar / Rank 0 input not supported");

    // Verify axis value
    if (stackDescriptor->m_Axis > inputTensorRank)
    {
        throw armnn::Exception("ConvertStackToTosaOperator: Axis is out of a valid range.");
    }

    // Verify output rank
    if (outputs[0]->GetNumDimensions() != inputTensorRank + 1)
    {
        throw armnn::Exception("ConvertStackToTosaOperator: Output shape mismatch.");
    }

    auto inputDType = ArmNNToDType(inputs[0]->GetDataType());

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    std::string blockName = std::string("Op_STACK_block_") + GetUniqueTosaMappingID();
    auto blockOutputShape = GetTosaTensorShape(outputs[0]->GetShape());

    // Create input tensors
    std::vector<std::string> inputNames;
    for (unsigned int i = 0; i < inputs.size(); ++i)
    {
        if (inputs[i]->GetShape() != stackDescriptor->m_InputShape)
        {
            throw armnn::Exception("ConvertStackToTosaOperator: Inputs have mismatched shapes.");
        }

        std::string inputName = "input" + std::to_string(i) + "_";

        if (layer != nullptr)
        {
            inputName = GenerateUniqueInputName(layer->GetInputSlot(i));
        }

        tensors.emplace_back(new TosaSerializationTensor(inputName,
                                                         GetTosaTensorShape(inputs[i]->GetShape()),
                                                         inputDType,
                                                         {}));
        inputNames.push_back(inputName);
    }

    // Create output tensor
    std::string outputName = std::string("output0_");
    tensors.emplace_back(new TosaSerializationTensor(outputName,
                                                     blockOutputShape,
                                                     inputDType,
                                                     {}));

    bool transposeOpNeeded = (stackDescriptor->m_Axis == inputTensorRank);

    // Determine concatenation properties and transpose permutation
    std::vector<int32_t> permutationOrder;
    std::vector<int32_t> reshapeOutputShape;
    uint32_t concatAxis;
    if (transposeOpNeeded)
    {
        concatAxis = 0;

        reshapeOutputShape.push_back(static_cast<int32_t>(blockOutputShape[stackDescriptor->m_Axis]));

        for (unsigned int d = 0; d < inputTensorRank; d++)
        {
            permutationOrder.push_back(static_cast<int32_t>(d) + 1);
            reshapeOutputShape.push_back(static_cast<int32_t>(blockOutputShape[d]));
        }
        permutationOrder.push_back(0);
    }
    else
    {
        concatAxis = stackDescriptor->m_Axis;
    }

    // Determine concatenated output shape
    std::vector<int32_t> concatOutputShape;
    auto inputTensorShape = GetTosaTensorShape(stackDescriptor->m_InputShape);
    for (unsigned int i = 0; i < inputTensorRank; i++)
    {
        concatOutputShape.push_back(inputTensorShape[i]);
    }

    concatOutputShape[concatAxis] *= static_cast<int>(stackDescriptor->m_NumInputs);

    // Concatenation operator
    std::string concatOutputName = std::string("intermediate1_concat_") + GetUniqueTosaMappingID();

    TosaAxisAttribute axisAttribute(static_cast<int32_t>(concatAxis));

    auto* concatOp = new TosaSerializationOperator(Op_CONCAT,
                                                   Attribute_AxisAttribute,
                                                   &axisAttribute,
                                                   inputNames,
                                                   {concatOutputName});
    operators.push_back(concatOp);

    tensors.emplace_back(new TosaSerializationTensor(concatOutputName,
                                                     concatOutputShape,
                                                     inputDType,
                                                     {}));

    // Reshape operator
    std::string reshapeOutputName = std::string("intermediate2_reshape_") + GetUniqueTosaMappingID();
    std::string& reshapeOpOutputName = transposeOpNeeded ? reshapeOutputName : outputName;

    TosaReshapeAttribute reshapeAttribute = transposeOpNeeded ? reshapeOutputShape : blockOutputShape;

    auto* reshapeOp = new TosaSerializationOperator(Op_RESHAPE,
                                                    Attribute_ReshapeAttribute,
                                                    &reshapeAttribute,
                                                    {concatOutputName},
                                                    {reshapeOpOutputName});
    operators.push_back(reshapeOp);

    if (transposeOpNeeded)
    {
        // Transpose operator
        tensors.emplace_back(new TosaSerializationTensor(reshapeOutputName,
                                                         reshapeOutputShape,
                                                         inputDType,
                                                         {}));

        TosaTransposeAttribute transposeAttribute(permutationOrder);

        TosaSerializationOperator *transposeOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                                               Attribute_TransposeAttribute,
                                                                               &transposeAttribute,
                                                                               {reshapeOutputName},
                                                                               {outputName});
        operators.push_back(transposeOp);
    }

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName,     // name
                                           mainName,      // region name
                                           operators,     // operators
                                           tensors,       // tensors
                                           inputNames,    // inputs
                                           {outputName}); // outputs
}
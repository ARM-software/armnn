//
// Copyright © 2024-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceOperator.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc

TosaSerializationBasicBlock* ConvertStridedSliceToTosaOperator(const Layer* layer,
                                                               const std::vector<const TensorInfo*>& inputs,
                                                               const std::vector<const TensorInfo*>& outputs,
                                                               const StridedSliceDescriptor* stridedSliceDescriptor)
{
    // Limitations
    if (stridedSliceDescriptor->m_EllipsisMask != 0)
    {
        throw armnn::Exception("ConvertStridedSliceToTosaOperator: Ellipses mask not supported.");
    }

    /// Begin with the slice
    std::vector<int32_t> begin(stridedSliceDescriptor->m_Begin);
    std::vector<int32_t> end(stridedSliceDescriptor->m_End);
    std::vector<int32_t> strides(stridedSliceDescriptor->m_Stride);

    for (auto stride : strides)
    {
        if (stride != 1)
        {
            // Only strides with values 1 supported otherwise reshape invoked which creates tensors with more than 5D
            throw armnn::Exception("ConvertStridedSliceToTosaOperator: Strides greater than 1 not supported.");
        }
    }

    std::string inputName = std::string("input_");
    std::string outputNameSlice = std::string("intermediate1_") + GetUniqueTosaMappingID();
    std::string outputNameReshape = std::string("intermediate2_") + GetUniqueTosaMappingID();
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_SLICE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator *> operators;

    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());
    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());

    // Figure out size
    uint32_t inputRank = inputs[0]->GetShape().GetNumDimensions();

    // handle cases where end or begin values are negative
    for (uint32_t i = 0; i < inputRank; ++i)
    {
        if (end[i] < 0)
        {
            end[i] = static_cast<int32_t>(inputShape[i]) + end[i];
        }
        if (begin[i] < 0)
        {
            begin[i] = static_cast<int32_t>(inputShape[i]) + begin[i];
        }
    }

    std::vector<int32_t> a1_size(inputRank);

    // If mask set default to begin and end size from input tensor
    for (uint32_t i = 0; i < inputRank; ++i)
    {
        if (stridedSliceDescriptor->m_BeginMask & (1 << i))
        {
            begin[i] = 0;
        }
        if (stridedSliceDescriptor->m_EndMask & (1 << i))
        {
            end[i] = inputShape[i];
        }

        a1_size[i] = end[i] - begin[i];
    }

    TosaSliceAttribute sliceAttribute(begin, a1_size);

    auto* sliceOp1 = new TosaSerializationOperator(Op_SLICE,
                                                   Attribute_SliceAttribute,
                                                   &sliceAttribute,
                                                   {inputName},
                                                   {outputNameSlice});

    tensors.push_back(new TosaSerializationTensor(outputNameSlice, a1_size, outputDType, {}));
    operators.push_back(sliceOp1);

    // If unary striding is used we can reverse, reshape, and return the result.
    std::vector<int32_t> newShape;

    for (uint32_t i = 0; i < inputRank; ++i)
    {
        // Remove dimension specified in ShrinkAxisMask
        if (!(stridedSliceDescriptor->m_ShrinkAxisMask & (1 << i)))
        {
            newShape.push_back(a1_size[i]);
        }
    }

    TosaReshapeAttribute reshapeAttribute2(newShape);

    auto* reshapeOp2 = new TosaSerializationOperator(Op_RESHAPE,
                                                     Attribute_ReshapeAttribute,
                                                     &reshapeAttribute2,
                                                     {outputNameSlice},
                                                     {outputName});

    tensors.push_back(new TosaSerializationTensor(outputName, newShape, outputDType, {}));
    operators.push_back(reshapeOp2);

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           operators, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
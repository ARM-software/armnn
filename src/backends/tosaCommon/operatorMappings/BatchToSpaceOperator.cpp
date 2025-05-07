//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "BatchToSpaceOperator.hpp"

TosaSerializationBasicBlock* ConvertBatchToSpaceToTosaOperator(const Layer* layer,
                                                               const std::vector<const TensorInfo*>& inputs,
                                                               const std::vector<const TensorInfo*>& outputs,
                                                               const BatchToSpaceNdDescriptor* batchToSpaceDescriptor)
{

    /*
    * BatchToSpaceND - TOSA Lowering Overview
    * --------------------------------------
    * This operation takes a tensor shaped like [B, D1, D2, ..., DN, C]
    * and moves data from the batch dimension into the spatial dimensions.
    * 
    * It essentially reverses the logic of SpaceToBatchND, undoing the folding of spatial data into the batch.
    *
    * List of the steps involved:
    *
    * 1. Reshape:
    *    - We begin by expanding the batch dimension into block shapes.
    *      Specifically, B is split into: [b1, b2, ..., bN, B’], where B’ is the original batch divided by the product
    *      of block sizes.
    *    - This produces an intermediate shape like:
    *      [b1, b2, ..., bN, B’, D1, D2, ..., DN, C]
    *      e.g. if input is [4, 2, 2, 1] with block size [2,2], then:
    *      Reshape to [2, 2, 1, 2, 2, 1]
    *
    * 2. Transpose:
    *    - We rearrange the dimensions so that the blocks align with the spatial dimensions.
    *    - The transpose permutation reorders the tensor to:
    *      [B’, D1, b1, D2, b2, ..., DN, bN, C]
    *      e.g. [2, 2, 1, 2, 2, 1] becomes [1, 2, 2, 2, 2, 1]
    *
    * 3. Reshape:
    *    - Each spatial dimension is now expanded:
    *      Di' = Di * bi
    *    - After reshaping, the tensor looks like:
    *      [B’, D1 * b1, D2 * b2, ..., DN * bN, C]
    *      Continuing the example: [1, 2, 2, 2, 2, 1] → [1, 4, 4, 1]
    *
    * 4. Slice:
    *    - The final step removes any excess padding that may have existed in the original SpaceToBatchND.
    *    - Begin and end paddings are subtracted from the spatial dimensions.
    *      This restores the original unpadded spatial shape.
    *      e.g. if padded spatial shape was [4,4] and crop sizes were [[0,0],[0,0]] → no slice needed,
    *      but with crops [[1,1],[1,1]] → output becomes [1,2,2,1]
    */

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() == 1,
                                        "ConvertBatchToSpaceToTosaOperator: BatchToSpace must have only one input");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() == 1,
                                        "ConvertBatchToSpaceToTosaOperator: BatchToSpace must have only one output");

    std::string inputName = "input_";
    std::string outputNameReshape1 = "layer_intermediate1_" + GetUniqueTosaMappingID();
    std::string outputNameTranspose = "layer_intermediate2_" + GetUniqueTosaMappingID();
    std::string outputNameReshape2 = "layer_intermediate3_" + GetUniqueTosaMappingID();
    std::string outputName = "output0_";
    std::string blockName = "Op_BATCHTOSPACE_block_" + GetUniqueTosaMappingID();

    if (layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }
    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    const auto& crops = batchToSpaceDescriptor->m_Crops;
    const auto& blockShape = batchToSpaceDescriptor->m_BlockShape;
    const std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());
    const DType inputDType = ArmNNToDType(inputs[0]->GetDataType());
    const size_t inputRank = inputShape.size();
    const size_t blockRank = blockShape.size();
    const size_t remRank = inputRank - blockRank - 1;

    if (inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    if (inputRank < 2 || blockRank < 1 || blockShape.size() != crops.size())
    {
        throw armnn::Exception("ConvertBatchToSpaceToTosaOperator: Unsupported BatchToSpaceND config.");
        return nullptr;
    }

    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    // calculate the total number of blockElements
    int32_t blockNumElems = 1;
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        blockNumElems *= static_cast<int32_t>(blockShape[i]);
    }
    // using the input batch work out the new batch value
    const int32_t inputBatch = inputShape[0];
    const int32_t newBatch = inputBatch / blockNumElems;

    // Reshape  input to [block_shape..., batch / product(block), input_dims[1..]]
    std::vector<int32_t> reshape1Shape;
    for (size_t i = 0; i < blockRank; ++i)
    {
        reshape1Shape.push_back(static_cast<int32_t>(blockShape[i]));
    }

    reshape1Shape.push_back(newBatch);
    reshape1Shape.insert(reshape1Shape.end(), inputShape.begin() + 1, inputShape.end());

    tensors.push_back(new TosaSerializationTensor(outputNameReshape1, reshape1Shape, inputDType, {}));
    TosaReshapeAttribute reshape1Attr(reshape1Shape);

    operators.push_back(new TosaSerializationOperator(Op_RESHAPE,
                                                      Attribute_ReshapeAttribute,
                                                      &reshape1Attr,
                                                      {inputName},
                                                      {outputNameReshape1}));

    // interleave block dimensions with spatial dims
    std::vector<int32_t> perm;
    perm.push_back(static_cast<int32_t>(blockRank));
    for (size_t i = 0; i < blockRank; ++i)
    {
        perm.push_back(static_cast<int32_t>(blockRank + 1 + i));
        perm.push_back(static_cast<int32_t>(i));
    }
    for (size_t i = 0; i < remRank; ++i)
    {
        perm.push_back(static_cast<int32_t>(2 * blockRank + 1 + i));
    }

    std::vector<int32_t> transposeShape(perm.size());
    for (size_t i = 0; i < perm.size(); ++i)
    {
        transposeShape[i] = reshape1Shape[static_cast<size_t>(perm[i])];
    }

    tensors.push_back(new TosaSerializationTensor(outputNameTranspose, transposeShape, inputDType, {}));
    TosaTransposeAttribute transposeAttr(perm);

    operators.push_back(new TosaSerializationOperator(Op_TRANSPOSE,
                                                      Attribute_TransposeAttribute,
                                                      &transposeAttr,
                                                      {outputNameReshape1},
                                                      {outputNameTranspose}));

    // Reshape data to [new_batch, spatial dims * block, remainder]
    std::vector<int32_t> reshape2Shape;
    reshape2Shape.push_back(newBatch);

    for (size_t i = 0; i < blockRank; ++i)
    {
        int32_t value = inputShape[1 + i] * static_cast<int32_t>(blockShape[i]);
        reshape2Shape.push_back(value);
    }

    for (size_t i = 0; i < remRank; ++i)
    {
        reshape2Shape.push_back(inputShape[1 + blockRank + i]);
    }

    tensors.push_back(new TosaSerializationTensor(outputNameReshape2, reshape2Shape, inputDType, {}));
    TosaReshapeAttribute reshape2Attr(reshape2Shape);

    operators.push_back(new TosaSerializationOperator(Op_RESHAPE,
                                                      Attribute_ReshapeAttribute,
                                                      &reshape2Attr,
                                                      {outputNameTranspose},
                                                      {outputNameReshape2}));

    // slice the data to remove cropped areas from spatial dims
    std::vector<int32_t> begin(reshape2Shape.size(), 0);
    std::vector<int32_t> slicedShape = reshape2Shape;

    for (size_t i = 0; i < crops.size(); ++i)
    {
        begin[1 + i] = static_cast<int32_t>(crops[i].first);
        slicedShape[1 + i] -= static_cast<int32_t>(crops[i].first + crops[i].second);
    }

    tensors.push_back(new TosaSerializationTensor(outputName, slicedShape, inputDType, {}));
    
    TosaSliceAttribute sliceAttr(begin, slicedShape);
    operators.push_back(new TosaSerializationOperator(Op_SLICE,
                                                      Attribute_SliceAttribute,
                                                      &sliceAttr,
                                                      {outputNameReshape2},
                                                      {outputName}));

    std::vector<int32_t> expectedShape = GetTosaTensorShape(outputs[0]->GetShape());
    
    if (slicedShape != expectedShape)
    {
       throw armnn::Exception("ConvertSpaceToBatchToTosaOperator: Mismatch expected output and generated shape differ");
    }
    return new TosaSerializationBasicBlock(blockName, mainName, operators, tensors, {inputName}, {outputName});
}
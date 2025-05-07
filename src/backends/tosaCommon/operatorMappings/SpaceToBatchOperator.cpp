//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SpaceToBatchOperator.hpp"

TosaSerializationBasicBlock* ConvertSpaceToBatchToTosaOperator(const Layer* layer,
                                                               const std::vector<const TensorInfo*>& inputs,
                                                               const std::vector<const TensorInfo*>& outputs,
                                                               const SpaceToBatchNdDescriptor* spaceToBatchDescriptor)
{
    /*
    * SpaceToBatchND - TOSA Lowering Overview
    * --------------------------------------
    * This operation takes a tensor for example one shaped like [B, D1, D2,- DN, C]
    * and moves data from the spatial dimensions (D1-DN) into batch dimension.
    *
    * List of the steps involved:
    *
    * 1. Pad
    *    - Padding is applied in all cases whether there is 0 padding or not
    *      The reason padding is required is so that the reshape can work properly
    *      The input spatial dimensions in the tensor have to be evenly divisible by the block size
    *      so for example if you had a tensor that was shaped [1,5,5,1] with a block size of [2,2] that means you would
    *      need to pad with at least a value of [1,1] to allow the operation to proceed 
    * 
    * 2. Reshape (plus padding):
    *    - For each spatial dimension and its block size we split it in two: [Di / bi, bi].
    *    - After doing that across all spatial dims, the tensor ends up looking like:
    *      [B, D1 / b1, b1, D2 / b2, b2, ..., DN / bN, bN, C]
    *      e.g. input tensor [1,4,4,1] with block size [2,2] padding [0,0]
    *      would become [1, 2, 2, 2, 2, 1]
    *
    * 3. Transpose:
    *    - We move data around so that the block dimensions (b1...bN) are at the beginning.
    *    - Batch (B) moves after the block dims, followed by the reduced spatial dims
    *      and whatever else is left after that (usually the channels).
    *    - The transpose permutation vector at this point looks something like:
    *      [block_dims..., B, spatial_dims..., remainder]
    *      e.g. following on from the last example the previous input of [1, 2, 2, 2, 2, 1] would transpose
    *      to [2, 2, 1, 2, 2, 1]
    *
    * 4. Final Reshape:
    *    - We fold all the block dims into the batch.
    *      So new_batch = B * b1 * b2 * ... * bN.
    *    - The final shape becomes:
    *      [new_batch, D1 / b1, D2 / b2, ..., DN / bN, C]
    *      [2, 2, 1, 2, 2, 1] -> [4, 2, 2, 1]
    */


    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() == 1,
                                        "ConvertSpaceToBatchToTosaOperator: SpaceToBatch must have only one input");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() == 1,
                                        "ConvertSpaceToBatchToTosaOperator: SpaceToBatch must have only one output");

    std::string inputName = "input_";
    std::string outputNamePad = "layer_intermediate1_" + GetUniqueTosaMappingID();
    std::string outputNameReshape1 = "layer_intermediate2_" + GetUniqueTosaMappingID();
    std::string outputNameTranspose = "layer_intermediate3_" + GetUniqueTosaMappingID();
    std::string outputName = "output0_";
    std::string blockName = "Op_SPACETOBATCH_block_" + GetUniqueTosaMappingID();

    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    const auto& paddings = spaceToBatchDescriptor->m_PadList;
    const auto& blockShape = spaceToBatchDescriptor->m_BlockShape;
    const unsigned int inputRank = inputs[0]->GetShape().GetNumDimensions();
    const unsigned int blockRank = static_cast<unsigned int>(blockShape.size());
    std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());

    if (inputRank <= blockRank)
    {
        throw armnn::Exception("ConvertSpaceToBatchToTosaOperator: input rank must be greater than block rank");
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // create a padding vector which is double the size of the inputRank
    // each dimension requires two values lo and hi padding
    std::vector<int32_t> a0Pad(2 * inputRank, 0);
    std::vector<int32_t> paddedShape = inputShape;

    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());

    if (inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }
    // Build up the padding for the pad operation
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        int32_t loPad = static_cast<int32_t>(paddings[i].first);
        int32_t hiPad = static_cast<int32_t>(paddings[i].second);
        size_t dimIndex = i + 1;
        a0Pad[2 * dimIndex] = loPad;
        a0Pad[2 * dimIndex + 1] = hiPad;
        paddedShape[dimIndex] = inputShape[dimIndex] + loPad + hiPad;
    }

    std::string padOutput = outputNamePad + "_padded"; 

    tensors.push_back(new TosaSerializationTensor(padOutput, paddedShape, inputDType, {}));

    // handle pad value if input is quantized
    float padValue = 0.0f;
    if (inputs[0]->IsQuantized())
    {
        padValue = static_cast<float>(inputs[0]->GetQuantizationOffset()) * inputs[0]->GetQuantizationScale();
    }

    TosaPadAttribute padAttr(a0Pad, 0, padValue);
    operators.push_back(new TosaSerializationOperator(Op_PAD,
                                                      Attribute_PadAttribute,
                                                      &padAttr,
                                                      {inputName},
                                                      {padOutput}));

    // setup the first reshape operation
    std::vector<int32_t> reshape1;
    // add the original batch dimension
    reshape1.push_back(inputShape[0]);

    // setup a variable to keep track of the total block multiplier
    int32_t blockNumElems = 1;

    // iterate over the rest of the spatial dimensions i.e. H, W, D
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        int32_t paddedDim = paddedShape[i + 1];  // padded spatial dimension
        int32_t blockDim = static_cast<int32_t>(blockShape[i]);  // block dimension to be transposed into batch
        if (paddedDim % blockDim != 0)
        {
            throw armnn::Exception("ConvertSpaceToBatchToTosaOperator: padded spatial dim not divisible by block size");
        }
        reshape1.push_back(paddedDim / blockDim);
        reshape1.push_back(blockDim);

        blockNumElems *= blockDim;
    }

    // append any remaining non spatial dimensions as is
    for (size_t i = 1 + blockShape.size(); i < inputShape.size(); ++i)
    {
        reshape1.push_back(inputShape[i]);
    }

    tensors.push_back(new TosaSerializationTensor(outputNameReshape1, reshape1, inputDType, {}));
    TosaReshapeAttribute reshapeAttr(reshape1);
    operators.push_back(new TosaSerializationOperator(Op_RESHAPE,
                                                      Attribute_ReshapeAttribute,
                                                      &reshapeAttr,
                                                      {padOutput},
                                                      {outputNameReshape1}));

    std::vector<int32_t> transposeVec;

    // move all the block dimensions to the front before the batch dimension
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        transposeVec.push_back(static_cast<int32_t>(1 + 2 * i + 1));
    }
    // add the original batch dimensions (always located at pos 0 of the previously reshaped data)
    transposeVec.push_back(0);

    // add the spatial dimensions
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        transposeVec.push_back(static_cast<int32_t>(1 + 2 * i));
    }

    // add any remaining dimensions
    for (size_t i = 1 + 2 * blockShape.size(); i < reshape1.size(); ++i)
    {
        transposeVec.push_back(static_cast<int32_t>(i));
    }
    // copy the reshaped1 value to begin applying the transpose to it
    std::vector<int32_t> transposeShape(transposeVec.size());
    for (size_t i = 0; i < transposeVec.size(); ++i)
    {
        transposeShape[i] = reshape1[static_cast<size_t>(transposeVec[i])];
    }
    tensors.push_back(new TosaSerializationTensor(outputNameTranspose, transposeShape, inputDType, {}));

    TosaTransposeAttribute transposeAttr(transposeVec);

    operators.push_back(new TosaSerializationOperator(Op_TRANSPOSE,
                                                      Attribute_TransposeAttribute,
                                                      &transposeAttr,
                                                      {outputNameReshape1},
                                                      {outputNameTranspose}));

    // setup vector to hold final reshape information
    std::vector<int32_t> reshape2;
    // determine the new batch size, which is the total number of block elements multiplied by the original batch
    const int32_t newBatch = static_cast<int32_t>(inputShape[0]) * static_cast<int32_t>(blockNumElems);
    reshape2.push_back(newBatch);

    // Add spatial dims each of which is reduced by its corresponding block i.e. padded / block
    for (size_t i = 0; i < blockShape.size(); ++i)
    {
        int32_t paddedDim = paddedShape[i + 1];
        int32_t blockDim  = static_cast<int32_t>(blockShape[i]);

        if (blockDim == 0 || paddedDim % blockDim != 0)
        {
           throw armnn::Exception("ConvertSpaceToBatchToTosaOperator: Invalid block Shape or padding in final reshape");
        }

        reshape2.push_back(paddedDim / blockDim);
    }

    // Add remaining dims
    reshape2.push_back(inputShape.back());
    tensors.push_back(new TosaSerializationTensor(outputName, reshape2, inputDType, {}));

    TosaReshapeAttribute reshape2Attr(reshape2);
    operators.push_back(new TosaSerializationOperator(Op_RESHAPE,
                                                    Attribute_ReshapeAttribute,
                                                    &reshape2Attr,
                                                    {outputNameTranspose},
                                                    {outputName}));

    std::vector<int32_t> expectedShape = GetTosaTensorShape(outputs[0]->GetShape());
    
    if (reshape2 != expectedShape)
    {
       throw armnn::Exception("ConvertSpaceToBatchToTosaOperator: Mismatch expected output and generated shape differ");
    }

    return new TosaSerializationBasicBlock(blockName, mainName, operators, tensors, {inputName}, {outputName});
}

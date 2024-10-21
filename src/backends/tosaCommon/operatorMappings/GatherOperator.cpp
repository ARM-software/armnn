//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "StackOperator.hpp"

#include <numeric>

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from function convertGatherOp
TosaSerializationBasicBlock* ConvertGatherToTosaOperator(const Layer* layer,
                                                         const std::vector<const TensorInfo*>& inputs,
                                                         const std::vector<const TensorInfo*>& outputs,
                                                         const GatherDescriptor* gatherDescriptor)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() == 2,
                                        "ConvertGatherToTosaOperator: Gather must have two inputs");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() == 1,
                                        "ConvertGatherToTosaOperator: Gather must have only one output");

    unsigned int paramsRank = inputs[0]->GetNumDimensions();
    unsigned int indicesRank = inputs[1]->GetNumDimensions();

    int batch_dims = 0; // ArmNN does not currently support this parameter, setting it to the default value.

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(gatherDescriptor->m_Axis >= 0 &&
                                        gatherDescriptor->m_Axis < static_cast<int32_t>(paramsRank),
                                        "ConvertGatherToTosaOperator: axis must be < values rank");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(batch_dims <= static_cast<int32_t>(indicesRank),
                                        "ConvertGatherToTosaOperator: batch dimensions must be <= indices rank");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(gatherDescriptor->m_Axis >= batch_dims,
                                        "ConvertGatherToTosaOperator: axis must be >= batch dimensions.");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs[0]->GetDataType() != DataType::QAsymmU8,
                                        "ConvertGatherToTosaOperator: Tosa gather does not support unsigned types.");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs[1]->GetDataType() != DataType::Signed64,
                                        "ConvertGatherToTosaOperator: Tosa gather does not support int 64 indices.");

    unsigned int axis = static_cast<unsigned int>(gatherDescriptor->m_Axis);
    unsigned int batchDims = static_cast<unsigned int>(batch_dims);

    std::string inputParamsName = std::string("input_0_params");
    std::string inputIndicesName = std::string("input_1_indices");
    std::string outputTransposeParamsName = std::string("intermediate_0_transpose_params") + GetUniqueTosaMappingID();
    std::string outputReshapeParamsName = std::string("intermediate_1_reshape_params") + GetUniqueTosaMappingID();
    std::string outputReshapeIndicesName = std::string("intermediate_2_reshape_indices") + GetUniqueTosaMappingID();
    std::string outputGatherName = std::string("intermediate_3_gather") + GetUniqueTosaMappingID();
    std::string outputReshapeGatherName = std::string("intermediate_4_reshape_gather") + GetUniqueTosaMappingID();
    std::string outputName = std::string("output_0");

    std::string blockName  = std::string("Op_GATHER_block_") + GetUniqueTosaMappingID();

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer)
    {
        // Get the layer connected to the input slot and determine unique tensor names.
        inputParamsName = GenerateUniqueInputName(layer->GetInputSlot(0));
        inputIndicesName = GenerateUniqueInputName(layer->GetInputSlot(1));
        outputName = GenerateUniqueOutputName(*layer);
    }

    auto inputParamsDType = ArmNNToDType(inputs[0]->GetDataType());
    auto inputIndicesDType = ArmNNToDType(inputs[1]->GetDataType());

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputParamsName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputParamsShape = GetTosaTensorShape(inputs[0]->GetShape());
        tensors.push_back(new TosaSerializationTensor(inputParamsName, inputParamsShape, inputParamsDType, {}));
    }
    if(inputIndicesName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputIndicesShape = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.push_back(new TosaSerializationTensor(inputIndicesName, inputIndicesShape, inputIndicesDType, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    std::vector<int32_t> paramsShape = GetTosaTensorShape(inputs[0]->GetShape());
    std::vector<int32_t> indicesShape = GetTosaTensorShape(inputs[1]->GetShape());

    // Parameters needed to calculate output shapes and transpose permutations
    std::vector<int32_t> paramsBatch;
    std::vector<int32_t> paramsIndices;
    std::vector<int32_t> paramsLeftChannels;
    std::vector<int32_t> paramsRightChannels;

    std::vector<int32_t> paramsIdxBatch;
    std::vector<int32_t> paramsIdxIndices;
    std::vector<int32_t> paramsIdxLeftChannels;
    std::vector<int32_t> paramsIdxRightChannels;

    for (unsigned int i = 0; i < paramsRank; i++)
    {
        if (i < batchDims && i < axis)
        {
            paramsBatch.push_back(paramsShape[i]);
            paramsIdxBatch.push_back(static_cast<int32_t>(i));
        }
        else if (i < axis)
        {
            paramsLeftChannels.push_back(paramsShape[i]);
            paramsIdxLeftChannels.push_back(static_cast<int32_t>(i));
        }
        else if (i < (axis + 1))
        {
            paramsIndices.push_back(paramsShape[i]);
            paramsIdxIndices.push_back(static_cast<int32_t>(i));
        }
        else
        {
            paramsRightChannels.push_back(paramsShape[i]);
            paramsIdxRightChannels.push_back(static_cast<int32_t>(i));
        }
    }

    // Calculate N, K, W, C
    // N: number of batches
    // W: number of indices in each batch
    // K: range of each index
    // C: number of channels for each index
    std::vector<int32_t> paramsLow;
    std::vector<int32_t> paramsMid;
    std::vector<int32_t> paramsHigh;
    std::vector<int32_t> indicesMid;

    // Copy the first batchDims number of paramsShape values to paramsLow
    for (unsigned int i = 0; i < batchDims; i++)
    {
        paramsLow.push_back(paramsShape[i]);
    }
    // Starting at batchDims index, copy the next (axis - batchDims) number of paramsShape values to paramsMid
    for (unsigned int i = 0; i < (axis - batchDims); i++)
    {
        paramsMid.push_back(paramsShape[batchDims + i]);
    }
    // Starting at (axis + 1) index, copy the next (paramsRank - axis - 1) number of paramsShape values to paramsHigh
    for (unsigned int i = 0; i < (paramsRank - axis - 1); i++)
    {
        paramsHigh.push_back(paramsShape[axis + 1 + i]);
    }
    // Starting at batchDims index, copy the next (indicesRank - batchDims) number of indicesShape values to indicesMid
    for (unsigned int i = 0; i < (indicesRank - batchDims); i++)
    {
        indicesMid.push_back(indicesShape[batchDims + i]);
    }

    auto lowProduct = static_cast<int32_t>(std::accumulate(std::begin(paramsMid),
                                                           std::end(paramsMid),
                                                           1,
                                                           std::multiplies<>() ));
    auto highProduct = static_cast<int32_t>(std::accumulate(std::begin(paramsHigh),
                                                            std::end(paramsHigh),
                                                            1,
                                                            std::multiplies<>() ));

    auto N = static_cast<int32_t>(std::accumulate(std::begin(paramsLow),
                                                  std::end(paramsLow),
                                                  1,
                                                  std::multiplies<>() ));
    auto W = static_cast<int32_t>(std::accumulate(std::begin(indicesMid),
                                                  std::end(indicesMid),
                                                  1,
                                                  std::multiplies<>() ));
    auto K = paramsShape[axis];
    auto C = lowProduct * highProduct;

    // Parameters needed for input transpose
    std::vector<int32_t> inputTransposePermutation;
    std::vector<int32_t> inputTransposeShape;
    for (unsigned int i = 0; i < paramsBatch.size(); i++)
    {
        inputTransposePermutation.push_back(paramsIdxBatch[i]);
        inputTransposeShape.push_back(paramsBatch[i]);
    }
    for (unsigned int i = 0; i < paramsIndices.size(); i++)
    {
        inputTransposePermutation.push_back(paramsIdxIndices[i]);
        inputTransposeShape.push_back(paramsIndices[i]);
    }
    for (unsigned int i = 0; i < paramsLeftChannels.size(); i++)
    {
        inputTransposePermutation.push_back(paramsIdxLeftChannels[i]);
        inputTransposeShape.push_back(paramsLeftChannels[i]);
    }
    for (unsigned int i = 0; i < paramsRightChannels.size(); i++)
    {
        inputTransposePermutation.push_back(paramsIdxRightChannels[i]);
        inputTransposeShape.push_back(paramsRightChannels[i]);
    }

    // Parameters needed for result/output transpose
    std::vector<int32_t> resultReshapeShape;
    resultReshapeShape.insert(resultReshapeShape.end(), indicesShape.begin(), indicesShape.end());
    resultReshapeShape.insert(resultReshapeShape.end(), paramsLeftChannels.begin(), paramsLeftChannels.end());
    resultReshapeShape.insert(resultReshapeShape.end(), paramsRightChannels.begin(), paramsRightChannels.end());

    std::vector<int32_t> resultTransposePerm;
    for (unsigned int i = 0; i < batchDims; i++)
    {
        resultTransposePerm.push_back(static_cast<int32_t>(i));
    }
    for (unsigned int i = 0; i < paramsLeftChannels.size(); i++)
    {
        resultTransposePerm.push_back(static_cast<int32_t>(i + inputs[1]->GetNumDimensions()));
    }
    for (unsigned int i = batchDims; i < inputs[1]->GetNumDimensions(); i++)
    {
        resultTransposePerm.push_back(static_cast<int32_t>(i));
    }
    for (unsigned int i = 0; i < paramsRightChannels.size(); i++)
    {
        resultTransposePerm.push_back(static_cast<int32_t>(i + inputs[1]->GetNumDimensions() +
                                                           paramsLeftChannels.size()));
    }

    std::vector<int32_t> tosaValuesShape = {N, K, C};
    std::vector<int32_t> tosaIndicesShape = {N, W};
    std::vector<int32_t> tosaGatherResultShape = {N, W, C};

    // 1. Transpose params values.  This operation is only need if the axis is not 0.
    if (axis > 0)
    {
        tensors.emplace_back(new TosaSerializationTensor(outputTransposeParamsName,
                                                         inputTransposeShape,
                                                         inputParamsDType,
                                                         {}));

        TosaTransposeAttribute transposeInputAttribute(inputTransposePermutation);

        auto *transposeInputOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                               Attribute_TransposeAttribute,
                                                               &transposeInputAttribute,
                                                               {inputParamsName},
                                                               {outputTransposeParamsName});
        operators.push_back(transposeInputOp);
    }

    // 2. Reshape params
    std::string& reshapeOpInputParamsName = axis > 0 ? outputTransposeParamsName : inputParamsName;

    tensors.emplace_back(new TosaSerializationTensor(outputReshapeParamsName,
                                                     tosaValuesShape,
                                                     inputParamsDType,
                                                     {}));

    TosaReshapeAttribute reshapeValuesAttribute(tosaValuesShape);

    auto* reshapeInputParamsOp = new TosaSerializationOperator(Op_RESHAPE,
                                                               Attribute_ReshapeAttribute,
                                                               &reshapeValuesAttribute,
                                                               {reshapeOpInputParamsName},
                                                               {outputReshapeParamsName});
    operators.push_back(reshapeInputParamsOp);

    // 3. Reshape indices
    tensors.emplace_back(new TosaSerializationTensor(outputReshapeIndicesName,
                                                     tosaIndicesShape,
                                                     inputIndicesDType,
                                                     {}));

    TosaReshapeAttribute reshapeIndicesAttribute(tosaIndicesShape);

    auto* reshapeInputIndicesOp = new TosaSerializationOperator(Op_RESHAPE,
                                                                Attribute_ReshapeAttribute,
                                                                &reshapeIndicesAttribute,
                                                                {inputIndicesName},
                                                                {outputReshapeIndicesName});
    operators.push_back(reshapeInputIndicesOp);

    // 4. Gather params, indices
    tensors.emplace_back(new TosaSerializationTensor(outputGatherName,
                                                     tosaGatherResultShape,
                                                     inputParamsDType,
                                                     {}));

    auto* gatherOp = new TosaSerializationOperator(Op_GATHER,
                                                   Attribute_NONE,
                                                   nullptr,
                                                   {outputReshapeParamsName, outputReshapeIndicesName},
                                                   {outputGatherName});
    operators.push_back(gatherOp);

    // 5. Reshape gather output
    if (axis > 0)
    {
        // If a Transpose op is needed below, an additional tensor is needed to store the reshape output.
        tensors.emplace_back(new TosaSerializationTensor(outputReshapeGatherName,
                                                         resultReshapeShape,
                                                         outputDType0,
                                                         {}));
    }

    std::string& reshapeOpOutputName = axis > 0 ? outputReshapeGatherName : outputName;

    TosaReshapeAttribute reshapeGatherAttribute(resultReshapeShape);

    auto* reshapeGatherOutputOp = new TosaSerializationOperator(Op_RESHAPE,
                                                               Attribute_ReshapeAttribute,
                                                               &reshapeGatherAttribute,
                                                               {outputGatherName},
                                                               {reshapeOpOutputName});
    operators.push_back(reshapeGatherOutputOp);

    // 6. Transpose result output.  This operator is only needed if the axis is not 0
    if (axis > 0)
    {
        TosaTransposeAttribute transposeOutputAttribute(resultTransposePerm);

        auto* transposeOutputOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                                Attribute_TransposeAttribute,
                                                                &transposeOutputAttribute,
                                                                {outputReshapeGatherName},
                                                                {outputName});
        operators.push_back(transposeOutputOp);
    }

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName,     // name
                                           mainName,      // region name
                                           operators,     // operators
                                           tensors,       // tensors
                                           {inputParamsName, inputIndicesName},    // inputs
                                           {outputName}); // outputs
}
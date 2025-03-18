//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "BatchMatMulOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLBatchMatMulOp
TosaSerializationBasicBlock* ConvertBatchMatMulToTosaOperator(const Layer* layer,
                                                              const std::vector<const TensorInfo*>& inputs,
                                                              const std::vector<const TensorInfo*>& outputs,
                                                              const BatchMatMulDescriptor* descriptor)
{
    if (descriptor->m_AdjointX || descriptor->m_AdjointY )
    {
        throw Exception("Support for adjoint not implemented.");
    }
    if (descriptor->m_DataLayoutX != armnn::DataLayout::NCHW || descriptor->m_DataLayoutY != armnn::DataLayout::NCHW )
    {
        throw Exception("MatMul only supported in the last 2 dimensions");
    }

    std::string input0Name = std::string("input_0");
    std::string input1Name = std::string("input_1");
    std::string outputName = std::string("output_0");
    std::string outputReshape0Name = std::string("layer_intermediate0_") + GetUniqueTosaMappingID();
    std::string outputReshape1Name = std::string("layer_intermediate0_") + GetUniqueTosaMappingID();
    std::string outputTranspose0Name = std::string("layer_intermediate1_") + GetUniqueTosaMappingID();
    std::string outputTranspose1Name = std::string("layer_intermediate1_") + GetUniqueTosaMappingID();

    std::string blockName  = std::string("Op_BATCHMATMUL_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layer connected to the input slot and determine unique tensor names.
        input0Name = GenerateUniqueInputName(layer->GetInputSlot(0));
        input1Name = GenerateUniqueInputName(layer->GetInputSlot(1));
        outputName = GenerateUniqueOutputName(*layer);
    }

    // Assumes both input types are same data type
    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());
    bool isInputInt8 = (inputDType == DType_INT8);
    bool isInputInt16 = (inputDType == DType_INT16);

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(input0Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        tensors.push_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType, {}));
    }
    if(input1Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.push_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType, {}));
    }

    std::string input0TransposeName = input0Name;
    std::string input1TransposeName = input1Name;
    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());

    std::string input0MatMulName = input0Name;
    std::string input1MatMulName = input1Name;

    // *** ADD OP STEPS ***

    // ADD a RESHAPE OPs if BATCH DIMS > 1
    // RESHAPE input 1
    std::vector<int32_t> targetShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    uint32_t input0Dimensions = inputs[0]->GetNumDimensions();
    if (input0Dimensions > 3)
    {
        uint32_t x = 1;
        for (uint32_t i = 0; i < (input0Dimensions - 2); ++i)
        {
            x *=(inputs[0]->GetShape()[i]);
        }

        targetShape0 = {static_cast<int32_t>(x),
                        static_cast<int32_t>(inputs[0]->GetShape()[input0Dimensions - 2]),
                        static_cast<int32_t>(inputs[0]->GetShape()[input0Dimensions - 1])};

        TosaReshapeAttribute attribute(targetShape0);

        auto* input0ReshapeOp = new TosaSerializationOperator(Op_RESHAPE,
                                                              Attribute_ReshapeAttribute,
                                                              &attribute,
                                                              {input0Name},
                                                              {outputReshape0Name});

        operators.push_back(input0ReshapeOp);
        tensors.push_back(new TosaSerializationTensor(outputReshape0Name, targetShape0, inputDType, {}));
        input0TransposeName = outputReshape0Name;
        input0MatMulName = outputReshape0Name;
    }

    // RESHAPE input 2
    std::vector<int32_t> targetShape1 = GetTosaTensorShape(outputs[0]->GetShape());
    uint32_t input1Dimensions = inputs[1]->GetNumDimensions();
    if (input1Dimensions > 3)
    {
        uint32_t x = 1;
        for (uint32_t i = 0; i < (input1Dimensions - 2); i++)
        {
            x *= (inputs[1]->GetShape()[i]);
        }

        targetShape1 = {static_cast<int32_t>(x),
                        static_cast<int32_t>(inputs[1]->GetShape()[input1Dimensions - 2]),
                        static_cast<int32_t>(inputs[1]->GetShape()[input1Dimensions - 1])};

        TosaReshapeAttribute attribute(targetShape1);

        auto* input1ReshapeOp = new TosaSerializationOperator(Op_RESHAPE,
                                                              Attribute_ReshapeAttribute,
                                                              &attribute,
                                                              {input1Name},
                                                              {outputReshape1Name});

        operators.push_back(input1ReshapeOp);
        tensors.push_back(new TosaSerializationTensor(outputReshape1Name, targetShape1, inputDType, {}));
        input1TransposeName = outputReshape1Name;
        input1MatMulName = outputReshape1Name;
    }
    bool needsReshape = input0Dimensions > 3 || input1Dimensions > 3;

    // ADD a TRANSPOSE OP for one/both inputs if transpose set to true
    if (descriptor->m_TransposeX)
    {
        auto permuteVec = BatchMatMulDescriptor::GetPermuteVec(descriptor->m_DataLayoutX,
                                                               inputs[0]->GetShape());
        std::vector<int32_t> mappings(permuteVec.begin(),
                                      permuteVec.end());
        if (input0Dimensions > 3)
        {
            auto input0BatchedDims = input0Dimensions - 3;
            mappings = {static_cast<int>(permuteVec[0]),
                        static_cast<int>(permuteVec[input0Dimensions - 2] - input0BatchedDims),
                        static_cast<int>(permuteVec[input0Dimensions - 1] - input0BatchedDims)};
        }

        TosaTransposeAttribute transposeAttribute(mappings);

        TosaSerializationOperator *transposeOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                                               Attribute_TransposeAttribute,
                                                                               &transposeAttribute,
                                                                               {input0TransposeName},
                                                                               {outputTranspose0Name});

        std::vector<int32_t> transpose0Shape =
        {
            targetShape0[static_cast<unsigned int>(mappings[0])],
            targetShape0[static_cast<unsigned int>(mappings[1])],
            targetShape0[static_cast<unsigned int>(mappings[2])]
        };

        operators.push_back(transposeOp);
        tensors.push_back(new TosaSerializationTensor(outputTranspose0Name, transpose0Shape, inputDType, {}));
        input0MatMulName = outputTranspose0Name;
    }

    if (descriptor->m_TransposeY)
    {
        auto permuteVec = BatchMatMulDescriptor::GetPermuteVec(descriptor->m_DataLayoutY,
                                                               inputs[1]->GetShape());

        std::vector<int32_t> mappings(permuteVec.begin(),
                                      permuteVec.end());

        auto input1BatchedDims = input1Dimensions - 3;
        if (input1Dimensions > 3)
        {
            mappings = {static_cast<int>(permuteVec[0]),
                        static_cast<int>(permuteVec[input1Dimensions - 2] - input1BatchedDims),
                        static_cast<int>(permuteVec[input1Dimensions - 1] - input1BatchedDims)};
        }

        TosaTransposeAttribute transposeAttribute(mappings);

        TosaSerializationOperator *transposeOp = new TosaSerializationOperator(Op_TRANSPOSE,
                                                                               Attribute_TransposeAttribute,
                                                                               &transposeAttribute,
                                                                               {input1TransposeName},
                                                                               {outputTranspose1Name});
        std::vector<int32_t> transpose1Shape =
        {
            targetShape1[static_cast<unsigned int>(mappings[0])],
            targetShape1[static_cast<unsigned int>(mappings[1])],
            targetShape1[static_cast<unsigned int>(mappings[2])]
        };

        operators.push_back(transposeOp);
        tensors.push_back(new TosaSerializationTensor(outputTranspose1Name, transpose1Shape, inputDType, {}));
        input1MatMulName = outputTranspose1Name;
    }

    // ADD MAT MUL layer
    std::string matMulOutputStr = needsReshape || isInputInt8 || isInputInt16 ?
                                  std::string("layer_intermediate2_") + GetUniqueTosaMappingID() : outputName;

    TosaMatMulAttribute matMulAttribute(0,0); // input0_zp, input1_zp
    DType matMulOutDType = ArmNNToDType(inputs[1]->GetDataType());
    if (isInputInt8)
    {
        matMulAttribute = TosaMatMulAttribute(inputs[0]->GetQuantizationOffset(),  inputs[1]->GetQuantizationOffset());
        matMulOutDType = DType_INT32;
    }
    if (isInputInt16)
    {
        matMulAttribute = TosaMatMulAttribute(inputs[0]->GetQuantizationOffset(),  inputs[1]->GetQuantizationOffset());
        matMulOutDType = DType_INT48;
    }
    TosaSerializationOperator* matMulOp = new TosaSerializationOperator(Op_MATMUL,
                                                                        Attribute_MatMulAttribute,
                                                                        &matMulAttribute,
                                                                        {input0MatMulName, input1MatMulName},
                                                                        {matMulOutputStr});

    uint32_t outputDimensions = outputs[0]->GetNumDimensions();
    if (outputDimensions > 3)
    {
        uint32_t x = 1;
        for (uint32_t i = 0; i < (outputDimensions - 2); ++i)
        {
            x *=(outputs[0]->GetShape()[i]);
        }

        outputShape0 = {static_cast<int32_t>(x),
                        static_cast<int32_t>(outputs[0]->GetShape()[outputDimensions - 2]),
                        static_cast<int32_t>(outputs[0]->GetShape()[outputDimensions - 1])};
    }

    operators.push_back(matMulOp);
    tensors.push_back(new TosaSerializationTensor(matMulOutputStr, outputShape0, matMulOutDType, {}));

    std::string outputRescale = needsReshape ?
                                     std::string("layer_intermediate3_") + GetUniqueTosaMappingID() : outputName;
    std::string inputReshape2Name = isInputInt8 ||  isInputInt16 ? outputRescale : matMulOutputStr;

    // ADD Rescale layer if it is int8
    if (isInputInt8 || isInputInt16)
    {
        bool scale32 = isInputInt16 ? false : true;
        bool doubleRound = isInputInt16 ? false : true;

        int32_t output_zp  = outputs[0]->GetQuantizationOffset();
        double output_scale = outputs[0]->GetQuantizationScales()[0];
        double input_scale = inputs[0]->GetQuantizationScales()[0];
        const std::vector<float>& weight_scales = inputs[1]->GetQuantizationScales();

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperatorForWeights(matMulOutputStr,
                                            outputRescale,
                                            0,
                                            output_zp,
                                            false,
                                            false,
                                            doubleRound,
                                            scale32,
                                            input_scale,
                                            output_scale,
                                            weight_scales,
                                            &rescaleOp);

        tensors.push_back(new TosaSerializationTensor(outputRescale,
                                                      outputShape0,
                                                      inputDType, {}));

        operators.push_back(rescaleOp);
    }

    // ADD a RESHAPE back to expected rank
    if (needsReshape)
    {
        const std::vector<int32_t>& targetShape = GetTosaTensorShape(TensorShape(outputs[0]->GetShape()));
        TosaReshapeAttribute attribute(targetShape);

        auto* outputReshapeOp = new TosaSerializationOperator(Op_RESHAPE,
                                                              Attribute_ReshapeAttribute,
                                                              &attribute,
                                                              {inputReshape2Name},
                                                              {outputName});

        operators.push_back(outputReshapeOp);
        tensors.push_back(new TosaSerializationTensor(outputName, targetShape, inputDType, {}));
    }

    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {operators}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}


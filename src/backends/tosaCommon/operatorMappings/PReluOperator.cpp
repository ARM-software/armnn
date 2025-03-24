//
// Copyright © 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "PReluOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

#include <layers/ActivationLayer.hpp>

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLPReluOp
TosaSerializationBasicBlock* ConvertPReluToTosaOperator(const Layer* layer,
                                                        const std::vector<const TensorInfo*>& inputs,
                                                        const std::vector<const TensorInfo*>& outputs)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(IsQuantized8BitType(inputs[0]->GetDataType()) &&
                                                IsQuantized8BitType(outputs[0]->GetDataType()),
                                    "ConvertPReluToTosaOperator: Prelu currently only supports Int8 Quantized inputs");
    if (inputs.size() != 2) {
        throw armnn::Exception("ConvertPReluToTosaOperator: 2 input tensors required.");
    }

    if (outputs.size() != 1) {
        throw armnn::Exception("ConvertPReluToTosaOperator: 1 output tensor required.");
    }

    std::string inputName0 = std::string("input_0");
    std::string inputName1 = std::string("input_1");
    std::string inputNameZero = std::string("constant1_") + GetUniqueTosaMappingID();
    std::string outputNameMul = std::string("layer_intermediate2_") + GetUniqueTosaMappingID();
    std::string outputNameClampEq = std::string("layer_intermediate3_") + GetUniqueTosaMappingID();
    std::string outputNameRescaleTo32 = std::string("layer_intermediate4_") + GetUniqueTosaMappingID();
    std::string outputNameRescaleSlope = std::string("layer_intermediate5_") + GetUniqueTosaMappingID();
    std::string outputNameRescaleSlope2 = std::string("layer_intermediate6_") + GetUniqueTosaMappingID();
    std::string outputNameRescaleIdentity = std::string("layer_intermediate7_") + GetUniqueTosaMappingID();
    std::string outputNameClamp = std::string("layer_intermediate8_") + GetUniqueTosaMappingID();
    std::string outputNameClampTo32 = std::string("layer_intermediate9_") + GetUniqueTosaMappingID();
    std::string outputName = std::string("output0_");
    std::string blockName = std::string("Op_PRELU_block_") + GetUniqueTosaMappingID();

    double scale_alpha = inputs[0]->GetQuantizationScale() * inputs[1]->GetQuantizationScale() /
                         outputs[0]->GetQuantizationScale();
    double scale_identity = inputs[0]->GetQuantizationScale() / outputs[0]->GetQuantizationScale();
    int32_t input0_zp = inputs[0]->GetQuantizationOffset();
    int32_t input1_zp = inputs[1]->GetQuantizationOffset();
    int32_t output_zp = outputs[0]->GetQuantizationOffset();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr) {
        inputName0 = GenerateUniqueInputName(layer->GetInputSlot(0));
        inputName1 = GenerateUniqueInputName(layer->GetInputSlot(1));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor *> tensors;
    std::vector<TosaSerializationOperator *> operators;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    std::vector<int32_t> inputShape0;
    inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    if (inputName0.find("input_") != std::string::npos) {
        tensors.push_back(new TosaSerializationTensor(inputName0, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> inputShape1;
    inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
    if (inputName1.find("input_") != std::string::npos) {
        tensors.push_back(new TosaSerializationTensor(inputName1, inputShape1, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Implement PReLU as:
    //   rescaled_in = rescale(in)
    //   rescaled_alpha = rescale(alpha)
    //   rescaled_identity_in = rescale(in, scale_identity)
    //   slope_in = mul(rescaled_in, rescaled_alpha)
    //   rescaled_slope_in = rescale(slope_in, scale_alpha)
    //   cond_result = greater_equal(rescaled_in, 0)
    //   output = select(cond_result, rescaled_identity_in, rescaled_slope_in)

    TosaSerializationOperator* op_rescale_in = nullptr;
    CreateRescaleTosaOperator(inputName0,
                              outputNameRescaleTo32,
                              1.0f,
                              input0_zp,
                              0,
                              false,
                              false,
                              true,
                              true,
                              &op_rescale_in);
    operators.emplace_back(op_rescale_in);
    tensors.emplace_back(new TosaSerializationTensor(outputNameRescaleTo32,
                                                     inputShape0,
                                                     DType_INT32,
                                                     {}));

    int32_t clamp_min = 0;
    int32_t clamp_max = std::numeric_limits<int8_t>::max();
    float float_max = std::numeric_limits<float>::max();

    // If CLAMP result matches original then no negative values
    // CLAMP does not support INT32
    TosaClampAttribute attribute(clamp_min, clamp_max, 0, float_max);
    auto* clamp_op = new TosaSerializationOperator(Op_CLAMP,
                                                   Attribute_ClampAttribute,
                                                   &attribute,
                                                   {inputName0},
                                                   {outputNameClamp});
    operators.push_back(clamp_op);
    tensors.push_back(new TosaSerializationTensor(outputNameClamp, inputShape0, inputDType0, {}));

    // EQUAL does not support INT8 so RESCALE CLAMP output required
    TosaSerializationOperator* op_clamp = nullptr;
    CreateRescaleTosaOperator(outputNameClamp,
                              outputNameClampTo32,
                              1.0f,
                              input0_zp,
                              0,
                              false,
                              false,
                              true,
                              true,
                              &op_clamp);
    operators.emplace_back(op_clamp);
    tensors.emplace_back(new TosaSerializationTensor(outputNameClampTo32,
                                                     inputShape0,
                                                     DType_INT32,
                                                     {}));

    TosaSerializationOperator *op_ge = new TosaSerializationOperator(Op_EQUAL,
                                                                     Attribute_NONE,
                                                                     nullptr,
                                                                     {outputNameRescaleTo32, outputNameClampTo32},
                                                                     {outputNameClampEq});
    operators.push_back(op_ge);
    tensors.push_back(new TosaSerializationTensor(outputNameClampEq, inputShape0, DType_BOOL, {}));

    // RESHAPE for outputNameRescaleSlope not required as already done on input2 by AddBroadCastReshapeLayer
    TosaSerializationOperator* op_rescale_slope_in = nullptr;
    CreateRescaleTosaOperator(inputName1,
                              outputNameRescaleSlope,
                              1.0f,
                              input1_zp,
                              0,
                              false,
                              false,
                              true,
                              true,
                              &op_rescale_slope_in);
    operators.emplace_back(op_rescale_slope_in);
    tensors.emplace_back(new TosaSerializationTensor(outputNameRescaleSlope,
                                                     inputShape1,
                                                     DType_INT32,
                                                     {}));
    // mul shift
    int32_t shift = 0;
    TosaMulAttribute mulAttribute(shift);
    TosaSerializationOperator *mulOp = new TosaSerializationOperator(Op_MUL,
                                                                     Attribute_MulAttribute,
                                                                     &mulAttribute,
                                                                     {outputNameRescaleTo32,
                                                                      outputNameRescaleSlope},
                                                                     {outputNameMul});
    operators.push_back(mulOp);
    tensors.push_back(new TosaSerializationTensor(outputNameMul, outputShape0, DType_INT32, {}));

    TosaSerializationOperator *op_rescale_slope_in2 = nullptr;
    CreateRescaleTosaOperator(outputNameMul,
                              outputNameRescaleSlope2,
                              scale_alpha,
                              0,
                              output_zp,
                              false,
                              false,
                              true,
                              true,
                              &op_rescale_slope_in2);
    operators.push_back(op_rescale_slope_in2);
    tensors.push_back(new TosaSerializationTensor(outputNameRescaleSlope2,
                                                  outputShape0,
                                                  outputDType0, {}));

    TosaSerializationOperator *op_rescale_identity_in = nullptr;
    CreateRescaleTosaOperator(inputName0,
                              outputNameRescaleIdentity,
                              scale_identity,
                              input0_zp,
                              output_zp,
                              false,
                              false,
                              true,
                              true,
                              &op_rescale_identity_in);
    operators.push_back(op_rescale_identity_in);
    tensors.push_back(new TosaSerializationTensor(outputNameRescaleIdentity,
                                                  outputShape0,
                                                  outputDType0, {}));

    TosaSerializationOperator *selectOp = new TosaSerializationOperator(Op_SELECT,
                                                                        Attribute_NONE,
                                                                        nullptr,
                                                                        {outputNameClampEq,
                                                                         outputNameRescaleIdentity,
                                                                         outputNameRescaleSlope2},
                                                                        {outputName});
    operators.push_back(selectOp);
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,              // name
                                           mainName,               // region name
                                           operators,              // operators
                                           tensors,                // tensors
                                           {inputName0,inputName1},// inputs
                                           {outputName});          // outputs

}
//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "ActivationOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

#include <layers/ActivationLayer.hpp>

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLLeakyReluOp
TosaSerializationBasicBlock* ConvertActivationToTosaOperator(const Layer* layer,
                                                             const std::vector<const TensorInfo*>& inputs,
                                                             const std::vector<const TensorInfo*>& outputs,
                                                             const ActivationDescriptor* activationDescriptor)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertActivationToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertActivationToTosaOperator: 1 output tensor required.");
    }

    std::string inputName       = std::string("input0_");
    std::string outputNameAlpha = std::string("intermediate1_") + GetUniqueTosaMappingID();
    std::string outputNameMul   = std::string("intermediate2_") + GetUniqueTosaMappingID();
    std::string outputName      = std::string("output0_");
    std::string blockName       = std::string("Op_ACTIVATION_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensors names.
        Layer& connectedInputLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        inputName = GenerateUniqueName(connectedInputLayer, 0);

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer, 0);
    }

    std::vector<TosaSerializationTensor*> tensors;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    std::vector<int32_t> inputShape0;
    DType inputDType0 =  DType::DType_UNKNOWN;
    if(inputName.find("input0_") != std::string::npos)
    {
        inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

#if TOSA_COMPAT_VERSION(0, 60, 0)
    std::string outputNameMAXMIN= std::string("intermediate3_") + GetUniqueTosaMappingID();

    if (inputDType0 == DType::DType_FP32 ||
        inputDType0 == DType::DType_FP16)
    {
        // const_alpha
        TosaSerializationOperator* alphaOp = nullptr;
        TosaSerializationTensor* alphaTensor = nullptr;
        CreateConstTosaOperator<float>(outputNameAlpha,
                                       activationDescriptor->m_A,
                                       inputDType0,
                                       inputShape0,
                                       alphaOp,
                                       alphaTensor);
        tensors.push_back(alphaTensor);

        // mul
        int32_t shift = 0;
        TosaMulAttribute mulAttribute(shift);
        TosaSerializationOperator* mulOp = new TosaSerializationOperator(Op_MUL,
                                                                         Attribute_MulAttribute,
                                                                         &mulAttribute,
                                                                         {inputName, outputNameAlpha},
                                                                         {outputNameMul});
        tensors.push_back(new TosaSerializationTensor(outputNameMul, inputShape0, inputDType0, {}));

        TosaSerializationOperator* op = nullptr;
        if (activationDescriptor->m_A <= 1.0)
        {
            op = new TosaSerializationOperator(Op_MAXIMUM,
                                               Attribute_NONE,
                                               nullptr,
                                               {inputName, outputNameMul},
                                               {outputName});
        }
        else
        {
            op = new TosaSerializationOperator(Op_MINIMUM,
                                               Attribute_NONE,
                                               nullptr,
                                               {inputName, outputNameMul},
                                               {outputName});

        }

        // operatorInputNames/operatorOutputNames ends up being the same as
        // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
        return new TosaSerializationBasicBlock(blockName,              // name
                                               mainName,               // region name
                                               {alphaOp, mulOp, op},   // operators
                                               tensors,                // tensors
                                               {inputName},            // inputs
                                               {outputName});          // outputs
    }
    else
    {
        std::string outputNameRescaleAlpha      = std::string("intermediate3_") + GetUniqueTosaMappingID();
        std::string outputNameRescaleIdentity   = std::string("intermediate4_") + GetUniqueTosaMappingID();
        std::string outputNameRescaleMaxMin     = std::string("intermediate5_") + GetUniqueTosaMappingID();

        DType rescale_type      = DType::DType_INT32;
        float alpha             = activationDescriptor->m_A;
        double scale_alpha      = inputs[0]->GetQuantizationScale() * alpha / outputs[0]->GetQuantizationScale();
        double scale_identity   = inputs[0]->GetQuantizationScale() / outputs[0]->GetQuantizationScale();
        int32_t input_zp        = inputs[0]->GetQuantizationOffset();
        int32_t output_zp       = outputs[0]->GetQuantizationOffset();

        // Value op_rescale_alpha_in =
        //        buildRescale(rewriter, op, rescale_type, input, scale_alpha,
        //                     input_qtype.getZeroPoint(), 0, true, true);
        TosaSerializationOperator* rescaleAlphaOp = nullptr;
        TosaSerializationTensor* rescaleAlphaTensor = nullptr;
        CreateRescaleTosaOperator(inputName,
                                  outputNameRescaleAlpha,
                                  rescale_type,
                                  inputShape0,
                                  scale_alpha,
                                  input_zp,
                                  0,
                                  true,
                                  true,
                                  &rescaleAlphaOp,
                                  &rescaleAlphaTensor);
        tensors.push_back(rescaleAlphaTensor);

        // Value op_rescale_identity_in =
        //       buildRescale(rewriter, op, rescale_type, input, scale_identity,
        //                    input_qtype.getZeroPoint(), 0, true, true);
        TosaSerializationOperator* rescaleIdentityOp = nullptr;
        TosaSerializationTensor* rescaleIdentityTensor = nullptr;
        CreateRescaleTosaOperator(inputName,
                                  outputNameRescaleIdentity,
                                  rescale_type,
                                  inputShape0,
                                  scale_identity,
                                  input_zp,
                                  0,
                                  true,
                                  true,
                                  &rescaleIdentityOp,
                                  &rescaleIdentityTensor);
        tensors.push_back(rescaleIdentityTensor);

        // Value result_int32;
        // if (alpha <= 1.0) {
        //    auto max_op = CreateOpAndInfer<tosa::MaximumOp>(
        //            rewriter, op->getLoc(), rescale_type, op_rescale_identity_in,
        //            op_rescale_alpha_in);
        //    result_int32 = max_op.getResult();
        // } else {
        //    auto min_op = CreateOpAndInfer<tosa::MinimumOp>(
        //            rewriter, op->getLoc(), rescale_type, op_rescale_identity_in,
        //            op_rescale_alpha_in);
        //    result_int32 = min_op.getResult();
        // }
        TosaSerializationOperator* op = nullptr;
        if (alpha <= 1.0)
        {
            op = new TosaSerializationOperator(Op_MAXIMUM,
                                               Attribute_NONE,
                                               nullptr,
                                               {outputNameRescaleAlpha, outputNameRescaleIdentity},
                                               {outputNameRescaleMaxMin});
        }
        else
        {
            op = new TosaSerializationOperator(Op_MINIMUM,
                                               Attribute_NONE,
                                               nullptr,
                                               {outputNameRescaleAlpha, outputNameRescaleIdentity},
                                               {outputNameRescaleMaxMin});

        }
        tensors.push_back(new TosaSerializationTensor(outputNameRescaleMaxMin, inputShape0, rescale_type, {}));

        // Value output = buildRescaleFromInt32(rewriter, op, output_type, result_int32,
        //                                      1.0, output_qtype.getZeroPoint());
        TosaSerializationOperator* rescaleOutputOp = nullptr;
        CreateFromInt32RescaleTosaOperator(outputNameRescaleMaxMin,
                                           outputName,
                                           outputDType0,
                                           outputShape0,
                                           1.0,
                                           output_zp,
                                           &rescaleOutputOp,
                                           nullptr);

        // operatorInputNames/operatorOutputNames ends up being the same as
        // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
        return new TosaSerializationBasicBlock(blockName,              // name
                                               mainName,               // region name
                                               {rescaleAlphaOp, rescaleIdentityOp, op, rescaleOutputOp}, // operators
                                               tensors,                // tensors
                                               {inputName},            // inputs
                                               {outputName});          // outputs
    }
#else
    std::string outputNameZero  = std::string("intermediate3_") + GetUniqueTosaMappingID();
    std::string outputNameGE    = std::string("intermediate4_") + GetUniqueTosaMappingID();

    // const_zero
    TosaSerializationOperator* zeroOp = nullptr;
    TosaSerializationTensor* zeroTensor = nullptr;
    CreateConstTosaOperator<float>(outputNameZero,
                                   0.0f,
                                   inputDType0,
                                   inputShape0,
                                   zeroOp,
                                   zeroTensor);
    tensors.push_back(zeroTensor);

    // const_alpha
    TosaSerializationOperator* alphaOp = nullptr;
    TosaSerializationTensor* alphaTensor = nullptr;
    CreateConstTosaOperator<float>(outputNameAlpha,
                                   activationDescriptor->m_A,
                                   inputDType0,
                                   inputShape0,
                                   alphaOp,
                                   alphaTensor);
    tensors.push_back(alphaTensor);

    // mul
    int32_t shift = 0;
    TosaMulAttribute mulAttribute(shift);
    TosaSerializationOperator* mulOp = new TosaSerializationOperator(Op_MUL,
                                                                     Attribute_MulAttribute,
                                                                     &mulAttribute,
                                                                     {inputName, outputNameAlpha},
                                                                     {outputNameMul});
    tensors.push_back(new TosaSerializationTensor(outputNameMul, inputShape0, inputDType0, {}));

    // greater_equal
    TosaSerializationOperator* geOp = new TosaSerializationOperator(Op_GREATER_EQUAL,
                                                                    Attribute_NONE,
                                                                    nullptr,
                                                                    {inputName, outputNameZero},
                                                                    {outputNameGE});
    tensors.push_back(new TosaSerializationTensor(outputNameGE, outputShape0, DType::DType_BOOL, {}));

    // select
    TosaSerializationOperator* selOp = new TosaSerializationOperator(Op_SELECT,
                                                                     Attribute_NONE,
                                                                     nullptr,
                                                                     {outputNameGE, inputName, outputNameMul},
                                                                     {outputName});

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,                               // name
                                           mainName,                                // region name
                                           {zeroOp, alphaOp, mulOp, geOp, selOp},   // operators
                                           tensors,                                 // tensors
                                           {inputName},                             // inputs
                                           {outputName});                           // outputs
#endif
}

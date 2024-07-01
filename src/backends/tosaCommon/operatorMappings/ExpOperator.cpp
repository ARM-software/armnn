//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//


#include "ExpOperator.hpp"

#include <cfloat>


// Abstract of getTosaConst8bitTable() function from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc
static std::vector<int16_t> getTosaConst8bitTable(float input_scale,
                                                  int32_t input_zp,
                                                  float output_scale,
                                                  int32_t output_zp,
                                                  std::function<float(float)> func)
{
    // TosaTableAttribute requires int16 vector input. However, TOSA TABLE legalizations are performed using int8.
    std::vector<int16_t> table;
    table.reserve(256);
    float inverse_scale = 1.0f / output_scale;
    for (int32_t i = -128; i < 128; i++)
    {
        float dequantized = input_scale * static_cast<float>(i - input_zp);
        float transformed = func(dequantized);

        float max = (output_scale > 1.0) ? FLT_MAX : (FLT_MAX * output_scale);
        if (transformed >= max)
        {
            table.push_back(INT8_MAX);
            continue;
        }

        int32_t rescaled = static_cast<int32_t>(std::round(transformed * inverse_scale));
        int32_t quantized = static_cast<int32_t>(rescaled + output_zp);
        table.push_back(
            static_cast<int8_t>(std::min(std::max(quantized, -128), 127)));
    }
    return table;
}

// Abstract of getTosaConst16bitTable() function from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc
template <typename FloatT>
static std::vector<int16_t> getTosaConst16bitTable(float input_scale,
                                                   int32_t input_zp,
                                                   float output_scale,
                                                   int32_t output_zp,
                                                   std::function<FloatT(FloatT)> func)
{
    std::vector<int16_t> table;
    table.reserve(513);

    FloatT input_min =
        input_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::min() - input_zp);
    FloatT input_max =
        input_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::max() - input_zp);
    FloatT output_min =
        output_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::min() - output_zp);
    FloatT output_max =
        output_scale * static_cast<FloatT>(std::numeric_limits<int16_t>::max() - output_zp);

    FloatT step = (input_max - input_min) / 512;
    FloatT half_step = step / 2;
    FloatT output_scaling_inv = 65536 / (output_max - output_min);

    for (int32_t i = 0; i < 512; i++)
    {
        FloatT iFloat = static_cast<FloatT>(i);
        FloatT sample_val =
            std::round(func(input_min + (iFloat * step)) * output_scaling_inv);
        FloatT midpoint_interp_val = std::round(
            ((func(input_min + (iFloat + 1) * step) * output_scaling_inv) +
                std::round(func(input_min + (iFloat * step)) * output_scaling_inv)) /
            2);
        FloatT midpoint_val = std::round(func(input_min + (iFloat * step) + half_step) *
                                            output_scaling_inv);
        FloatT midpoint_err = midpoint_interp_val - midpoint_val;
        FloatT bias = std::round(midpoint_err / 2);

        table.push_back(static_cast<int16_t>(
            std::min<FloatT>(std::max<FloatT>(sample_val - bias, -32768), 32767)));
    }

    FloatT max_val = std::round(func(input_max) * output_scaling_inv);
    table.push_back(static_cast<int16_t>(
        std::min<FloatT>(std::max<FloatT>(max_val, -32768), 32767)));
    return table;
}

TosaSerializationBasicBlock* ConvertExpOperator(const Layer* layer,
                                                const std::vector<const TensorInfo*>& inputs,
                                                const std::vector<const TensorInfo*>& outputs,
                                                const ElementwiseUnaryDescriptor* unaryDescriptor)
{
    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_EXP_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    if (unaryDescriptor->m_Operation != UnaryOperation::Exp)
    {
        throw armnn::Exception("ConvertExpOperator: Unsupported elementwise unary operation in descriptor.");
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    float input_scale = inputs[0]->GetQuantizationScale();
    float output_scale = outputs[0]->GetQuantizationScale();
    int32_t input_zp = inputs[0]->GetQuantizationOffset();
    int32_t output_zp = outputs[0]->GetQuantizationOffset();
    DataType inputDType = inputs[0]->GetDataType();
    if (inputDType == DataType::QAsymmS8 ||
        inputDType == DataType::QSymmS8)
    {
        auto exp_func = [](float x) -> float { return std::exp(x); };
        TosaTableAttribute attribute(
            getTosaConst8bitTable(input_scale, input_zp, output_scale, output_zp, exp_func));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QSymmS16)
    {
        throw Exception("ConvertExpOperator() unsupported int 16 not implemented yet.");
        // The following generates the table, tosa attribute and operator for int16 exponential.
        // However, running the int16 EXP EndToEnd test causes incorrect output values.
        // At the time of writing the EXP operator there is no requirment for int16 support.
        // Points to enable int16 in the future:
        //     - TOSA specifies EXP int16 input must have int32 output
        //     - We potentially need a rescale after the int32 EXP output to convert back to int16.
        /*
        auto exp_func = [](float x) -> float { return std::exp(x); };
        TosaTableAttribute attribute(
            getTosaConst16bitTable<float>(input_scale, input_zp, output_scale, output_zp, exp_func));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
        */
    }
    else if (inputDType == DataType::Signed32 ||
             inputDType == DataType::Signed64)
    {
        throw Exception(
            "ConvertExpOperator() unsupported int 32. Only int 8 and int 16 quantized types are supported.");
    }
    // Floating point EXP operator
    else
    {
        operators.push_back(new TosaSerializationOperator(tosa::Op_EXP,
                                                          Attribute_NONE,
                                                          nullptr,
                                                          {inputName},
                                                          {outputName}));
    }

    // Only add input tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputDType);
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());

    // Re-enable below line for int16 EXP support which requires int32 output in TOSA and remove second line.
    // DType outputDType0 =
    //     (inputDType == DataType::QSymmS16) ? DType::DType_INT32 : ArmNNToDType(outputs[0]->GetDataType());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           operators, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "HardSwishOperator.hpp"

#include <gemmlowp/fixedpoint.h>

// This function is paraphrased from:
// tensorflow/lite/kernels/internal/quantization_util.cc QuantizeMultiplier()
static void quantizeMultiplier(double doubleMultiplier, int32_t* quantizedMultiplier, int* shift)
{
    if (doubleMultiplier == 0.)
    {
        *quantizedMultiplier = 0;
        *shift = 0;
        return;
    }

    const double q = std::frexp(doubleMultiplier, shift);
    auto qFixed = static_cast<int64_t>(std::round(q * (1LL << 31)));

    ARMNN_THROW_INVALIDARG_IF_FALSE(qFixed <= (1LL << 31));

    if (qFixed == (1LL << 31))
    {
        qFixed /= 2;
        ++*shift;
    }

    ARMNN_THROW_INVALIDARG_IF_FALSE(qFixed <= std::numeric_limits<int32_t>::max());

    if (*shift < -31)
    {
        *shift = 0;
        qFixed = 0;
    }

    *quantizedMultiplier = static_cast<int32_t>(qFixed);
}

// This function is paraphrased from:
// tensorflow/lite/kernels/internal/reference/hard_swish.h SaturatingDoublingHighMul()
static int16_t saturatingDoublingHighMul(int16_t a, int16_t b)
{
    bool overflow = (a == b && a == std::numeric_limits<int16_t>::min());
    if (overflow)
    {
        return std::numeric_limits<int16_t>::max();
    }

    int32_t a32(a);
    int32_t b32(b);
    int32_t ab32 = a32 * b32;
    int16_t abX2High16 = static_cast<int16_t>((ab32) / (1 << 15));

    return abX2High16;
}

// This function is paraphrased from:
// tensorflow/lite/kernels/internal/common.h DownScaleInt32ToInt16Multiplier()
static void downScaleInt32ToInt16Multiplier(int32_t multiplierInt32, int16_t* multiplierInt16)
{
    ARMNN_THROW_INVALIDARG_IF_FALSE(multiplierInt32 >= 0);

    static constexpr int32_t kRoundingOffset = 1 << 15;
    if (multiplierInt32 >= std::numeric_limits<int32_t>::max() - kRoundingOffset)
    {
        *multiplierInt16 = std::numeric_limits<int16_t>::max();
        return;
    }

    const int32_t result = (multiplierInt32 + kRoundingOffset) >> 16;

    ARMNN_THROW_INVALIDARG_IF_FALSE(result << 16 <= multiplierInt32 + kRoundingOffset);
    ARMNN_THROW_INVALIDARG_IF_FALSE(result << 16 > multiplierInt32 - kRoundingOffset);

    *multiplierInt16 = static_cast<int16_t>(result);

    ARMNN_THROW_INVALIDARG_IF_FALSE(*multiplierInt16 == result);
}

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_utils.cc getTosaConstHardSwish8bitTable()
std::vector<int16_t> getTosaConstHardSwish8bitTable(float inputScale,
                                                    int32_t inputZp,
                                                    float outputScale,
                                                    int32_t outputZp)
{
    const float hiresInputScale = (1.0f / 128.0f) * inputScale;
    const float outputMultiplier = hiresInputScale / outputScale;
    int outputMultiplierExponent;
    int16_t outputMultiplierFixedpointInt16;
    int32_t outputMultiplierFixedpointInt32;

    quantizeMultiplier(outputMultiplier, &outputMultiplierFixedpointInt32, &outputMultiplierExponent);
    downScaleInt32ToInt16Multiplier(outputMultiplierFixedpointInt32, &outputMultiplierFixedpointInt16);

    ARMNN_THROW_INVALIDARG_IF_FALSE(outputMultiplierExponent <= 0);

    const float reluishScale = 3.0f / 32768.0f;
    const float reluishMultiplier = hiresInputScale / reluishScale;
    int reluishMultiplierExponent;
    int16_t reluishMultiplierFixedpointInt16;
    int32_t reluishMultiplierFixedpointInt32;

    quantizeMultiplier(reluishMultiplier, &reluishMultiplierFixedpointInt32, &reluishMultiplierExponent);
    downScaleInt32ToInt16Multiplier(reluishMultiplierFixedpointInt32, &reluishMultiplierFixedpointInt16);

    std::vector<int16_t> table;
    table.reserve(256);
    for (int32_t i = -128; i < 128; i++)
    {
        const int16_t inputValue = static_cast<int16_t>(i - inputZp);
        const int16_t inputValueHiresInputScale = static_cast<int16_t>(inputValue * (1 << 7));

        int16_t reluishValue = inputValueHiresInputScale;
        if (reluishMultiplierExponent > 0)
        {
            reluishValue = gemmlowp::ShiftLeft(reluishValue, reluishMultiplierExponent - 1);
        }

        reluishValue = gemmlowp::SaturatingRoundingDoublingHighMul(reluishValue, reluishMultiplierFixedpointInt16);

        if (reluishMultiplierExponent > 0)
        {
            reluishValue = gemmlowp::ShiftLeft(reluishValue, 1);
        }
        else if (reluishMultiplierExponent < 0)
        {
            reluishValue = gemmlowp::RoundingDivideByPOT(reluishValue, -reluishMultiplierExponent);
        }

        reluishValue = static_cast<int16_t>((reluishValue + (1 << 15)) >> 1);

        const int16_t inputValPreshiftOutputScale =
            gemmlowp::SaturatingRoundingDoublingHighMul(inputValueHiresInputScale, outputMultiplierFixedpointInt16);

        const int16_t preshiftOutputValue = saturatingDoublingHighMul(reluishValue, inputValPreshiftOutputScale);

        int16_t outputValue = gemmlowp::RoundingDivideByPOT(preshiftOutputValue, -outputMultiplierExponent);

        outputValue = static_cast<int16_t>(outputValue + outputZp);
        outputValue = std::min<int16_t>(outputValue, std::numeric_limits<int8_t>::max());
        outputValue = std::max<int16_t>(outputValue, std::numeric_limits<int8_t>::min());

        table.push_back(outputValue);
    }

    return table;
}

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc ConvertTFLHardSwishOp()
TosaSerializationBasicBlock* ConvertHardSwishToTosaOperator(const Layer* layer,
                                                            const std::vector<const TensorInfo*>& inputs,
                                                            const std::vector<const TensorInfo*>& outputs,
                                                            const ActivationDescriptor* desc)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertHardSwishToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertHardSwishToTosaOperator: 1 output tensor required.");
    }

    if (desc->m_Function != ActivationFunction::HardSwish)
    {
        throw armnn::Exception("ConvertHardSwishToTosaOperator ActivationDescriptor only supports function HardSwish.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_HARDSWISH_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    DataType inputDType = inputs[0]->GetDataType();

    bool isInt8 = (inputDType == DataType::QAsymmS8 || inputDType == DataType::QSymmS8);
    if (isInt8)
    {
        float inputScale = inputs[0]->GetQuantizationScale();
        float outputScale = outputs[0]->GetQuantizationScale();
        int32_t inputZp = inputs[0]->GetQuantizationOffset();
        int32_t outputZp = outputs[0]->GetQuantizationOffset();

        TosaTableAttribute attribute(
            getTosaConstHardSwish8bitTable(inputScale, inputZp, outputScale, outputZp));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else
    {
        throw Exception("ConvertHardSwishToTosaOperator() type currently unimplemented.");
    }

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    std::vector<int32_t> inputShape0;
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    if(inputName.find("input_") != std::string::npos)
    {
        inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,      // name
                                           mainName,       // region name
                                           operators,      // operators
                                           tensors,        // tensors
                                           {inputName},    // inputs
                                           {outputName});  // outputs
}

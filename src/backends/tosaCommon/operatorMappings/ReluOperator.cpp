//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "LeakyReluOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

#include <layers/ActivationLayer.hpp>

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLReluOp
TosaSerializationBasicBlock* ConvertReluToTosaOperator(const Layer* layer,
                                                       const std::vector<const TensorInfo*>& inputs,
                                                       const std::vector<const TensorInfo*>& outputs,
                                                       const ActivationDescriptor* desc)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertReluToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertReluToTosaOperator: 1 output tensor required.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_RELU_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    if(inputName.find("input_") != std::string::npos)
    {
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());
    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    int32_t clamp_min = 0;
    int32_t clamp_max = 0;
    float float_max = 0.0f;
    switch (desc->m_Function)
    {
        case ActivationFunction::ReLu:
        {
            clamp_max = std::numeric_limits<int32_t>::max();
            float_max = std::numeric_limits<float>::max();
            break;
        }
        case ActivationFunction::BoundedReLu:
        {
            clamp_max = static_cast<int32_t>(desc->m_A);
            float_max = desc->m_A;
            break;
        }
        case ActivationFunction::LeakyReLu:
        {
            throw Exception("LeakyRelu TOSA mappings are performed in ConvertLeakyReluToTosaOperator().");
        }
        default:
        {
            throw Exception("Activation function is not supported in ConvertReluToTosaOperator().");
        }
    }

    std::string clampInputNameStr = inputName;
    if (inputDType0 == tosa::DType::DType_INT8 || inputDType0 == tosa::DType::DType_INT16)
    {
        std::string outputNameRescale = std::string("intermediate0_") + GetUniqueTosaMappingID();
        clampInputNameStr = outputNameRescale;

        double scale = inputs[0]->GetQuantizationScale() / outputs[0]->GetQuantizationScale();
        int32_t input_zp = inputs[0]->GetQuantizationOffset();
        int32_t output_zp = outputs[0]->GetQuantizationOffset();

        clamp_min = output_zp;

        if (desc->m_Function == ActivationFunction::BoundedReLu)
        {
            clamp_max = static_cast<int32_t>(std::round(desc->m_A / outputs[0]->GetQuantizationScale())) + output_zp;
        }

        if (inputDType0 == tosa::DType::DType_INT8)
        {
            clamp_min =
                clamp_min < std::numeric_limits<int8_t>::min() ? std::numeric_limits<int8_t>::min() : clamp_min;
            clamp_max =
                clamp_max > std::numeric_limits<int8_t>::max() ? std::numeric_limits<int8_t>::max() : clamp_max;
        }
        else
        {
            clamp_min =
                clamp_min < std::numeric_limits<int16_t>::min() ? std::numeric_limits<int16_t>::min() : clamp_min;
            clamp_max =
                clamp_max > std::numeric_limits<int16_t>::max() ? std::numeric_limits<int16_t>::max() : clamp_max;
        }

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperator(inputName,
                                  outputNameRescale,
                                  scale,
                                  input_zp,
                                  output_zp,
                                  false, //input unsigned
                                  false, //output unsigned
                                  false,
                                  true,
                                  &rescaleOp);
        operators.push_back(rescaleOp);
        tensors.push_back(new TosaSerializationTensor(outputNameRescale,
                                                      inputShape0,
                                                      inputDType0,
                                                      {}));
    }
    
    TosaClampAttribute attribute(clamp_min, clamp_max, 0, float_max);
    auto* clamp_op = new TosaSerializationOperator(Op_CLAMP,
                                                   Attribute_ClampAttribute,
                                                   &attribute,
                                                   {clampInputNameStr},
                                                   {outputName});
    operators.push_back(clamp_op);

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,      // name
                                           mainName,       // region name
                                           operators,      // operators
                                           tensors,        // tensors
                                           {inputName},    // inputs
                                           {outputName});  // outputs
}

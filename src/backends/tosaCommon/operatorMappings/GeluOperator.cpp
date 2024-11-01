//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "GeluOperator.hpp"
#include "TosaTableUtils.hpp"

#include <layers/ActivationLayer.hpp>

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLGeluOp
TosaSerializationBasicBlock* ConvertGeluToTosaOperator(const Layer* layer,
                                                       const std::vector<const TensorInfo*>& inputs,
                                                       const std::vector<const TensorInfo*>& outputs,
                                                       const ActivationDescriptor* desc)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertGeluToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertGeluToTosaOperator: 1 output tensor required.");
    }

    if (desc->m_Function != ActivationFunction::Gelu)
    {
        throw armnn::Exception("ConvertGeluToTosaOperator ActivationDescriptor only supports function Gelu.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_GELU_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    float input_scale = inputs[0]->GetQuantizationScale();
    float output_scale = outputs[0]->GetQuantizationScale();
    int32_t input_zp = inputs[0]->GetQuantizationOffset();
    int32_t output_zp = outputs[0]->GetQuantizationOffset();
    DataType inputDType = inputs[0]->GetDataType();

    bool isInt8 = inputDType == DataType::QAsymmS8 || inputDType == DataType::QSymmS8;
    if (isInt8)
    {
        auto gelu_transform = [](float in) -> float {
            return 0.5f * in * std::erfc(in * static_cast<float>(-0.70710678118654752440));
        };

        TosaTableAttribute attribute(
            getTosaConst8bitTable(input_scale, input_zp, output_scale, output_zp, gelu_transform));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QSymmS16 ||
             inputDType == DataType::Signed32 ||
             inputDType == DataType::Signed64)
    {
        throw Exception("ConvertGeluOperator() only supports int8 quantized types.");
    }
    else
    {
        throw Exception("ConvertGeluOperator() floating point types currently unimplemented.");
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

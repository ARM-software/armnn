//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "TanhOperator.hpp"
#include "TosaTableUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLTanhOp
TosaSerializationBasicBlock* ConvertTanHToTosaOperator(const Layer* layer,
                                                       const std::vector<const TensorInfo*>& inputs,
                                                       const std::vector<const TensorInfo*>& outputs,
                                                       const ActivationDescriptor* desc)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertTanHToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertTanHToTosaOperator: 1 output tensor required.");
    }

    if (desc->m_Function != ActivationFunction::TanH)
    {
        throw armnn::Exception("ConvertTanHToTosaOperator ActivationDescriptor only supports function TanH.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_TANH_block_") + GetUniqueTosaMappingID();
    std::string supportTypes = std::string(" Supported Datatypes: INT8");

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
    std::vector<int32_t> inputShape0;
    if(inputName.find("input_") != std::string::npos)
    {
        inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    DataType inputDType = inputs[0]->GetDataType();

    bool isInt8 = inputDType == DataType::QAsymmS8 || inputDType == DataType::QSymmS8;
    if (isInt8)
    {
        float inputScale = inputs[0]->GetQuantizationScale();
        float outputScale = outputs[0]->GetQuantizationScale();
        int32_t inputZp = inputs[0]->GetQuantizationOffset();
        int32_t outputZp = outputs[0]->GetQuantizationOffset();

        auto tanhFunc = [desc](float x) -> float
        {
            // Need to include 'Alpha upper bound value, m_A' and 'Beta lower bound value, m_B'
            return desc->m_A * (std::tanh(desc->m_B * x));
        };

        TosaTableAttribute attribute(
            getTosaConst8bitTable(inputScale, inputZp, outputScale, outputZp, tanhFunc));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QSymmS16)
    {
        throw Exception("ConvertTanHToTosaOperator(): INT16 is not yet implemented." + supportTypes);
    }
    else if (inputDType == DataType::Float16 ||
             inputDType == DataType::Float32)
    {
        throw Exception("ConvertTanHToTosaOperator(): FLOAT16 or FLOAT32 is not yet implemented." + supportTypes);
    }
    else
    {
        throw Exception("ConvertTanHToTosaOperator(): TOSA Spec doesn't support this datatype." + supportTypes);
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

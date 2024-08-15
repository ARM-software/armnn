//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include "SigmoidOperator.hpp"
#include "TosaTableUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLLogisticOp
TosaSerializationBasicBlock* ConvertSigmoidToTosaOperator(const Layer* layer,
                                                          const std::vector<const TensorInfo*>& inputs,
                                                          const std::vector<const TensorInfo*>& outputs,
                                                          const ActivationDescriptor* desc)
{
    if (inputs.size() != 1)
    {
        throw armnn::Exception("ConvertSigmoidToTosaOperator: 1 input tensors required.");
    }

    if (outputs.size() != 1)
    {
        throw armnn::Exception("ConvertSigmoidToTosaOperator: 1 output tensor required.");
    }

    if (desc->m_Function != ActivationFunction::Sigmoid)
    {
        throw armnn::Exception("ConvertSigmoidToTosaOperator ActivationDescriptor only supports function Sigmoid.");
    }

    std::string inputName  = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_SIGMOID_block_") + GetUniqueTosaMappingID();
    std::string supportedTypes = std::string(" Supported Datatypes: INT8, FLOAT16, FLOAT32");

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

        auto sigmoidFunc = [](float x) -> float
        {
            return 1.0f / (1.0f + std::exp(-x));
        };

        TosaTableAttribute attribute(
            getTosaConst8bitTable(inputScale, inputZp, outputScale, outputZp, sigmoidFunc));
        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QSymmS16)
    {
        throw Exception("ConvertSigmoidToTosaOperator(): INT16 is not implemented." + supportedTypes);
    }
    else if (inputDType == DataType::Float16 ||
             inputDType == DataType::Float32)
    {
        operators.push_back(new TosaSerializationOperator(tosa::Op_SIGMOID,
                                                          Attribute_NONE,
                                                          nullptr,
                                                          {inputName},
                                                          {outputName}));
    }
    else
    {
        throw Exception("ConvertSigmoidToTosaOperator(): TOSA Spec doesn't support this datatype." + supportedTypes);
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

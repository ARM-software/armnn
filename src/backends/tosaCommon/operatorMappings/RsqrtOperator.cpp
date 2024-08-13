//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RsqrtOperator.hpp"
#include "TosaTableUtils.hpp"

TosaSerializationBasicBlock* ConvertRsqrtOperator(const Layer* layer,
                                                  const std::vector<const TensorInfo*>& inputs,
                                                  const std::vector<const TensorInfo*>& outputs,
                                                  const ElementwiseUnaryDescriptor* unaryDescriptor)
{
    if (unaryDescriptor->m_Operation != UnaryOperation::Rsqrt)
    {
        throw armnn::Exception("ConvertRsqrtOperator: Unsupported elementwise unary operation in descriptor.");
    }

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(inputs.size() == 1,
                                        "ConvertRsqrtOperator: Rsqrt must have only one input");

    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE(outputs.size() == 1,
                                        "ConvertRsqrtOperator: Rsqrt must have only one output");


    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_RSQRT_block_") + GetUniqueTosaMappingID();
    std::string supportedTypes = std::string(" Supported Types: FLOAT32, FLOAT16 & INT8.");

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator *> operators;

    DataType inputDType = inputs[0]->GetDataType();

    if (inputDType == DataType::QAsymmS8 || inputDType == DataType::QSymmS8)
    {
        float input_scale = inputs[0]->GetQuantizationScale();
        float output_scale = outputs[0]->GetQuantizationScale();
        int32_t input_zp = inputs[0]->GetQuantizationOffset();
        int32_t output_zp = outputs[0]->GetQuantizationOffset();

        const float output_max = static_cast<float>(127 - output_zp) * output_scale;

        auto rsqrt_func = [&](float x) -> float
        {
            if (x <= 0.0f)
            {
                return output_max;
            }

            return 1.0f / std::sqrt(x);
        };

        TosaTableAttribute attribute(
            getTosaConst8bitTable(input_scale, input_zp, output_scale, output_zp, rsqrt_func));

        operators.push_back(new TosaSerializationOperator(tosa::Op_TABLE,
                                                          Attribute_TableAttribute,
                                                          &attribute,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::Float32 || inputDType == DataType::Float16)
    {
        operators.push_back(new TosaSerializationOperator(tosa::Op_RSQRT,
                                                          Attribute_NONE,
                                                          nullptr,
                                                          {inputName},
                                                          {outputName}));
    }
    else if (inputDType == DataType::QSymmS16)
    {
        throw Exception("ConvertRsqrtOperator(): unsupported datatype INT16 is not implemented yet." + supportedTypes);
    }
    else if (inputDType == DataType::Signed32 || inputDType == DataType::Signed64)
    {
        throw Exception("ConvertRsqrtOperator(): unsupported datatype INT32 or INT64." + supportedTypes);
    }
    else
    {
        throw Exception("ConvertRsqrtOperator(): TOSA specification does not support this datatype." + supportedTypes);
    }

    // Only add input tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if (inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputDType);
        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName,     // name
                                           mainName,      // region name
                                           operators,     // operators
                                           tensors,       // tensors
                                           {inputName},   // inputs
                                           {outputName}); // outputs
}
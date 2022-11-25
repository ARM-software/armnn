//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaRefLayerSupport.hpp"
#include <tosaCommon/TosaMappings.hpp>

#include <armnn/Types.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <tosaCommon/TosaLayerSupportRules.hpp>
#include <LayerSupportCommon.hpp>

#include <vector>
#include <array>
#include <tuple>

namespace armnn
{

static bool RunTosaLayerChecksSingleDataType(TosaSerializationOperator* op,
                                             const std::vector<TosaSerializationTensor*>& inputs,
                                             const std::vector<TosaSerializationTensor*>& outputs,
                                             const std::vector<Attribute>& supportedAttributes,
                                             const std::vector<DType>& supportedTypes,
                                             Optional<string&> reasonIfUnsupported)
{
    bool supported = true;

    std::string opString = TosaOpToString(op->GetOp());

    // Check Attribute from operator (GetAttribute)
    supported &= CheckSupportRule(TosaOperatorAttributeOfAny(op, supportedAttributes), reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opString +
                                              " has an unsupported attribute.").c_str());

    for (auto input : inputs)
    {
        std::string dataTypeCode = std::to_string(input->GetDtype());

        // Check Dtype from tensor (GetDtype)
        supported &= CheckSupportRule(TosaTypeAnyOf(input, supportedTypes),
                                      reasonIfUnsupported,
                                      std::string("TOSA Reference Operator: " + opString + " for input: " +
                                                  input->GetName() + " has an unsupported data type: " +
                                                  dataTypeCode).c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(input),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for input: " +
                                                  input->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    for (auto output : outputs)
    {
        std::string dataTypeCode = std::to_string(output->GetDtype());

        // Check Dtype from tensor (GetDtype)
        supported &= CheckSupportRule(TosaTypeAnyOf(output, supportedTypes),
                                      reasonIfUnsupported,
                                      std::string("TOSA Reference Operator: " + opString + " for output: " +
                                                  output->GetName() + " has an unsupported data type: " +
                                                  dataTypeCode).c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(output),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for output: " +
                                                  output->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    return supported;
}

static bool RunTosaLayerChecksInputOutputDataType(TosaSerializationOperator* op,
                                                  const std::vector<TosaSerializationTensor*>& inputs,
                                                  const std::vector<TosaSerializationTensor*>& outputs,
                                                  const std::vector<Attribute>& supportedAttributes,
                                                  const std::vector<std::tuple<DType,DType>>& supportedMappingTypes,
                                                  Optional<string&> reasonIfUnsupported)
{
    bool supported = true;

    std::string opString = TosaOpToString(op->GetOp());

    // Check Attribute from operator (GetAttribute)
    supported &= CheckSupportRule(TosaOperatorAttributeOfAny(op, supportedAttributes), reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opString +
                                      " has an unsupported attribute.").c_str());

    supported &= CheckSupportRule(TosaAssertSize(inputs, outputs), reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opString +
                                      " must have 1-to-1 mapping of inputs-to-outputs.").c_str());

    for (uint32_t i = 0; i < inputs.size(); i++)
    {
        auto input = inputs[i];
        auto output = outputs[i];
        std::string inputDataTypeCode = std::to_string(input->GetDtype());
        std::string outputDataTypeCode = std::to_string(output->GetDtype());
        std::tuple<DType, DType> mappingType(input->GetDtype(), output->GetDtype());

        // Check Dtype from tensor (GetDtype)
        supported &= CheckSupportRule(TosaContainerContainsTwoTypes(mappingType, supportedMappingTypes),
                                      reasonIfUnsupported,
                                      std::string("TOSA Reference Operator: " + opString + " for input: " +
                                          input->GetName() + " and output: " + output->GetName() +
                                          " has an unsupported input data type: " + inputDataTypeCode +
                                          " to output data type: " + outputDataTypeCode).c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(input),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for input: " +
                                          input->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(output),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for output: " +
                                          output->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    return supported;
}

static bool RunTosaLayerChecksInputWeightsOutputDataType(
        TosaSerializationOperator* op,
        const std::vector<TosaSerializationTensor*>& inputs,
        const std::vector<TosaSerializationTensor*>& outputs,
        const std::vector<Attribute>& supportedAttributes,
        const std::vector<std::tuple<DType, DType, DType>>& supportedMappingTypes,
        Optional<string&> reasonIfUnsupported)
{
    bool supported = true;

    std::string opString = TosaOpToString(op->GetOp());

    // Check Attribute from operator (GetAttribute)
    supported &= CheckSupportRule(TosaOperatorAttributeOfAny(op, supportedAttributes), reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opString +
                                              " has an unsupported attribute.").c_str());

    // Check combination of input, weights and output types.
    // Bias is the same as output type, so it is covered.
    std::tuple<DType, DType, DType> mappingTypes(inputs[0]->GetDtype(), inputs[1]->GetDtype(), outputs[0]->GetDtype());

    // Check Dtype from tensor (GetDtype)
    supported &= CheckSupportRule(TosaContainerContainsThreeTypes(mappingTypes, supportedMappingTypes),
                                  reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opString + " for input 0: " +
                                              inputs[0]->GetName() + ", input 1: " + inputs[1]->GetName() +
                                              " and output: " + outputs[0]->GetName() +
                                              " has an unsupported input data type combination.").c_str());

    for (auto input : inputs)
    {
        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(input),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for input: " +
                                                  input->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    for (auto output : outputs)
    {
        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(output),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opString + " for output: " +
                                                  output->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    return supported;
}



static bool IsTosaLayerSupported(TosaSerializationOperator* op,
                                 const std::vector<TosaSerializationTensor*>& inputs,
                                 const std::vector<TosaSerializationTensor*>& outputs,
                                 Optional<string&> reasonIfUnsupported)
{
    switch(op->GetOp())
    {
        case tosa::Op_ADD:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_NONE };

            // Only Int32, Fp32 and Fp16 are currently supported by the TOSA Reference Model.
            std::vector<DType> supportedTypes =
            {
                DType_INT32,
                DType_FP16,
                DType_FP32
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            return RunTosaLayerChecksSingleDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypes, reasonIfUnsupported);
        }
        case tosa::Op_CONST:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_NONE };

            std::vector<DType> supportedTypes =
            {
                DType_FP16,
                DType_FP32,
                DType_UINT8,
                DType_INT8,
                DType_INT16,
                DType_INT32,
                DType_BOOL
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            return RunTosaLayerChecksSingleDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypes, reasonIfUnsupported);
        }
        case tosa::Op_CONV2D:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_ConvAttribute };

            std::vector<std::tuple<DType, DType, DType>> supportedTypesMapping =
            {
                std::tuple<DType, DType, DType>(DType_FP16, DType_FP16, DType_FP16),
                std::tuple<DType, DType, DType>(DType_FP16, DType_FP16, DType_FP32),
                std::tuple<DType, DType, DType>(DType_FP32, DType_FP32, DType_FP32),
                std::tuple<DType, DType, DType>(DType_INT8, DType_INT8, DType_INT32)
            };

            return RunTosaLayerChecksInputWeightsOutputDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypesMapping, reasonIfUnsupported);
        }
        case tosa::Op_AVG_POOL2D:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_PoolAttribute };

            std::vector<std::tuple<DType, DType>> supportedTypesMapping =
            {
                std::tuple<DType, DType>(DType_FP16, DType_FP16),
                std::tuple<DType, DType>(DType_FP16, DType_FP32),
                std::tuple<DType, DType>(DType_FP32, DType_FP32),
                std::tuple<DType, DType>(DType_INT8, DType_INT32),
                std::tuple<DType, DType>(DType_INT16, DType_INT32)
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            return RunTosaLayerChecksInputOutputDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypesMapping, reasonIfUnsupported);
        }
        case tosa::Op_MAX_POOL2D:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_PoolAttribute };

            std::vector<DType> supportedTypes =
            {
                DType_FP16,
                DType_FP32,
                DType_INT8,
                DType_INT16
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            return RunTosaLayerChecksSingleDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypes, reasonIfUnsupported);
        }
        case tosa::Op_PAD:
        {
            std::vector<Attribute> supportedAttributes = { Attribute_PadAttribute };

            std::vector<DType> supportedTypes =
            {
                DType_FP16,
                DType_FP32,
                DType_INT8,
                DType_INT16,
                DType_INT32,
                DType_BOOL
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            return RunTosaLayerChecksSingleDataType(
                    op, inputs, outputs, supportedAttributes, supportedTypes, reasonIfUnsupported);
        }
        default:
            SetValueChecked(reasonIfUnsupported, "Operation is currently unsupported by the TOSA Reference Backend.");
            return false;
    }
}

bool TosaRefLayerSupport::IsLayerSupported(const LayerType& type,
                                           const std::vector<TensorInfo>& infos,
                                           const BaseDescriptor& descriptor,
                                           const Optional<LstmInputParamsInfo>& lstmParamsInfo,
                                           const Optional<QuantizedLstmInputParamsInfo>& quantizedLstmInputParamsInfo,
                                           Optional<std::string&> reasonIfUnsupported) const
{
    IgnoreUnused(lstmParamsInfo);
    IgnoreUnused(quantizedLstmInputParamsInfo);

    std::vector<const TensorInfo*> inputInfos;
    std::vector<const TensorInfo*> outputInfos;

    switch (type)
    {
        case LayerType::Input:
        case LayerType::Output:
            return true;
        case LayerType::Addition:
            // Setup inputs and outputs
            inputInfos.push_back(&infos[0]);
            inputInfos.push_back(&infos[1]);
            outputInfos.push_back(&infos[2]);
            break;
        case LayerType::Constant:
            outputInfos.push_back(&infos[0]);
            break;
        case LayerType::Convolution2d:
        {
            inputInfos.push_back(&infos[0]); // input
            outputInfos.push_back(&infos[1]); // output
            inputInfos.push_back(&infos[2]); // weights

            auto conv2dDesc = PolymorphicDowncast<const Convolution2dDescriptor*>(&descriptor);
            if(conv2dDesc->m_BiasEnabled)
            {
                inputInfos.push_back(&infos[3]); // bias
            }
            break;
        }
        case LayerType::Pooling2d:
            // Setup inputs and outputs
            inputInfos.push_back(&infos[0]);
            outputInfos.push_back(&infos[1]);
            break;
        default:
            break;
    }

    auto mappings = GetTosaMapping(nullptr, type, inputInfos, outputInfos, descriptor);
    if (mappings->GetName() == "")
    {
        // There currently isn't a TOSA mapping for this layer, as the default was returned.
        return false;
    }

    // Loop through block and get each tensor and operator
    for (long unsigned int i = 0; i < mappings->GetOperators().size(); ++i)
    {
        // While looping over operators check for op_UNKNOWN which is unsupported
        if (mappings->GetOperators()[i]->GetOp() == tosa::Op_UNKNOWN) { return false; }

        // Loop over operators and get GetInput/OutputTensorNames, loop over resulting names and
        // use GetTensorByName to pass pointers to tensors on to the IsTosaLayerSupported()
        std::vector<TosaSerializationTensor*> inputTensorsVect;
        for (const auto& name : mappings->GetOperators()[i]->GetInputTensorNames())
        {
            inputTensorsVect.push_back(mappings->GetTensorByName(name));
        }

        std::vector<TosaSerializationTensor*> outputTensorsVect;
        for (const auto& name : mappings->GetOperators()[i]->GetOutputTensorNames())
        {
            outputTensorsVect.push_back(mappings->GetTensorByName(name));
        }

        if (!IsTosaLayerSupported(mappings->GetOperators()[i],
                                  inputTensorsVect,
                                  outputTensorsVect,
                                  reasonIfUnsupported))
        {
            return false;
        }
    }
    return true;
}

} // namespace armnn

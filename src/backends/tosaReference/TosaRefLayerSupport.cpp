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

namespace armnn
{

static bool RunTosaLayerChecks(TosaSerializationOperator* op,
                               const std::vector<TosaSerializationTensor*>& inputs,
                               const std::vector<TosaSerializationTensor*>& outputs,
                               const std::vector<Attribute>& supportedAttributes,
                               const std::vector<DType>& supportedTypes,
                               Optional<string&> reasonIfUnsupported)
{
    bool supported = true;

    std::string opCode = std::to_string(op->GetOp());

    // Check Attribute from operator (GetAttribute)
    supported &= CheckSupportRule(TosaOperatorAttributeOfAny(op, supportedAttributes), reasonIfUnsupported,
                                  std::string("TOSA Reference Operator: " + opCode +
                                              " has an unsupported attribute.").c_str());

    for (auto input : inputs)
    {
        std::string dataTypeCode = std::to_string(input->GetDtype());

        // Check Dtype from tensor (GetDtype)
        supported &= CheckSupportRule(TosaTypeAnyOf(input, supportedTypes),
                                      reasonIfUnsupported,
                                      std::string("TOSA Reference Operator: " + opCode + " for input: " +
                                                  input->GetName() + " has an unsupported data type: " +
                                                  dataTypeCode).c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(input),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opCode + " for input: " +
                                                  input->GetName() + " exceeds MaxNumOfTensorDimensions.").c_str());
    }

    for (auto output : outputs)
    {
        std::string dataTypeCode = std::to_string(output->GetDtype());

        // Check Dtype from tensor (GetDtype)
        supported &= CheckSupportRule(TosaTypeAnyOf(output, supportedTypes),
                                      reasonIfUnsupported,
                                      std::string("TOSA Reference Operator: " + opCode + " for output: " +
                                                  output->GetName() + " has an unsupported data type: " +
                                                  dataTypeCode).c_str());

        // Check Shape from tensor (GetShape)
        supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(output),
                                      reasonIfUnsupported,
                                      std::string("Tosa Reference Operator: " + opCode + " for output: " +
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
            bool supported = true;

            std::vector<Attribute> supportedAttributes =
            {
                Attribute_NONE
            };

            // Only Int32, Fp32 and Fp16 are currently supported by the TOSA Reference Model.
            std::vector<DType> supportedTypes =
            {
                DType_INT32,
                DType_FP16,
                DType_FP32
            };

            // Check the attribute, data types and bounds for inputs and outputs.
            supported = RunTosaLayerChecks(op,
                                           inputs,
                                           outputs,
                                           supportedAttributes,
                                           supportedTypes,
                                           reasonIfUnsupported);

            return supported;
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
        case LayerType::Addition:
            // Setup inputs and outputs
            inputInfos.push_back(&infos[0]);
            inputInfos.push_back(&infos[1]);
            outputInfos.push_back(&infos[2]);
            break;
        case LayerType::Input:
        case LayerType::Output:
            return true;
        default:
            break;
    }

    auto mappings = GetTosaMapping(type, inputInfos, outputInfos, descriptor, false);
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

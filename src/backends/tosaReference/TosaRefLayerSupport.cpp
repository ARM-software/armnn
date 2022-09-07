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

            std::array<Attribute, 1> supportedAttributes =
                    {
                            Attribute_NONE
                    };

            // Check Attribute from operator (GetAttribute)
            supported &= CheckSupportRule(TosaOperatorAttributeOfAny(op, supportedAttributes), reasonIfUnsupported,
                std::string("TOSA Reference addition: operator has an unsupported attribute.").c_str());

            std::array<DType, 8> supportedTypes =
                    {
                            DType_BOOL,
                            DType_UINT8,
                            DType_INT4,
                            DType_INT8,
                            DType_INT16,
                            DType_INT32,
                            DType_FLOAT,
                            DType_UINT16
                    };

            for (auto tensor : inputs)
            {
                // Check Dtype from tensor (GetDtype)
                supported &= CheckSupportRule(TosaTypeAnyOf(tensor, supportedTypes),
                    reasonIfUnsupported,
                    std::string("TOSA Reference addition: " + tensor->GetName() +
                    " is not a supported type.").c_str());

                // Check Shape from tensor (GetShape)
                supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(tensor),
                    reasonIfUnsupported,
                    std::string("Tosa Reference addition: " + tensor->GetName() + " Shape.Size()"
                    " outside bounds of between Zero and MaxNumOfTensorDimensions.").c_str());
            }

            // Check Dtype from tensor (GetDtype)
            supported &= CheckSupportRule(TosaTypeAnyOf(outputs[0], supportedTypes),
                reasonIfUnsupported,
                std::string("TOSA Reference addition: " + outputs[0]->GetName() +
                " is not a supported type.").c_str());

            // Check Shape from tensor (GetShape)
            supported &= CheckSupportRule(TosaTensorNumDimensionsWithinBounds(outputs[0]),
                reasonIfUnsupported,
                std::string("Tosa Reference addition: " + outputs[0]->GetName() + " Shape.Size()"
                " outside bounds of between Zero and MaxNumOfTensorDimensions.").c_str());

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

    // Setup Inputs
    const auto input0      = infos[0];
    const TensorInfo* ptr0 = &input0;
    const auto input1      = infos[1];
    const TensorInfo* ptr1 = &input1;
    std::vector<const TensorInfo*> inputInfos = {ptr0, ptr1};

    // Setup Outputs
    const auto output      = infos[2];
    const TensorInfo* ptr2 = &output;
    std::vector<const TensorInfo*> outputInfos = {ptr2};

    auto mappings = GetTosaMapping(type, inputInfos, outputInfos, descriptor);

    // Loop through block and get each tensor and operator
    for (long unsigned int i = 0; i < mappings->GetOperators().size(); ++i)
    {
        // While looping over operators check for op_UNKNOWN which is unsupported
        if (mappings->GetOperators()[i]->GetOp() == tosa::Op_UNKNOWN) { return false;}

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

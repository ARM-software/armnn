//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadOperator.hpp"
#include <armnnUtils/QuantizeHelper.hpp>

TosaSerializationBasicBlock* ConvertPadToTosaOperator(const Layer* layer,
                                                      const std::vector<const TensorInfo*>& inputs,
                                                      const std::vector<const TensorInfo*>& outputs,
                                                      const PadDescriptor* padDescriptor)
{
    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_PAD_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<int32_t> padding;
    padding.reserve(padDescriptor->m_PadList.size());
    for (size_t it = 0; it < padDescriptor->m_PadList.size(); ++it) {
        padding.push_back(static_cast<int32_t>(padDescriptor->m_PadList[it].first));
        padding.push_back(static_cast<int32_t>(padDescriptor->m_PadList[it].second));
    }

    auto intPadValue = armnnUtils::SelectiveQuantize<int32_t>(padDescriptor->m_PadValue,
                                                              inputs[0]->GetQuantizationScale(),
                                                              inputs[0]->GetQuantizationOffset());
    TosaPadAttribute padAttribute(padding, intPadValue ,padDescriptor->m_PadValue);

    auto* op = new TosaSerializationOperator(Op_PAD,
                                             Attribute_PadAttribute,
                                             &padAttribute,
                                             {inputName},
                                             {outputName});

    std::vector<TosaSerializationTensor*> tensors;

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(inputName, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {op}, // operators
                                           tensors, // tensors
                                           {inputName}, // inputs
                                           {outputName}); // outputs
}
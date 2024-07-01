//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryOperator.hpp"

TosaSerializationBasicBlock* ConvertElementwiseUnaryOperator(const Layer* layer,
                                                             const std::vector<const TensorInfo*>& inputs,
                                                             const std::vector<const TensorInfo*>& outputs,
                                                             const ElementwiseUnaryDescriptor* unaryDescriptor)
{
    std::string input0Name = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_ELEMENTWISEUNARY_block_") + GetUniqueTosaMappingID();


    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        input0Name = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    TosaSerializationOperator* op = nullptr;
    switch(unaryDescriptor->m_Operation)
    {
        case UnaryOperation::Rsqrt:
        {
            op = new TosaSerializationOperator(tosa::Op_RSQRT,
                                               Attribute_NONE,
                                               nullptr,
                                               {input0Name},
                                               {outputName});
            blockName = std::string("Op_RSQRT_block_") + GetUniqueTosaMappingID();
            break;
        }
        case UnaryOperation::Exp:
            throw armnn::Exception(
                "ConvertElementwiseUnaryToTosaOperator: Unexpected operation Exp. Use ConvertExpOperator() instead.");
        default:
            throw armnn::Exception("ConvertElementwiseUnaryToTosaOperator: Unsupported layer type.");
    }

    std::vector<TosaSerializationTensor*> tensors;
    // Only add input tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(input0Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {op}, // operators
                                           tensors, // tensors
                                           {input0Name}, // inputs
                                           {outputName}); // outputs
}
//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryOperator.hpp"

TosaSerializationBasicBlock* ConvertElementwiseBinaryToTosaOperator(const Layer* layer,
                                                                    const LayerType type,
                                                                    const std::vector<const TensorInfo*>& inputs,
                                                                    const std::vector<const TensorInfo*>& outputs)
{
    std::string input0Name = std::string("input0_");
    std::string input1Name = std::string("input1_");
    std::string outputName = std::string("output0_");
    std::string blockName;

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensor names.
        Layer& connectedLayer0 = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        input0Name = GenerateUniqueName(connectedLayer0, 0);

        Layer& connectedLayer1 = layer->GetInputSlot(1).GetConnectedOutputSlot()->GetOwningLayer();
        input1Name = GenerateUniqueName(connectedLayer1, 1);

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer, 0);
    }

    TosaSerializationOperator* op = nullptr;
    switch(type)
    {
        case LayerType::Addition:
        {
            op = new TosaSerializationOperator(Op_ADD,
                                               Attribute_NONE,
                                               nullptr,
                                               {input0Name, input1Name},
                                               {outputName});
            blockName = std::string("Op_ADD_block_") + GetUniqueTosaMappingID();
            break;
        }
        case LayerType::Multiplication:
        {
            int32_t shift = 0;
            TosaMulAttribute mulAttribute(shift);
            op = new TosaSerializationOperator(Op_MUL,
                                               Attribute_MulAttribute,
                                               &mulAttribute,
                                               {input0Name, input1Name},
                                               {outputName});
            blockName = std::string("Op_MUL_block_") + GetUniqueTosaMappingID();
            break;
        }
        case LayerType::Subtraction:
        {
            op = new TosaSerializationOperator(Op_SUB,
                                               Attribute_NONE,
                                               nullptr,
                                               {input0Name, input1Name},
                                               {outputName});
            blockName = std::string("Op_SUB_block_") + GetUniqueTosaMappingID();
            break;
        }
        default:
            throw armnn::Exception("ConvertElementwiseBinaryToTosaOperator: Unsupported layer type.");
    }
    ARMNN_ASSERT(op != nullptr);

    std::vector<TosaSerializationTensor*> tensors;
    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(input0Name.find("input0_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }
    if(input1Name.find("input1_") != std::string::npos)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to Tosa mappings
    return new TosaSerializationBasicBlock(blockName, // name
                                           {op}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}


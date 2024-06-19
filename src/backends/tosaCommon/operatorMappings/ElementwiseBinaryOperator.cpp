//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

TosaSerializationOperator* AddRescaleOp(const string &inputName,
                                        const string &outputName,
                                        std::vector<TosaSerializationTensor *> &tensors,
                                        const std::vector<const TensorInfo *> &inputs,
                                        const std::vector<const TensorInfo *> &outputs)
{
        double scale_alpha = inputs[1]->GetQuantizationScale() / outputs[0]->GetQuantizationScale();
        int32_t input_zp   = inputs[1]->GetQuantizationOffset();
        int32_t output_zp  = outputs[0]->GetQuantizationOffset();

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperator(inputName,
                                  outputName,
                                  scale_alpha,
                                  input_zp,
                                  output_zp,
                                  true,
                                  true,
                                  &rescaleOp);

        std::vector<int32_t> shape = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.push_back(new TosaSerializationTensor(outputName,
                                                      shape,
                                                      DType_INT32, {}));
        return rescaleOp;
}

TosaSerializationBasicBlock* ConvertElementwiseBinaryToTosaOperator(const Layer* layer,
                                                                    const LayerType type,
                                                                    const std::vector<const TensorInfo*>& inputs,
                                                                    const std::vector<const TensorInfo*>& outputs,
                                                                    const ElementwiseBinaryDescriptor* descriptor)
{
    std::string input0Name = std::string("input_0");
    std::string input1Name = std::string("input_1");
    std::string outputName = std::string("output0_");
    std::string input0ElemenwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string input1ElemenwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string blockName;

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        input0Name = GenerateUniqueInputName(layer->GetInputSlot(0));
        input1Name = GenerateUniqueInputName(layer->GetInputSlot(1));
        outputName = GenerateUniqueOutputName(*layer);
    }

    TosaSerializationOperator* op = nullptr;

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;
    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());
    bool isInputInt8 = (inputDType0 == DType_INT8);

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if(input0Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        tensors.push_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }
    if(input1Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.push_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());

    // Assign an output name and add to tensors based on the input type
    // An int8 input for all ops will require the output to be rescaled from int32 to int8
    std::string outputElemenwiseBinaryName;
    if (isInputInt8)
    {
        outputElemenwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
        tensors.push_back(new TosaSerializationTensor(outputElemenwiseBinaryName, outputShape0, DType_INT32, {}));
    }
    else
    {
        tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));
    }

    std::string& elementwiseInput0Str = isInputInt8 ? input0ElemenwiseBinaryName : input0Name;
    std::string& elementwiseInput1Str = isInputInt8 ? input1ElemenwiseBinaryName : input1Name;
    std::string& elementwiseOutputStr = isInputInt8 ? outputElemenwiseBinaryName : outputName;
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
        case LayerType::ElementwiseBinary:
        {
            switch (descriptor->m_Operation)
            {
                case armnn::BinaryOperation::Add:
                {
                    // Add supports DType_INT32 input only, so a rescale is required when input is DType_INT8
                    if (inputDType0 == DType_INT8)
                    {
                        operators.push_back(
                                AddRescaleOp(input0Name, input0ElemenwiseBinaryName, tensors, inputs, outputs));

                        operators.push_back(
                                AddRescaleOp(input1Name, input1ElemenwiseBinaryName, tensors, inputs, outputs));
                    }
                    op = new TosaSerializationOperator(Op_ADD,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_ADD_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case armnn::BinaryOperation::Maximum:
                {
                    // Add supports DType_INT32 input only, so a rescale is required when input is DType_INT8
                    if (inputDType0 == DType_INT8)
                    {
                        operators.push_back(
                                AddRescaleOp(input0Name, input0ElemenwiseBinaryName, tensors, inputs, outputs));

                        operators.push_back(
                                AddRescaleOp(input1Name, input1ElemenwiseBinaryName, tensors, inputs, outputs));
                    }
                    op = new TosaSerializationOperator(Op_MAXIMUM,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_MAXIMUM_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case armnn::BinaryOperation::Mul:
                {
                    int8_t shift = 0;
                    TosaMulAttribute mulAttribute(shift);

                    // Mul supports input DType_INT8 so will not require a rescale before the op.
                    // i.e "input0Name" is used for the input and not intermediate "elementwiseInput0Str"
                    op = new TosaSerializationOperator(Op_MUL,
                                                       Attribute_MulAttribute,
                                                       &mulAttribute,
                                                       {input0Name, input1Name},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_MUL_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case armnn::BinaryOperation::Sub:
                {
                    // Sub supports DType_INT32 input only, so a rescale is required when input is DType_INT8
                    if (inputDType0 == DType_INT8)
                    {
                        operators.push_back(
                                AddRescaleOp(input0Name, input0ElemenwiseBinaryName, tensors, inputs, outputs));

                        operators.push_back(
                                AddRescaleOp(input1Name, input1ElemenwiseBinaryName, tensors, inputs, outputs));
                    }

                    op = new TosaSerializationOperator(Op_SUB,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_SUB_block_") + GetUniqueTosaMappingID();
                    break;
                }
                default:
                    throw armnn::Exception("ConvertElementwiseBinaryToTosaOperator: Unsupported layer type.");
            }
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

    operators.push_back(op);

    // All operators require a rescale of the output from DType_INT32 to DType_INT8 when the input is DType_INT8
    if (inputDType0 == DType_INT8)
    {
        operators.push_back(
                AddRescaleOp(outputElemenwiseBinaryName, outputName, tensors, inputs, outputs));
    }

    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {operators}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}


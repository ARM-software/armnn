//
// Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

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

    // Add supports DType_INT32 input only, so a rescale is required when input is DType_INT8
    // MUL op is the only exception which has TOSA int8 support
    bool isMulDesc = descriptor ? descriptor->m_Operation == BinaryOperation::Mul : false;
    bool isMulOp = (type == LayerType::Multiplication) || isMulDesc ? true : false;
    if (isInputInt8 && !isMulOp)
    {
        TosaSerializationOperator* rescaleOp0 = nullptr;
        CreateRescaleTosaOperator(input0Name,
                                  input0ElemenwiseBinaryName,
                                  inputs[0]->GetQuantizationScale() / outputs[0]->GetQuantizationScale(),
                                  inputs[0]->GetQuantizationOffset(),
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp0);
        tensors.push_back(new TosaSerializationTensor(input0ElemenwiseBinaryName,
                                                      GetTosaTensorShape(inputs[0]->GetShape()),
                                                      DType_INT32,
                                                      {}));
        operators.push_back(rescaleOp0);

        TosaSerializationOperator* rescaleOp1 = nullptr;
        CreateRescaleTosaOperator(input1Name,
                                  input1ElemenwiseBinaryName,
                                  inputs[1]->GetQuantizationScale() / outputs[0]->GetQuantizationScale(),
                                  inputs[1]->GetQuantizationOffset(),
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp1);
        tensors.push_back(new TosaSerializationTensor(input1ElemenwiseBinaryName,
                                                      GetTosaTensorShape(inputs[1]->GetShape()),
                                                      DType_INT32,
                                                      {}));
        operators.push_back(rescaleOp1);
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
                case BinaryOperation::Add:
                {
                    op = new TosaSerializationOperator(Op_ADD,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_ADD_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case BinaryOperation::Maximum:
                {
                    op = new TosaSerializationOperator(Op_MAXIMUM,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_MAXIMUM_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case BinaryOperation::Mul:
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
                case BinaryOperation::Sub:
                {
                    op = new TosaSerializationOperator(Op_SUB,
                                                       Attribute_NONE,
                                                       nullptr,
                                                       {elementwiseInput0Str, elementwiseInput1Str},
                                                       {elementwiseOutputStr});
                    blockName = std::string("Op_SUB_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case BinaryOperation::SqDiff:
                {
                    throw Exception("TOSA mappings of Squared Difference operator "
                                    "implemented under ConvertSquaredDifferenceToTosaOperator().");
                }
                default:
                    throw Exception("ConvertElementwiseBinaryToTosaOperator: Unsupported layer type.");
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
            throw Exception("ConvertElementwiseBinaryToTosaOperator: Unsupported layer type.");
    }

    operators.push_back(op);

    // All ElementwiseBinary operators require a rescale of output
    // from DType_INT32 to DType_INT8 when the input is DType_INT8
    if (inputDType0 == DType_INT8)
    {
        // double output_rescale_scale = in_lhs_scale * in_rhs_scale / output_scale;
        float input0QScale = inputs[0]->IsQuantized()?inputs[0]->GetQuantizationScale():1.0f;
        float input1QScale = inputs[1]->IsQuantized()?inputs[1]->GetQuantizationScale():1.0f;
        float outputQScale = outputs[0]->IsQuantized()?outputs[0]->GetQuantizationScale():1.0f;
        double combinedQScale = input0QScale * input1QScale / outputQScale;

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperator(outputElemenwiseBinaryName,
                                  outputName,
                                  combinedQScale,
                                  0,
                                  outputs[0]->GetQuantizationOffset(),
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp);
        tensors.push_back(new TosaSerializationTensor(outputName,
                                                      GetTosaTensorShape(outputs[0]->GetShape()),
                                                      DType_INT8,
                                                      {}));
        operators.push_back(rescaleOp);
    }

    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {operators}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}

TosaSerializationBasicBlock* ConvertSquaredDifferenceToTosaOperator(const Layer* layer,
                                                                    const LayerType,
                                                                    const std::vector<const TensorInfo*>& inputs,
                                                                    const std::vector<const TensorInfo*>& outputs,
                                                                    const ElementwiseBinaryDescriptor* descriptor)
{
    if (descriptor->m_Operation != BinaryOperation::SqDiff)
    {
        throw Exception("ElementwiseBinaryDescriptor operation must be SqDiff"
                        "in ConvertSquaredDifferenceToTosaOperator().");
    }

    std::string input0Name = std::string("input_0");
    std::string input1Name = std::string("input_1");
    std::string outputName = std::string("output0_");
    std::string interElemenwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string blockName = std::string("Op_SQDIFF_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if (layer != nullptr)
    {
        if (layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer().GetType() == LayerType::Reshape ||
            layer->GetInputSlot(1).GetConnectedOutputSlot()->GetOwningLayer().GetType() == LayerType::Reshape)
        {
            interElemenwiseBinaryName = std::string("intermediate1_") + GetUniqueTosaMappingID();
        }

        input0Name = GenerateUniqueInputName(layer->GetInputSlot(0));
        input1Name = GenerateUniqueInputName(layer->GetInputSlot(1));
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors {};
    std::vector<TosaSerializationOperator*> operators {};
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

    if (inputDType0 == DType_FP32 ||
        inputDType0 == DType_FP16 ||
        inputDType0 == DType_INT32)
    {
        operators.push_back(new TosaSerializationOperator(
            Op_SUB,
            Attribute_NONE,
            nullptr,
            {input0Name, input1Name},
            {interElemenwiseBinaryName}));
        tensors.push_back(new TosaSerializationTensor(interElemenwiseBinaryName,
                                                      outputShape0,
                                                      outputDType0,
                                                      {}));

        int8_t shift = 0;
        TosaMulAttribute mulAttribute(shift);

        operators.push_back(new TosaSerializationOperator(
            Op_MUL,
            Attribute_MulAttribute,
            &mulAttribute,
            {interElemenwiseBinaryName, interElemenwiseBinaryName},
            {outputName}));
    }
    else if (isInputInt8)
    {
        std::string rescale0Output0Name = std::string("intermediate0_") + GetUniqueTosaMappingID();
        std::string rescale0Output1Name = std::string("intermediate1_") + GetUniqueTosaMappingID();
        std::string rescale1Output0Name = std::string("intermediate2_") + GetUniqueTosaMappingID();
        std::string rescale1Output1Name = std::string("intermediate3_") + GetUniqueTosaMappingID();
        std::string mulOutputName = std::string("intermediate4_") + GetUniqueTosaMappingID();
        interElemenwiseBinaryName = std::string("intermediate5_") + GetUniqueTosaMappingID();

        // We need to make sure the inputs are rescaled correctly
        // Following the behaviour defined here lite/kernels/squared_difference.cc
        double in_x_scale = inputs[0]->GetQuantizationScale();
        double in_y_scale = inputs[1]->GetQuantizationScale();
        double result_scale = outputs[0]->GetQuantizationScale();
        double twice_max_input_scale = 2.0 * std::max(in_x_scale, in_y_scale);
        const int32_t LEFT_SHIFT = 7;
        double x_rescale_scale = in_x_scale / twice_max_input_scale;
        double y_rescale_scale = in_y_scale / twice_max_input_scale;
        double output_rescale_scale =
            (twice_max_input_scale * twice_max_input_scale) /
            ((static_cast<double>(1 << LEFT_SHIFT * 2)) * result_scale);

        TosaSerializationOperator* xShiftOp = nullptr;
        CreateRescaleTosaOperator(input0Name,
                                  rescale0Output0Name,
                                  (1 << LEFT_SHIFT),
                                  inputs[0]->GetQuantizationOffset(),
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &xShiftOp);
        operators.push_back(xShiftOp);
        tensors.push_back(new TosaSerializationTensor(rescale0Output0Name,
                                                      GetTosaTensorShape(inputs[0]->GetShape()),
                                                      DType_INT32,
                                                      {}));

        TosaSerializationOperator* yShiftOp = nullptr;
        CreateRescaleTosaOperator(input1Name,
                                  rescale0Output1Name,
                                  (1 << LEFT_SHIFT),
                                  inputs[1]->GetQuantizationOffset(),
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &yShiftOp);
        operators.push_back(yShiftOp);
        tensors.push_back(new TosaSerializationTensor(rescale0Output1Name,
                                                      GetTosaTensorShape(inputs[1]->GetShape()),
                                                      DType_INT32,
                                                      {}));

        TosaSerializationOperator* xScaledOp = nullptr;
        CreateRescaleTosaOperator(rescale0Output0Name,
                                  rescale1Output0Name, //change
                                  x_rescale_scale,
                                  0,
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &xScaledOp);
        operators.push_back(xScaledOp);
        tensors.push_back(new TosaSerializationTensor(rescale1Output0Name,
                                                      GetTosaTensorShape(inputs[0]->GetShape()),
                                                      DType_INT32,
                                                      {}));

        TosaSerializationOperator* yScaledOp = nullptr;
        CreateRescaleTosaOperator(rescale0Output1Name,
                                  rescale1Output1Name, //change
                                  y_rescale_scale,
                                  0,
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &yScaledOp);
        operators.push_back(yScaledOp);
        tensors.push_back(new TosaSerializationTensor(rescale1Output1Name,
                                                      GetTosaTensorShape(inputs[1]->GetShape()),
                                                      DType_INT32,
                                                      {}));



        operators.push_back(new TosaSerializationOperator(
            Op_SUB,
            Attribute_NONE,
            nullptr,
            {rescale1Output0Name, rescale1Output1Name},
            {interElemenwiseBinaryName}));
        tensors.push_back(new TosaSerializationTensor(interElemenwiseBinaryName,
                                                      GetTosaTensorShape(outputs[0]->GetShape()),
                                                      DType_INT32,
                                                      {}));
        int8_t shift = 0;
        TosaMulAttribute mulAttribute(shift);

        operators.push_back(new TosaSerializationOperator(
            Op_MUL,
            Attribute_MulAttribute,
            &mulAttribute,
            {interElemenwiseBinaryName, interElemenwiseBinaryName},
            {mulOutputName}));
        tensors.push_back(new TosaSerializationTensor(mulOutputName,
                                                      GetTosaTensorShape(outputs[0]->GetShape()),
                                                      DType_INT32,
                                                      {}));


        TosaSerializationOperator* rescaleOutputOp = nullptr;
        CreateRescaleTosaOperator(mulOutputName,
                                  outputName,
                                  output_rescale_scale,
                                  0,
                                  outputs[0]->GetQuantizationOffset(),
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOutputOp);
        operators.push_back(rescaleOutputOp);
    }
    else
    {
        throw Exception("TOSA spec only supports INT8, INT32, FP16 and FP32 datatypes for SqDiff.");
    }

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {operators}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}


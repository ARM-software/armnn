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
    auto input0Name = std::string("input_0");
    auto input1Name = std::string("input_1");
    auto outputName = std::string("output0_");
    std::string input0ElementwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string input1ElementwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
    std::string input2ElementwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
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
        tensors.emplace_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }
    if(input1Name.find("input_") != std::string::npos && input0Name != input1Name)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.emplace_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {}));
    }

    // Assign an output name and add to tensors based on the input type
    // An int8 input for all ops will require the output to be rescaled from int32 to int8
    std::string outputElemenwiseBinaryName;
    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    if (isInputInt8)
    {
        outputElemenwiseBinaryName = std::string("intermediate0_") + GetUniqueTosaMappingID();
        tensors.emplace_back(new TosaSerializationTensor(outputElemenwiseBinaryName, outputShape0, DType_INT32, {}));
    }
    else
    {
        tensors.emplace_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));
    }

    float input0Scale = 0;
    float input1Scale = 0;
    float outputScale = 0;

    if (isInputInt8)
    {
        input0Scale = inputs[0]->GetQuantizationScale();
        input1Scale = inputs[1]->GetQuantizationScale();
        outputScale = outputs[0]->GetQuantizationScale();

        CalculateRescaleScales(input0Scale, input1Scale, outputScale, descriptor->m_Operation);

        TosaSerializationOperator* rescaleOp0 = nullptr;
        CreateRescaleTosaOperator(input0Name, input0ElementwiseBinaryName,
                                  input0Scale,
                                  inputs[0]->GetQuantizationOffset(),
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp0);

        tensors.emplace_back(new TosaSerializationTensor(input0ElementwiseBinaryName,
                                                         GetTosaTensorShape(inputs[0]->GetShape()),
                                                         DType_INT32,
                                                         {}));
        operators.emplace_back(rescaleOp0);

        TosaSerializationOperator* rescaleOp1 = nullptr;

        bool isSub = type == LayerType::Subtraction || (descriptor && descriptor->m_Operation == BinaryOperation::Sub);
        if(isSub)
        {
            // Correct rescale values comes from model converter values which matches TFLite reference outputs.
            auto maxScale = 2.0 * std::max(inputs[0]->GetQuantizationScale(), inputs[1]->GetQuantizationScale());
            auto rescaleScale = static_cast<float>((inputs[0]->GetQuantizationScale() / maxScale) * (1 << 21));
            CreateRescaleTosaOperator(input1Name,
                                      input1ElementwiseBinaryName,
                                      rescaleScale,
                                      inputs[1]->GetQuantizationOffset(),
                                      0,
                                      false,
                                      false,
                                      true,
                                      true,
                                      &rescaleOp1);
            operators.emplace_back(rescaleOp1);
            tensors.emplace_back(new TosaSerializationTensor(input1ElementwiseBinaryName,
                                                             GetTosaTensorShape(inputs[1]->GetShape()),
                                                             DType_INT32,
                                                             {}));

            TosaSerializationOperator* rescaleOp2 = nullptr;
            CreateRescaleTosaOperator(input1ElementwiseBinaryName,
                                      input2ElementwiseBinaryName,
                                      input1Scale,
                                      0,
                                      0,
                                      false,
                                      false,
                                      true,
                                      true,
                                      &rescaleOp2);
            operators.emplace_back(rescaleOp2);
            tensors.emplace_back(new TosaSerializationTensor(input2ElementwiseBinaryName,
                                                             GetTosaTensorShape(inputs[1]->GetShape()),
                                                             DType_INT32,
                                                             {}));
        }
        else
        {
            CreateRescaleTosaOperator(input1Name,
                                      input1ElementwiseBinaryName,
                                      input1Scale,
                                      inputs[1]->GetQuantizationOffset(),
                                      0,
                                      false,
                                      false,
                                      true,
                                      true,
                                      &rescaleOp1);
            operators.emplace_back(rescaleOp1);
            tensors.emplace_back(new TosaSerializationTensor(input1ElementwiseBinaryName,
                                                             GetTosaTensorShape(inputs[1]->GetShape()),
                                                             DType_INT32,
                                                             {}));
        }
    }

    std::string const& elementwiseInput0Str = isInputInt8 ? input0ElementwiseBinaryName : input0Name;
    std::string elementwiseInput1Str = isInputInt8 ? input1ElementwiseBinaryName : input1Name;
    std::string const& elementwiseOutputStr = isInputInt8 ? outputElemenwiseBinaryName : outputName;

    switch(type)
    {
        case LayerType::ElementwiseBinary:
        {
            switch (descriptor->m_Operation)
            {
                case BinaryOperation::Add:
                {
                    ConvertAddToTosaOperator({elementwiseInput0Str, elementwiseInput1Str},
                                             {elementwiseOutputStr},
                                             operators);
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
                    ConvertMulToTosaOperator({elementwiseInput0Str, elementwiseInput1Str},
                                             {elementwiseOutputStr},
                                             operators);
                    blockName = std::string("Op_MUL_block_") + GetUniqueTosaMappingID();
                    break;
                }
                case BinaryOperation::Sub:
                {
                    if (isInputInt8)
                    {
                        elementwiseInput1Str = input2ElementwiseBinaryName;
                    }

                    ConvertSubToTosaOperator({elementwiseInput0Str, elementwiseInput1Str},
                                             {elementwiseOutputStr},
                                             operators);
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
        case LayerType::Addition:
        {
            ConvertAddToTosaOperator({input0Name, input1Name},
                                     {outputName},
                                     operators);
            blockName = std::string("Op_ADD_block_") + GetUniqueTosaMappingID();
            break;
        }
        case LayerType::Multiplication:
        {
            ConvertMulToTosaOperator({input0Name, input1Name},
                                     {outputName},
                                     operators);
            blockName = std::string("Op_MUL_block_") + GetUniqueTosaMappingID();
            break;
        }
        case LayerType::Subtraction:
        {
            ConvertSubToTosaOperator({input0Name, input1Name},
                                     {outputName},
                                     operators);
            blockName = std::string("Op_SUB_block_") + GetUniqueTosaMappingID();
            break;
        }
        default:
            throw Exception("ConvertElementwiseBinaryToTosaOperator: Unsupported layer type.");
    }

    if(op != nullptr)
    {
        operators.emplace_back(op);
    }

    // All ElementwiseBinary operators require a rescale of output
    // from DType_INT32 to DType_INT8 when the input is DType_INT8
    if (inputDType0 == DType_INT8)
    {
        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperator(outputElemenwiseBinaryName,
                                  outputName,
                                  outputScale,
                                  0,
                                  outputs[0]->GetQuantizationOffset(),
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp);
        tensors.emplace_back(new TosaSerializationTensor(outputName,
                                                         GetTosaTensorShape(outputs[0]->GetShape()),
                                                         DType_INT8,
                                                         {}));
        operators.emplace_back(rescaleOp);
    }

    if(input0Name == input1Name)
    {
        return new TosaSerializationBasicBlock(blockName,     // name
                                               mainName,      // region name
                                               {operators},   // operators
                                               tensors,       // tensors
                                               {input0Name},  // inputs
                                               {outputName}); // outputs
    }

    return new TosaSerializationBasicBlock(blockName,                // name
                                           mainName,                 // region name
                                           {operators},              // operators
                                           tensors,                  // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName});            // outputs
}

void ConvertAddToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators)
{
    operators.emplace_back(new TosaSerializationOperator(Op_ADD,
                                                         Attribute_NONE,
                                                         nullptr,
                                                         inputs,
                                                         outputs));
}


void ConvertMulToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators)
{
    TosaMulAttribute mulAttribute(0);
    operators.emplace_back(new TosaSerializationOperator(Op_MUL,
                                                         Attribute_MulAttribute,
                                                         &mulAttribute,
                                                         inputs,
                                                         outputs));
}

void ConvertSubToTosaOperator(const std::vector<string>& inputs,
                              const std::vector<string>& outputs,
                              std::vector<TosaSerializationOperator*>& operators)
{
    operators.emplace_back(new TosaSerializationOperator(Op_SUB,
                                                         Attribute_NONE,
                                                         nullptr,
                                                         inputs,
                                                         outputs));
}

void CalculateRescaleScales(float& input0Scale,
                            float& input1Scale,
                            float& outputScale,
                            const BinaryOperation& operation)
{
    // Correct Rescale values coming from model converter tosa values, which matches TFLite reference outputs.
    auto maxScale = 2.0 * std::max(input0Scale, input1Scale);
    if(operation == armnn::BinaryOperation::Add && outputScale != 0 && maxScale != 0)
    {
        auto inputShift = 20;

        input0Scale = static_cast<float>((input0Scale / maxScale) * (1 << inputShift));
        input1Scale = static_cast<float>((input1Scale / maxScale) * (1 << inputShift));
        outputScale = static_cast<float>(maxScale / (outputScale * (static_cast<float>(1 << inputShift))));
    }
    else if(operation == armnn::BinaryOperation::Mul && outputScale != 0 && maxScale != 0)
    {
        auto inputShift = 1;
        if(input0Scale > input1Scale)
        {
            outputScale = (input0Scale * input1Scale) / outputScale;
            input1Scale = static_cast<float>((input0Scale / maxScale) * (1 << inputShift));
            input0Scale = static_cast<float>((input0Scale / maxScale) * (1 << inputShift));
        }
        else
        {
            outputScale = (input0Scale * input1Scale) / outputScale;
            input0Scale = static_cast<float>((input1Scale / maxScale) * (1 << inputShift));
            input1Scale = static_cast<float>((input1Scale / maxScale) * (1 << inputShift));
        }
    }
    else if(operation == armnn::BinaryOperation::Sub && outputScale != 0 && maxScale != 0)
    {
        auto inputShift = 20;

        input0Scale = static_cast<float>((input0Scale / maxScale) * (1 << inputShift));
        input1Scale = static_cast<float>((input1Scale / maxScale) * (1 << 0));
        outputScale = static_cast<float>(maxScale / (outputScale * (static_cast<float>(1 << inputShift))));
    }
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

    auto input0Name = std::string("input_0");
    auto input1Name = std::string("input_1");
    auto outputName = std::string("output0_");
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
        tensors.emplace_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }
    if(input1Name.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        tensors.emplace_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {}));
    }

    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());

    if (inputDType0 == DType_FP32 ||
        inputDType0 == DType_FP16 ||
        inputDType0 == DType_INT32)
    {
        ConvertSubToTosaOperator({input0Name, input1Name},
                                 {interElemenwiseBinaryName},
                                 operators);

        tensors.emplace_back(new TosaSerializationTensor(interElemenwiseBinaryName,
                                                         outputShape0,
                                                         outputDType0,
                                                         {}));
        ConvertMulToTosaOperator({interElemenwiseBinaryName, interElemenwiseBinaryName},
                                 {outputName},
                                 operators);
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
        operators.emplace_back(xShiftOp);
        tensors.emplace_back(new TosaSerializationTensor(rescale0Output0Name,
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
        operators.emplace_back(yShiftOp);
        tensors.emplace_back(new TosaSerializationTensor(rescale0Output1Name,
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
        operators.emplace_back(xScaledOp);
        tensors.emplace_back(new TosaSerializationTensor(rescale1Output0Name,
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
        operators.emplace_back(yScaledOp);
        tensors.emplace_back(new TosaSerializationTensor(rescale1Output1Name,
                                                         GetTosaTensorShape(inputs[1]->GetShape()),
                                                         DType_INT32,
                                                         {}));

        ConvertSubToTosaOperator({rescale1Output0Name, rescale1Output1Name},
                                 {interElemenwiseBinaryName},
                                 operators);

        tensors.emplace_back(new TosaSerializationTensor(interElemenwiseBinaryName,
                                                         GetTosaTensorShape(outputs[0]->GetShape()),
                                                         DType_INT32,
                                                         {}));

        ConvertMulToTosaOperator({interElemenwiseBinaryName, interElemenwiseBinaryName},
                                 {mulOutputName},
                                 operators);

        tensors.emplace_back(new TosaSerializationTensor(mulOutputName,
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
        operators.emplace_back(rescaleOutputOp);
    }
    else
    {
        throw Exception("TOSA spec only supports INT8, INT32, FP16 and FP32 datatypes for SqDiff.");
    }

    tensors.emplace_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    return new TosaSerializationBasicBlock(blockName, // name
                                           mainName, // region name
                                           {operators}, // operators
                                           tensors, // tensors
                                           {input0Name, input1Name}, // inputs
                                           {outputName}); // outputs
}
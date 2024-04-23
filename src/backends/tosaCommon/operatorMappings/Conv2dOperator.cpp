//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Conv2dOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"
#include <ResolveType.hpp>

TosaSerializationBasicBlock* ConvertConv2dToTosaOperator(const Layer* layer,
                                                         const std::vector<const TensorInfo*>& inputs,
                                                         const std::vector<const TensorInfo*>& outputs,
                                                         const Convolution2dDescriptor* conv2dDescriptor)
{
    std::vector<std::string> inputNames;
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_CONV2D_block_") + GetUniqueTosaMappingID();

    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Set input names for validation purposes only.
    if(layer == nullptr)
    {
        inputNames.emplace_back("input_0");
        inputNames.emplace_back("input_1");
        if(conv2dDescriptor->m_BiasEnabled)
        {
            inputNames.emplace_back("input_2");
        }
    }
    // If a layer is present then the block will be used for execution, so input and output names need to be
    // determined using the previous and following layers so the graph is connected correctly.
    // For validation this doesn't matter.
    else
    {
        // Get the layer connected to the input slot and determine unique tensor names.
        for (uint32_t i = 0; i < inputs.size(); ++i)
        {
            std::string inputName = GenerateUniqueInputName(layer->GetInputSlot(i));
            inputNames.push_back(inputName);
        }

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Setup input Tensor
    // Only add tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensors.
    if(inputNames[0].find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());

        tensors.push_back(new TosaSerializationTensor(inputNames[0], inputShape0, inputDType0, {}));
    }

    // Only add input tensors if weights and bias are not constant or if running validation.
    // Constant tensors will be created in the ConvertConstantToTosaOperator function.
    if(!inputs[1]->IsConstant() || layer == nullptr)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(inputNames[1], inputShape1, inputDType1, {}));
    }

    if(conv2dDescriptor->m_BiasEnabled)
    {
        if(!inputs[2]->IsConstant() || layer == nullptr)
        {
            std::vector<int32_t> inputShape2 = GetTosaTensorShape(inputs[2]->GetShape());
            DType inputDType2 = ArmNNToDType(inputs[2]->GetDataType());

            tensors.push_back(new TosaSerializationTensor(inputNames[2], inputShape2, inputDType2, {}));
        }
    }
    else
    {
        // If bias is disabled, create a constant bias of 0 as three inputs are required.
        std::string constantName = std::string("constant_") + GetUniqueTosaMappingID();

        operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {constantName}));

        // The size of the bias must match the channels dimension, so get the correct index.
        unsigned int index = (conv2dDescriptor->m_DataLayout == DataLayout::NHWC) ? 3 : 1;

        const DType dType = (inputDType0 == DType_INT8) ? DType_INT32 : outputDType0;
        std::vector<float> data(outputs[0]->GetShape()[index], 0);

        std::vector<uint8_t> uint8Data;
        TosaSerializationHandler::ConvertF32toU8(data, uint8Data);

        tensors.push_back(new TosaSerializationTensor(constantName,
                                                      {static_cast<int32_t>(outputs[0]->GetShape()[index])},
                                                      dType,
                                                      uint8Data));
        inputNames.emplace_back(constantName);
    }

    // Setup Output Tensor
    std::vector<int32_t> outputShape0 = {GetTosaTensorShape(outputs[0]->GetShape())};
    std::string outputConv2dName;
    bool isInputInt8 = (inputDType0 == DType_INT8);
    if (isInputInt8)
    {
        outputConv2dName = std::string("intermediate0_") + GetUniqueTosaMappingID();
        tensors.push_back(new TosaSerializationTensor(outputConv2dName, outputShape0, DType_INT32, {}));
    }
    else
    {
        tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));
    }

    // Set up CONV2D operator
    std::vector<int> pad = {static_cast<int>(conv2dDescriptor->m_PadTop),
                            static_cast<int>(conv2dDescriptor->m_PadBottom),
                            static_cast<int>(conv2dDescriptor->m_PadLeft),
                            static_cast<int>(conv2dDescriptor->m_PadRight)};
    std::vector<int> stride = {static_cast<int>(conv2dDescriptor->m_StrideY),
                               static_cast<int>(conv2dDescriptor->m_StrideX)};
    std::vector<int> dilation = {static_cast<int>(conv2dDescriptor->m_DilationY),
                                 static_cast<int>(conv2dDescriptor->m_DilationX)};
    TosaConvAttribute attribute(pad, stride, dilation,
                                inputs[0]->GetQuantizationOffset(), // input_zp
                                inputs[1]->GetQuantizationOffset(), // weight_zp
                                false); // local_bound

    std::string& convOutStr = isInputInt8 ? outputConv2dName : outputName;
    auto* conv2d_op = new TosaSerializationOperator(Op_CONV2D,
                                                    Attribute_ConvAttribute,
                                                    &attribute,
                                                    inputNames,
                                                    {convOutStr});
    operators.push_back(conv2d_op);

    if (isInputInt8)
    {
        int32_t output_zp = outputs[0]->GetQuantizationOffset();
        double output_scale = outputs[0]->GetQuantizationScales()[0];
        double input_scale = inputs[0]->GetQuantizationScales()[0];
        const std::vector<float>& weight_scales = inputs[1]->GetQuantizationScales();

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperatorPerChannel(outputConv2dName,
                                            outputName,
                                            0,
                                            output_zp,
                                            true,
                                            true,
                                            input_scale,
                                            output_scale,
                                            weight_scales,
                                            &rescaleOp);
        operators.push_back(rescaleOp);
        tensors.push_back(new TosaSerializationTensor(outputName,
                                                      outputShape0,
                                                      DType_INT8, {}));
    }

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName,     // name
                                           mainName,      // region name
                                           operators,     // operators
                                           tensors,       // tensors
                                           inputNames,    // inputs
                                           {outputName}); // outputs
}
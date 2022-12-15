//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeConv2dOperator.hpp"

#include "layers/TransposeConvolution2dLayer.hpp"

TosaSerializationBasicBlock* ConvertTransposeConv2dToTosaOperator(const Layer* layer,
                                                                  const std::vector<const TensorInfo*>& inputs,
                                                                  const std::vector<const TensorInfo*>& outputs,
                                                                  const TransposeConvolution2dDescriptor* descriptor)
{
    std::string input0Name = std::string("input0_");
    std::string input1Name = std::string("constant_") + GetUniqueTosaMappingID();
    std::string input2Name = std::string("constant_") + GetUniqueTosaMappingID();
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_TRANSPOSE_CONV2D_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        // Get the layers connected to the input slots and determine unique tensor names.
        Layer& connectedInputLayer = layer->GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();
        input0Name = GenerateUniqueName(connectedInputLayer, 0);

        // Determine unique output tensor name.
        outputName = GenerateUniqueOutputName(*layer, 0);
    }

    std::vector<TosaSerializationTensor*> tensors;
    std::vector<TosaSerializationOperator*> operators;

    // Setup input tensor
    // Only add tensor if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensors.
    if(input0Name.find("input0_") != std::string::npos)
    {
        std::vector<int32_t> inputShape0 = GetTosaTensorShape(inputs[0]->GetShape());
        DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(input0Name, inputShape0, inputDType0, {}));
    }

    // Setup weights tensor, constant data will get copied during SetConstantTensorData
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {input1Name}));

    // During validation the TensorInfo can be retrieved from the inputs.
    // During execution, it is only available through the layer so use m_Weight.
    if(layer == nullptr)
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, {}));
    }
    else
    {
        auto transposeConv2dLayer = PolymorphicDowncast<const TransposeConvolution2dLayer*>(layer);

        std::vector<int32_t> inputShape1 = GetTosaTensorShape(
                transposeConv2dLayer->m_Weight->GetTensorInfo().GetShape());
        DType inputDType1 = ArmNNToDType(transposeConv2dLayer->m_Weight->GetTensorInfo().GetDataType());

        std::vector<uint8_t> uint8Data = ConvertConstantTensorDataToBuffer(transposeConv2dLayer->m_Weight);
        tensors.push_back(new TosaSerializationTensor(input1Name, inputShape1, inputDType1, uint8Data));
    }

    // Setup bias operator and tensor, constant data will get copied during SetConstantTensorData
    operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {input2Name}));

    // During validation the TensorInfo can be retrieved from the inputs.
    // During execution, it is only available through the layer so use m_Bias.
    if(layer == nullptr && descriptor->m_BiasEnabled)
    {
        std::vector<int32_t> inputShape2 = GetTosaTensorShape(inputs[2]->GetShape());
        DType inputDType2 = ArmNNToDType(inputs[2]->GetDataType());

        tensors.push_back(new TosaSerializationTensor(input2Name, inputShape2, inputDType2, {}));
    }
    else if(descriptor->m_BiasEnabled)
    {
        auto transposeConv2dLayer = PolymorphicDowncast<const TransposeConvolution2dLayer*>(layer);

        std::vector<int32_t> inputShape2 = GetTosaTensorShape(
                transposeConv2dLayer->m_Bias->GetTensorInfo().GetShape());
        DType inputDType2 = ArmNNToDType(transposeConv2dLayer->m_Bias->GetTensorInfo().GetDataType());

        std::vector<uint8_t> uint8Data = ConvertConstantTensorDataToBuffer(transposeConv2dLayer->m_Bias);
        tensors.push_back(new TosaSerializationTensor(input2Name, inputShape2, inputDType2, uint8Data));
    }
    else
    {
        // If bias is disabled, create a constant bias tensor of 0's as three inputs are required.
        // The size of the bias must match the channels dimension, so get the correct index.
        unsigned int index = (descriptor->m_DataLayout == DataLayout::NHWC) ? 3 : 1;

        std::vector<uint8_t> uint8Data;
        std::vector<float> data(outputs[0]->GetShape()[index], 0.0f);

        TosaSerializationHandler::ConvertF32toU8(data, uint8Data);

        tensors.push_back(new TosaSerializationTensor(input2Name,
                                                      {static_cast<int32_t>(outputs[0]->GetShape()[index])},
                                                      DType_FP32,
                                                      uint8Data));
    }

    // Setup Output Tensor
    std::vector<int32_t> outputShape0 = GetTosaTensorShape(outputs[0]->GetShape());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));

    // Set up TRANSPOSE_CONV2D operator
    // The TOSA Reference Model pads the output shape, so it is added to output shape.
    // In Arm NN we pad the input shape, so it is taken away.
    // To offset this the negative padding value can be used.
    std::vector<int> pad = {-static_cast<int>(descriptor->m_PadTop),
                            -static_cast<int>(descriptor->m_PadBottom),
                            -static_cast<int>(descriptor->m_PadLeft),
                            -static_cast<int>(descriptor->m_PadRight)};
    std::vector<int> stride = {static_cast<int>(descriptor->m_StrideY),
                               static_cast<int>(descriptor->m_StrideX)};

    std::vector<int> outputShape;
    // If available use shape in descriptor otherwise use output shape.
    if (descriptor->m_OutputShape.size() == 4)
    {
        for (uint32_t i = 0; i < descriptor->m_OutputShape.size(); ++i)
        {
            outputShape.push_back(static_cast<int>(descriptor->m_OutputShape[i]));
        }
    }
    else
    {
        for (uint32_t i = 0; i < outputs[0]->GetNumDimensions(); ++i)
        {
            outputShape.push_back(static_cast<int>(outputs[0]->GetShape()[i]));
        }
    }

    TosaTransposeConvAttribute attribute(pad, stride, outputShape, 0, 0, ArmNNToDType(inputs[0]->GetDataType()));

    auto* op = new TosaSerializationOperator(Op_TRANSPOSE_CONV2D,
                                             Attribute_TransposeConvAttribute,
                                             &attribute,
                                             {input0Name, input1Name, input2Name},
                                             {outputName});
    operators.push_back(op);

    // operatorInputNames/operatorOutputNames ends up being the same as
    // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
    return new TosaSerializationBasicBlock(blockName,                            // name
                                           operators,                            // operators
                                           tensors,                              // tensors
                                           {input0Name, input1Name, input2Name}, // inputs
                                           {outputName});                        // outputs
}
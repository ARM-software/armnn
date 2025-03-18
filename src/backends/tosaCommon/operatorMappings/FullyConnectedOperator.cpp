//
// Copyright © 2024 2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include "FullyConnectedOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"


// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_tfl.cc from function ConvertTFLFullyConnectedOp
TosaSerializationBasicBlock* ConvertFullyConnectedToTosaOperator(const Layer* layer,
                                                                 const std::vector<const TensorInfo*>& inputs,
                                                                 const std::vector<const TensorInfo*>& outputs,
                                                                 const FullyConnectedDescriptor* fcDescriptor)
{
    std::string inputName;
    std::vector<std::string> inputNames;
    std::vector<std::string> fcInputNames;
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_FULLY_CONNECTED_block_") + GetUniqueTosaMappingID();

    DType inputDType0 = ArmNNToDType(inputs[0]->GetDataType());
    DType outputDType0 = ArmNNToDType(outputs[0]->GetDataType());

    // Set input names for validation purposes only.
    if(layer == nullptr)
    {
        inputNames.emplace_back("input_0");
        inputNames.emplace_back("constant_1");
        if(fcDescriptor->m_BiasEnabled)
        {
            inputNames.emplace_back("constant_2");
        }
    }
    // If a layer is present then the block will be used for execution, so input and output names need to be
    // determined using the previous and following layers so the graph is connected correctly.
    // For validation this doesn't matter.
    else
    {
        inputName = GenerateUniqueInputName(layer->GetInputSlot(0));
        inputNames.push_back(inputName);

        inputName = GenerateUniqueInputName(layer->GetInputSlot(1));
        inputNames.push_back(inputName);

        if(fcDescriptor->m_BiasEnabled)
        {
            inputName = GenerateUniqueInputName(layer->GetInputSlot(2));
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
    if(layer == nullptr || (!inputs[1]->IsConstant() && !WeightFromDifferentLayer(*layer)))
    {
        std::vector<int32_t> inputShape1 = GetTosaTensorShape(inputs[1]->GetShape());
        DType inputDType1 = ArmNNToDType(inputs[1]->GetDataType());
        tensors.push_back(new TosaSerializationTensor(inputNames[1], inputShape1, inputDType1, {}));
    }

    if(fcDescriptor->m_BiasEnabled)
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
        inputName = std::string("constant_") + GetUniqueTosaMappingID();
        inputNames.push_back(inputName);

        operators.push_back(new TosaSerializationOperator(Op_CONST, Attribute_NONE, nullptr, {}, {inputName}));

        const DType dType = (inputDType0 == DType_INT8) ? DType_INT32 : outputDType0;
        std::vector<float> data(outputs[0]->GetShape()[1], 0);

        std::vector<uint8_t> uint8Data;
        TosaSerializationHandler::ConvertF32toU8(data, uint8Data);

        tensors.push_back(new TosaSerializationTensor(inputName,
                                                      {static_cast<int32_t>(outputs[0]->GetShape()[1])},
                                                      dType,
                                                      uint8Data));
    }

    fcInputNames = inputNames;

    // Set up Reshape operator. TOSA Fully Connected only accepts 2D rank tensors.
    if (inputs[0]->GetShape().GetNumDimensions() != 2)
    {
        uint32_t num_elems = inputs[1]->GetShape()[1];
        uint32_t num_batch = inputs[0]->GetShape().GetNumElements() / num_elems;

        std::string outputReshapeName = std::string("layer_intermediate0_") + GetUniqueTosaMappingID();
        const std::vector<int32_t>& targetShape = {static_cast<int32_t>(num_batch), static_cast<int32_t>(num_elems)};
        TosaReshapeAttribute attribute(GetTosaTensorShape(TensorShape({num_batch, num_elems})));

        auto* reshapeOp = new TosaSerializationOperator(Op_RESHAPE,
                                                        Attribute_ReshapeAttribute,
                                                        &attribute,
                                                        {inputNames[0]},
                                                        {outputReshapeName});
        operators.push_back(reshapeOp);

        tensors.push_back(new TosaSerializationTensor(outputReshapeName, targetShape, inputDType0, {}));

        fcInputNames[0] = outputReshapeName;
    }


    // Setup Output Tensor
    std::vector<int32_t> outputShape0 = {GetTosaTensorShape(outputs[0]->GetShape())};
    std::string fcOutputName;
    bool isInputInt8 = (inputDType0 == DType_INT8);
    if (isInputInt8)
    {
        fcOutputName = std::string("layer_intermediate0_") + GetUniqueTosaMappingID();
        tensors.push_back(new TosaSerializationTensor(fcOutputName, outputShape0, DType_INT32, {}));
    }
    else
    {
        tensors.push_back(new TosaSerializationTensor(outputName, outputShape0, outputDType0, {}));
    }

    // Set up Fully Connected operator
    TosaFullyConnectedAttribute attribute(inputs[0]->GetQuantizationOffset(),  // input_zp
                                          inputs[1]->GetQuantizationOffset()); // weight_zp

    std::string& fcOutStr = isInputInt8 ? fcOutputName : outputName;
    auto* fullyConnected_op = new TosaSerializationOperator(Op_FULLY_CONNECTED,
                                                            Attribute_FullyConnectedAttribute,
                                                            &attribute,
                                                            fcInputNames,
                                                            {fcOutStr});
    operators.push_back(fullyConnected_op);

    if (isInputInt8)
    {
        int32_t output_zp = outputs[0]->GetQuantizationOffset();
        double output_scale = outputs[0]->GetQuantizationScales()[0];
        double input_scale = inputs[0]->GetQuantizationScales()[0];
        const std::vector<float>& weight_scales = inputs[1]->GetQuantizationScales();

        TosaSerializationOperator* rescaleOp = nullptr;
        CreateRescaleTosaOperatorForWeights(fcOutputName,
                                            outputName,
                                            0,
                                            output_zp,
                                            false,
                                            false,
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

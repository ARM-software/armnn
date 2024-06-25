//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TosaTestUtils.hpp"

using namespace armnn;
using namespace tosa;

void VerifyFullyConnected(TosaSerializationBasicBlock* fcBlock,
                          const std::vector<std::vector<int32_t>>& inputShape,
                          const std::vector<std::vector<int32_t>>& outputShape,
                          const BaseDescriptor& descriptor,
                          DType dataType = DType_FP32)
{
    uint32_t numInputs = static_cast<uint32_t>(inputShape.size());
    uint32_t numOutputs = static_cast<uint32_t>(outputShape.size());
    uint32_t numInputTensors = PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor)->m_BiasEnabled ? 3 : 2;

    std::string blockStr = "Op_FULLY_CONNECTED_block_";
    CHECK(fcBlock->GetName().find(blockStr)  != std::string::npos);
    CHECK(fcBlock->GetInputs().size() == numInputTensors);
    CHECK(fcBlock->GetOutputs().size() == numOutputs);
    CHECK(fcBlock->GetOperators().size() == 2);
    CHECK(fcBlock->GetTensors().size() == (numInputs + numOutputs + 1));

    //
    // Verify Reshape operator
    //

    TosaSerializationOperator* reshapeOp = fcBlock->GetOperators().at(0);
    CHECK(reshapeOp->GetAttributeType() == tosa::Attribute_ReshapeAttribute);
    CHECK(reshapeOp->GetOp() == tosa::Op_RESHAPE);

    //
    // Verify FullyConnected operator
    //

    TosaSerializationOperator* fullyConnectedOp = fcBlock->GetOperators().at(1);
    CHECK(fullyConnectedOp->GetAttributeType() == tosa::Attribute_FullyConnectedAttribute);
    CHECK(fullyConnectedOp->GetOp() == tosa::Op_FULLY_CONNECTED);

    // Inputs
    CHECK(fullyConnectedOp->GetInputTensorNames().size() == numInputTensors);

    // - Input
    std::basic_string<char> blockInputName = fcBlock->GetInputs()[0];
    std::basic_string<char> operatorInputName = reshapeOp->GetInputTensorNames()[0];
    std::basic_string<char> inputTensorName = fcBlock->GetTensors()[0]->GetName();

    CHECK(blockInputName == operatorInputName);
    CHECK(inputTensorName == operatorInputName);
    CHECK(blockInputName.find("input_")  != std::string::npos);

    TosaSerializationTensor* inputTensor = fcBlock->GetTensorByName(operatorInputName);
    CHECK(inputTensor->GetDtype() == dataType);
    CHECK(inputTensor->GetData().size() == 0);
    CHECK(inputTensor->GetShape() == inputShape[0]);

    // - Weights
    std::basic_string<char> blockWeightsName = fcBlock->GetInputs()[1];
    std::basic_string<char> operatorWeightsName = fullyConnectedOp->GetInputTensorNames()[1];

    CHECK(blockWeightsName == operatorWeightsName);
    CHECK(blockWeightsName.find("constant_")  != std::string::npos);

    // - Bias
    if (PolymorphicDowncast<const FullyConnectedDescriptor*>(&descriptor)->m_BiasEnabled)
    {
        std::basic_string<char> blockBiasName = fcBlock->GetInputs()[2];
        std::basic_string<char> operatorBiasName = fullyConnectedOp->GetInputTensorNames()[2];

        CHECK(blockBiasName == operatorBiasName);
        CHECK(blockBiasName.find("constant_")  != std::string::npos);
    }


    // Outputs
    CHECK(fullyConnectedOp->GetOutputTensorNames().size() == numOutputs);

    std::basic_string<char> blockOutputName = fcBlock->GetOutputs()[0];
    std::basic_string<char> operatorOutputName = fullyConnectedOp->GetOutputTensorNames()[0];
    std::basic_string<char> outputTensorName = fcBlock->GetTensors()[numInputs+1]->GetName();

    CHECK(blockOutputName == operatorOutputName);
    CHECK(outputTensorName == operatorOutputName);
    CHECK(blockOutputName.find("output0_")  != std::string::npos);

    TosaSerializationTensor* outputTensor = fcBlock->GetTensorByName(operatorOutputName);
    CHECK(outputTensor->GetDtype() == dataType);
    CHECK(outputTensor->GetData().size() == 0);
    CHECK(outputTensor->GetShape() == outputShape[0]);

    CHECK(blockOutputName == operatorOutputName);

    // Verify Attribute
    TosaFullyConnectedAttribute attribute = fullyConnectedOp->GetAttribute();
    CHECK( 0 == attribute.weight_zp());
    CHECK( 0 == attribute.input_zp());
}